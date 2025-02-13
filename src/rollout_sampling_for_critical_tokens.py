import argparse
import os

import random
from configs.prompt import MATH_PROMPT, GSM8K_PROMPT
from tqdm import tqdm
import json
from math import comb
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F

from configs.data_config import get_model_sampling_config

from utils.sampling import load_model, load_vllm_model, vllm_model_sampling
from utils.utils import multi_result_check

random.seed(33)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str,
                        choices=["Meta-Llama-3-8B", "deepseek-math-7b-base", "Meta-Llama-3-70B"])
    parser.add_argument("--dataset_name", type=str, choices=["GSM8K", "MATH"])
    parser.add_argument("--task_name", type=str, choices=["sampling", "analyse"])
    parser.add_argument("--cdpo_data_path", type=str, help="datasource for rollout sampling")
    parser.add_argument("--gpus", type=str, default=None)

    args = parser.parse_args()
    return args


def dpo_format_to_score_format(model_name, dataset_name, cdpo_data_path):
    if not os.path.exists(f"./data/rollout_sampling/{model_name}/{dataset_name}_train_data_for_sampling.json"):
        with open(cdpo_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        target_data = {}

        for idx, (prompt, rejected, answer) in enumerate(zip(data["prompt"],
                                                                          data["rejected"],
                                                                          data["answer"])):
            q_id = f"#{idx}#"

            target_data[q_id] = {
                "id": q_id,
                "question": prompt,
                "answer": answer,
                "generation_result": [rejected],
            }
            if len(target_data) == 100:
                break
        with open(f"./data/rollout_sampling/{model_name}/{dataset_name}_train_data_for_sampling.json", "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(target_data, indent=2))





def get_final_prompt(dataset, dataset_name):
    inputs = []
    if dataset_name == "GSM8K":
        few_shot = GSM8K_PROMPT
        response_start_str = "Answer:"
    elif dataset_name == "MATH":
        few_shot = MATH_PROMPT
        response_start_str = "Solution:"
    for sample in dataset:
        final_prompt = few_shot + sample["question"]
        inputs.append(final_prompt)

    return inputs, few_shot, response_start_str


def generate_with_and_without_for_sampling(model_name, dataset_name, model_path):
    path = f"./data/rollout_sampling/{model_name}/{dataset_name}_train_data_for_sampling.json"
    print(f">>>>> open file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = []
    for q_id in data:
        line = data[q_id]
        question_and_response = line['question'] + line["generation_result"][0]
        dataset.append({"id": q_id,
                        "question": question_and_response,
                        "answer": line["answer"],
                        "response_label": False})

    with_token_save_path = f"./data/rollout_sampling/{model_name}/{dataset_name}_with_token_on_train_set.jsonl"
    if os.path.exists(with_token_save_path):
        print(">>>>> Existing result in file: ", with_token_save_path)
        model = None
        tokenizer = None
    else:
        model, tokenizer = load_model(model_path["base"])
        inputs, few_shot, response_start_str = get_final_prompt(dataset, dataset_name)
        inputs = [tokenizer(i, return_tensors="pt").input_ids.to("cpu") for i in inputs]
        few_shot_length = len(tokenizer.encode(few_shot))

        print(">>>>> save path: ", with_token_save_path)
        with_token_list = []

        for input_ids, sample in tqdm(list(zip(inputs, dataset)), desc="Prepare with data: "):
            response_start_idx = 0
            step_ids = []
            for idx, id in enumerate(input_ids.squeeze()):
                step_ids.append(int(id))
                if len(step_ids) >= few_shot_length - 5:
                    step_text = tokenizer.decode(step_ids)
                    if few_shot in step_text and response_start_str in step_text.replace(few_shot, ""):
                        response_start_idx += 1
                        break
                response_start_idx += 1
            for i in range(response_start_idx + 1, len(input_ids[0])):
                sample["question"] = tokenizer.decode(input_ids[0][:i], skip_special_tokens=True).replace(few_shot, "")
                with_token_list.append({**sample})
        with open(with_token_save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(with_token_list, indent=2))

    without_token_save_path = f"./data/rollout_sampling/{model_name}/{dataset_name}_without_token_on_train_set.jsonl"
    if os.path.exists(without_token_save_path):
        print(">>>>> Existing result in file: ", without_token_save_path)
    else:
        if model == None:
            model, tokenizer = load_model(model_path["base"])
            inputs, few_shot, response_start_str = get_final_prompt(dataset, dataset_name)
            inputs = [tokenizer(i, return_tensors="pt").input_ids.to("cpu") for i in inputs]
            few_shot_length = len(tokenizer.encode(few_shot))

        print(">>>>> save path: ", without_token_save_path)
        without_token_list = []
        with (torch.no_grad()):
            for input_ids, sample in tqdm(list(zip(inputs, dataset)), desc="Prepare without data: "):
                outputs = model(
                    input_ids.to("cuda"),
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=True
                )
                probs = F.softmax(outputs.logits, dim=-1)[:, :, :].squeeze().to("cpu")
                for i, token_id in enumerate(input_ids.squeeze()):
                    if i == 0:
                        continue
                    probs[i - 1, token_id] = 0
                response_start_idx = 0
                step_ids = []
                for idx, id in enumerate(input_ids.squeeze()):
                    step_ids.append(int(id))
                    if len(step_ids) >= few_shot_length - 5:
                        step_text = tokenizer.decode(step_ids)
                        if few_shot in step_text and response_start_str in step_text.replace(few_shot, ""):
                            response_start_idx += 1
                            break
                    response_start_idx += 1
                replace_token = torch.multinomial(probs[response_start_idx - 1:, :], num_samples=64, replacement=True)[
                                :-1, :]
                for replace_idx in range(replace_token.shape[0]):
                    for token_id in replace_token[replace_idx]:
                        sample["question"] = tokenizer.decode(
                            input_ids.squeeze().tolist()[:response_start_idx + replace_idx] + [token_id.tolist()],
                            skip_special_tokens=True).replace(few_shot, "")
                        without_token_list.append({
                            "replace_token": {
                                "from": tokenizer.decode(input_ids.squeeze().tolist()[response_start_idx + replace_idx],
                                                         remove_special_tokens=True),
                                "to": tokenizer.decode([token_id.tolist()], remove_special_tokens=True)},
                            **sample
                        })
        with open(without_token_save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(without_token_list, indent=2))
    return with_token_save_path, without_token_save_path


def rollout_sampling(model_name, dataset_name, cdpo_data_path, sampling_config):
    print("=" * 50)
    print("Rollout Sampling")

    print("=" * 50)
    print("Current Stage: Prepare data")
    print("=" * 50)
    dpo_format_to_score_format(model_name=model_name,
                               dataset_name=dataset_name,
                               cdpo_data_path=cdpo_data_path)
    with_token_save_path, without_token_save_path = (
        generate_with_and_without_for_sampling(model_name=model_name,
                                               dataset_name=dataset_name,
                                               model_path=sampling_config["model_path"]))
    print("=" * 50)
    print("Current Stage: Sampling on all token")
    print("=" * 50)
    llm = load_vllm_model(model_path=sampling_config["model_path"]["base"], gpus=gpus)
    tokenizer = AutoTokenizer.from_pretrained(sampling_config["model_path"]["base"])
    with open(with_token_save_path, "r", encoding="utf-8") as f:
        with_token_data = json.load(f)
    with open(without_token_save_path, "r", encoding="utf-8") as f:
        without_token_data = json.load(f)
    save_path = with_token_save_path.replace(".jsonl", "_result.jsonl")
    with_token_result_path = vllm_model_sampling(llm=llm,
                                                 dataset=with_token_data,
                                                 prompt_key="question",
                                                 tokenizer=tokenizer,
                                                 temperature=sampling_config["temperature"],
                                                 save_path=save_path,
                                                 dataset_name=dataset_name,
                                                 sampling_n=sampling_config["with_sampling_n"],
                                                 stop=sampling_config["stop"])

    save_path = without_token_save_path.replace(".jsonl", "_result.jsonl")
    without_token_result_path = vllm_model_sampling(llm=llm,
                                                    dataset=without_token_data,
                                                    prompt_key="question",
                                                    tokenizer=tokenizer,
                                                    temperature=sampling_config["temperature"],
                                                    save_path=save_path,
                                                    dataset_name=dataset_name,
                                                    sampling_n=sampling_config["without_sampling_n"],
                                                    stop=sampling_config["stop"])
    print("=" * 50)
    print("Current Stage: Check result")
    print("=" * 50)
    if dataset_name == "GSM8K":
        few_shot_prompt = GSM8K_PROMPT
    elif dataset_name == "MATH":
        few_shot_prompt = MATH_PROMPT
    with_token_result = []
    with open(with_token_result_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            ori_line = json.loads(line)
            ori_line["question"] = ori_line["question"].replace(few_shot_prompt, "")
            with_token_result.append(ori_line)
    multi_result_check(file_path=with_token_result_path, result_list=with_token_result,
                       dataset_name=dataset_name, save_id_with_qid_count=True)

    without_token_result = []
    with open(without_token_result_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            ori_line = json.loads(line)
            ori_line["question"] = ori_line["question"].replace(few_shot_prompt, "")
            without_token_result.append(ori_line)
    multi_result_check(file_path=without_token_result_path, result_list=without_token_result,
                       dataset_name=dataset_name, save_id_with_qid_count=True)


def get_pass_at_k(result_dict, max_k):
    pass_at_k = {}
    for k in range(1, max_k + 1):
        correct_count = 0
        for q_id, data in result_dict.items():
            prefix = list(data.keys())[0]
            labels = [entry[-1] for entry in data[prefix]]
            if len(labels) == 0:
                continue
            n = len(labels)
            c = sum(labels)
            k_limit = min(k, n)
            if c == 0:
                pass_prob = 0.0
            else:
                incorrect_combinations = comb(n - c, k_limit) if k_limit <= (n - c) else 0
                total_combinations = comb(n, k_limit)
                pass_prob = 1 - (incorrect_combinations / total_combinations)
            correct_count += pass_prob
        pass_rate = correct_count / len(result_dict)
        pass_at_k[k] = pass_rate
    return pass_at_k


def analyse_all_token(model_name, dataset_name, tokenizer, max_k):
    with_token_data_path = f"./data/rollout_sampling/{model_name}/{dataset_name}_with_token_on_train_set_checked_result.json"
    without_token_data_path = f"./data/rollout_sampling/{model_name}/{dataset_name}_without_token_on_train_set_checked_result.json"

    with open(without_token_data_path, "r", encoding="utf-8") as f:
        without_token_data = json.load(f)
    with open(with_token_data_path, "r", encoding="utf-8") as f:
        with_token_data = json.load(f)

    with_token_dict_by_qid = {}
    for q_id in with_token_data:
        orginal_q_id = q_id.split("_")[0]
        if orginal_q_id not in with_token_dict_by_qid:
            with_token_dict_by_qid[orginal_q_id] = {}
        prefix = with_token_data[q_id]["question"]
        if prefix not in with_token_dict_by_qid[orginal_q_id].keys():
            with_token_dict_by_qid[orginal_q_id][prefix] = with_token_data[q_id]["generation_check"]
    del with_token_data

    without_token_dict_by_qid = {}
    for q_id in tqdm(without_token_data):
        without_token_line = without_token_data[q_id]
        from_token = without_token_line["replace_token"]["from"]
        to_token = without_token_line["replace_token"]["to"]
        prefix = without_token_line["question"][:-len(to_token)] + from_token
        assert without_token_line["question"][-len(to_token):] == to_token or to_token in tokenizer.all_special_tokens
        orginal_q_id = q_id.split("_")[0]
        if orginal_q_id not in without_token_dict_by_qid:
            without_token_dict_by_qid[orginal_q_id] = {}
        if prefix not in without_token_dict_by_qid[orginal_q_id]:
            without_token_dict_by_qid[orginal_q_id][prefix] = []
        without_token_dict_by_qid[orginal_q_id][prefix].append(without_token_line["generation_check"][0])
    del without_token_data

    min_prefix_result_dict = {}
    for q_id in without_token_dict_by_qid:
        without_token_line = without_token_dict_by_qid[q_id]
        prefix_list = sorted(list(without_token_line.keys()), key=lambda x: len(x), reverse=False)
        min_predix = ""
        last_min_predix = None
        has_find = False
        for prefix in prefix_list:
            if prefix in with_token_dict_by_qid[q_id]:
                orginal_true_count = sum(1 for _, _, label in with_token_dict_by_qid[q_id][prefix] if label)
                if orginal_true_count == 0 and not has_find:
                    min_predix = prefix
                    has_find = True

                if has_find and orginal_true_count > 3:
                    last_min_predix = min_predix
                    min_predix = ""
                    has_find = False
        if has_find:
            min_prefix_result_dict[q_id] = {min_predix: without_token_dict_by_qid[q_id][min_predix]}
        elif last_min_predix is not None:
            min_prefix_result_dict[q_id] = {last_min_predix: without_token_dict_by_qid[q_id][last_min_predix]}
        else:
            print("not find")

    max_k = 64
    pass_at_k = get_pass_at_k(min_prefix_result_dict, max_k)
    for k in pass_at_k:
        pass_at_k[k] = str(round(pass_at_k[k] * 100, 2)) + "%"
    print("=" * 50)
    print(f"Model: {model_name}\nDataset: {dataset_name}")
    print(f"Pass at K: {pass_at_k}")
    print("=" * 50)


if __name__ == '__main__':
    args = parse_args()
    task_name = args.task_name
    dataset_name = args.dataset_name
    model_name = args.model_name
    cdpo_data_path = args.cdpo_data_path
    sampling_config = get_model_sampling_config(model_name=model_name, dataset_name=dataset_name)
    if task_name == "sampling":
        gpus = args.gpus
        rollout_sampling(model_name=model_name, dataset_name=dataset_name,
                         cdpo_data_path=cdpo_data_path, sampling_config=sampling_config)
    elif task_name == "analyse":
        tokenizer = AutoTokenizer.from_pretrained(sampling_config["model_path"]["base"])
        analyse_all_token(model_name=model_name, dataset_name=dataset_name, tokenizer=tokenizer, max_k=sampling_config["with_sampling_n"])
