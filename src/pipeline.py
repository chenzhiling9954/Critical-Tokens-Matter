import argparse
import json
import math
import os
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


from configs.data_config import get_model_sampling_config, DATASET_PATH, get_data_selection_config
from configs.train_config import get_sft_config, get_cdpo_config
from train.train_cdpo import cdpo_model_with_lora
from utils.sampling import vllm_model_sampling, load_vllm_model, load_model
from utils.utils import load_jsonl, save_merged_model, multi_result_check
from train.train_sft import sft_model_with_lora





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str,
                        choices=["Meta-Llama-3-8B", "deepseek-math-7b-base", "Meta-Llama-3-70B"])
    parser.add_argument("--dataset_name", type=str, choices=["GSM8K", "MATH"])
    parser.add_argument("--task_name", type=str,
                        choices=["train_ce", "train_cdpo", "prepare_startup_data", "calculate_ce_score", "evaluation"])
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)

    args = parser.parse_args()

    return args

class Pipeline:
    def __init__(self, model_name, dataset_name, gpus):
        self.model_name = model_name
        self.gpus = gpus
        self.dataset_name = dataset_name
        self.dataset_path = DATASET_PATH[dataset_name]

        self.sampling_config = get_model_sampling_config(model_name=model_name, dataset_name=dataset_name)
        self.data_selection_config = get_data_selection_config()
        self.sft_config = get_sft_config(model_name=model_name)
        self.cdpo_config = get_cdpo_config(model_name=model_name)

        name_for_sft_selection = self.data_selection_config["sft_pos"]["name"] + "_" + self.data_selection_config["sft_neg"]["name"]
        name_for_dpo_selection = self.data_selection_config["dpo"]["name"]
        temperature = self.sampling_config['temperature']
        self.sft_data_path = f"./data/{model_name}/{dataset_name}.{name_for_sft_selection}.T{temperature:.2f}.json"
        self.cdpo_data_path = f"./data/{model_name}/{dataset_name}.{name_for_dpo_selection}.T{temperature:.2f}.json"
        self.cdpo_data_with_ce_score_path = f"./data/{model_name}/{dataset_name}.{name_for_dpo_selection}.T{temperature:.2f}.CE_scored.json"

        self.sft_ckpt_path = f"./checkpoints/{self.model_name}/{dataset_name}/sft"
        self.cdpo_ckpt_path = f"./checkpoints/{self.model_name}/{dataset_name}/cdpo"

    def select_data_for_sft_and_dpo(self, checked_result, data_selection_config):
        if not os.path.exists(self.sft_data_path) and not os.path.exists(self.cdpo_data_path):
            print(f"Selection data save path: \nSFT: {self.sft_data_path}\nDPO: {self.cdpo_data_path}")
            sft_data = {'config': data_selection_config, "pos": {}, "neg": {}}
            dpo_data = {'config': data_selection_config, "prompt": [], "chosen": [], "rejected": [], "answer": []}
            ava_false_ans_count = 0
            for q_id in checked_result:
                line = checked_result[q_id]
                responses_with_check = line["generation_check"]
                num_error_response = sum(
                    [1 if line[2] == False and line[1] != "" else 0 for line in responses_with_check])
                has_add_true = False
                dpo_chosen = None
                dpo_rejected = None
                need_add_false = math.ceil(data_selection_config["sft_neg"]["top-p"] * num_error_response)
                add_false_count = 0
                random.shuffle(responses_with_check)
                answer_sorted_dict = {}
                for response in responses_with_check:
                    answer = response[1]
                    if answer in answer_sorted_dict:
                        answer_sorted_dict[answer].append(response)
                    else:
                        answer_sorted_dict[answer] = [response]
                answer_count_list = [(a, len(answer_sorted_dict[a])) for a in answer_sorted_dict.keys()]
                answer_count_list = sorted(answer_count_list, key=lambda x: x[1], reverse=True)
                for a, count in answer_count_list:
                    if a == "":
                        continue
                    response = random.sample(answer_sorted_dict[a], k=1)[0]
                    clean_response = response[0].strip()
                    if not has_add_true and response[2] == True:
                        sft_data["pos"][q_id] = {"question": line["question"],
                                                 "response": clean_response,
                                                 "answer": line["answer"][0]}
                        has_add_true = True
                    if need_add_false > 0 and response[2] == False:
                        sft_data["neg"][f"{q_id}_{add_false_count}"] = {"question": line["question"],
                                                                        "response": clean_response,
                                                                        "answer": line["answer"][0]}

                        need_add_false -= count
                        add_false_count += 1
                    if dpo_chosen is None and response[2] == True:
                        dpo_chosen = clean_response
                    if dpo_rejected is None and response[2] == False:
                        dpo_rejected = clean_response
                    if response[2] == False:
                        ava_false_ans_count += 1
                if dpo_chosen is not None and dpo_rejected is not None:
                    dpo_data["prompt"].append(line["prompt"])
                    dpo_data["answer"].append(line["answer"])
                    dpo_data["chosen"].append(dpo_chosen)
                    dpo_data["rejected"].append(dpo_rejected)
            print(f"num of data in true: {len(sft_data['pos'])}, num of data in false: {len(sft_data['neg'])}")
            with open(self.sft_data_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(sft_data, indent=2))
            with open(self.cdpo_data_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(dpo_data, indent=2))
        else:
            print(f"Existing result found in {self.sft_data_path}\nand {self.cdpo_data_path}")

    def prepare_startup_data(self):
        # ##### sampling ##### #
        save_path = f"./data/{model_name}/{dataset_name}_train_sampling.T{self.sampling_config['temperature']:.2f}.finished.jsonl"
        print("-" * 50)
        print("Sampling startup data: \n"
              f">>>>> model: {model_name}\n"
              f">>>>> dataset: {dataset_name}\n"
              f">>>>> temperature: {self.sampling_config['temperature']:.2f}")
        if os.path.exists(save_path):
            print(f"Existing result found in {save_path}")
        else:
            llm = load_vllm_model(model_path=self.sampling_config["model_path"]["base"], gpus=self.gpus)
            tokenizer = AutoTokenizer.from_pretrained(self.sampling_config["model_path"]["base"])
            dataset = list(load_jsonl(self.dataset_path["train"]))
            for line in dataset:
                if dataset_name == "GSM8K":
                    line["prompt"] = "Question: " + line["question"] + "\nAnswer: "
                elif dataset_name == "MATH":
                    line["prompt"] = "Problem:\n" + line["question"] + "\n\nSolution:\n"
                else:
                    raise Exception(f"Unsupported dataset: {dataset_name}")

            vllm_model_sampling(llm=llm,
                                dataset=dataset,
                                prompt_key="prompt",
                                tokenizer=tokenizer,
                                temperature=self.sampling_config["temperature"],
                                save_path=save_path,
                                dataset_name=dataset_name,
                                sampling_n=self.sampling_config["with_sampling_n"],
                                stop=self.sampling_config["stop"])
        print("-" * 50)


        print("-" * 50)
        print("Checking sampling result")
        result_list = list(load_jsonl(save_path))
        checked_result_path = multi_result_check(file_path=save_path, result_list=result_list,
                                                 dataset_name=dataset_name, save_id_with_qid_count=False)
        print("-" * 50)


        print("-" * 50)
        print("Select data for sft and dpo")

        with open(checked_result_path, "r", encoding="utf-8") as f:
            checked_result = json.load(f)
        self.select_data_for_sft_and_dpo(checked_result=checked_result,
                                         data_selection_config=self.data_selection_config)
        print("-" * 50)

    def calculate_ce_score(self):
        with open(self.cdpo_data_path, "r", encoding="utf-8") as f:
            preference_data = json.load(f)
        data_with_cd_score = {"prompt": [], "chosen": [], "rejected": [], "prob_result": []}
        pos_model, tokenizer = load_model(model_name=self.sft_ckpt_path + "/pos")
        neg_model, _ = load_model(model_name=self.sft_ckpt_path + "/neg")
        pos_model.eval()
        neg_model.eval()

        count = 0
        with torch.no_grad():
            for question, pos_response, neg_response in tqdm(zip(preference_data["prompt"],
                                                                 preference_data["chosen"],
                                                                 preference_data["rejected"]),
                                                             desc="calculate cd score: "):
                data_with_cd_score["prompt"].append(question)
                data_with_cd_score["chosen"].append(pos_response)
                data_with_cd_score["rejected"].append(neg_response)
                prob_result = {"pos": {}, "neg": {}}
                text_by_type = {"pos": question + pos_response,
                        "neg": question + neg_response}
                for key in prob_result.keys():
                    text = text_by_type[key]
                    input_ids = tokenizer(text, return_tensors="pt").input_ids.squeeze()
                    question_length = len(tokenizer.encode(question))
                    response_start_idx = 0
                    previous_ids = []
                    for idx, id in enumerate(input_ids):
                        previous_ids.append(int(id))
                        if len(previous_ids) >= question_length - 5:
                            previous_text = tokenizer.decode(previous_ids)
                            if question in previous_text:
                                response_start_idx = idx
                                break

                    pos_outputs = pos_model(input_ids.unsqueeze(0).to("cuda"), return_dict=True)
                    pos_logits = pos_outputs.logits.to("cpu")
                    neg_outputs = neg_model(input_ids.unsqueeze(0).to("cuda"), return_dict=True)
                    neg_logits = neg_outputs.logits.to("cpu")
                    cd_logits = (1 + 1) * pos_logits - 1 * neg_logits
                    cd_probs = cd_logits.softmax(dim=-1)

                    expanded_input_ids = input_ids.unsqueeze(0).unsqueeze(-1)[:, 1:, :]
                    cd_selected_probs = torch.gather(cd_probs, 2, expanded_input_ids).squeeze()
                    input_ids_list = input_ids.tolist()[1:]
                    prob_result[key]["ce_prob"] = cd_selected_probs.tolist()
                    prob_result[key]["response_start_idx"] = response_start_idx
                    prob_result[key]["text"] = tokenizer.decode(input_ids_list[response_start_idx:])
                data_with_cd_score["prob_result"].append(prob_result)
                count += 1
                if count % 50 == 0:
                    with open(self.cdpo_data_with_ce_score_path.replace(".json", "unfinished.json"), "w",
                              encoding="utf-8") as f:
                        f.write(json.dumps(data_with_cd_score, indent=2))
            print(f"preference data with ce score saved at: {self.cdpo_data_with_ce_score_path}")
            with open(self.cdpo_data_with_ce_score_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(data_with_cd_score, indent=2))
            del pos_model, neg_model
            torch.cuda.empty_cache()



    def start_ce_train_pipeline(self):
        if not os.path.exists(self.sft_data_path):
            print("-" * 50)
            print("Preparing startup data")
            self.prepare_startup_data()
            print("-" * 50)
        print("=" * 50)
        for type in ["pos", "neg"]:
            sft_model_with_lora(model_path=self.sft_config["model_path"]["base"],
                                model_name=self.model_name,
                                dataset_path=self.sft_data_path,
                                dataset_name=self.dataset_name,
                                train_type=type,
                                lora_path=self.sft_ckpt_path + "/" + type,
                                hyperparameters=self.sft_config["hyperparameters"])
        print("=" * 50)

    def start_cdpo_train_pipeline(self):
        if not os.path.exists(self.cdpo_data_with_ce_score_path):
            print("-" * 50)
            print("Calculate CE score")
            self.calculate_ce_score()
            print("-" * 50)
        cdpo_model_with_lora(model_path=self.cdpo_config["model_path"]["base"], lora_path=self.cdpo_ckpt_path,
                             dpo_data_with_ce_score_path=self.cdpo_data_with_ce_score_path,
                             hyperparameters=self.cdpo_config["hyperparameters"])

    def evaluation(self, lora_path, model_path):
        print("=" * 50)
        # vLLM does not support PeftModel directly, so we need to merge the LoRA weights and save the model.
        if model_path is None:
            model_path = lora_path.replace("/checkpoints/", "/checkpoints/merged_model/")
            save_merged_model(base_model=self.cdpo_config["model_path"]["base"], lora_path=lora_path, save_path=model_path)
        print(f">>>>> Evaluate {model_path}")
        save_path = model_path.replace("/checkpoints", "/output")
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(model_path.replace("/checkpoints", "/output"), "result.jsonl")
        if not os.path.exists(save_path):
            print(f">>>>> Result will be saved to {save_path}")

            llm = load_vllm_model(model_path=model_path, gpus=self.gpus)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            dataset = list(load_jsonl(self.dataset_path["test"]))
            for line in dataset:
                if self.dataset_name == "GSM8K":
                    line["prompt"] = "Question: " + line["question"] + "\nAnswer: "
                elif self.dataset_name == "MATH":
                    line["prompt"] = "Problem:\n" + line["question"] + "\n\nSolution:\n"
                else:
                    raise Exception(f"Unsupported dataset: {self.dataset_name}")
            vllm_model_sampling(llm=llm,
                                dataset=dataset,
                                prompt_key="prompt",
                                tokenizer=tokenizer,
                                temperature=0,
                                save_path=save_path,
                                dataset_name=dataset_name,
                                sampling_n=1,
                                stop=self.sampling_config["stop"])

        result_list = list(load_jsonl(save_path))
        checked_result_file_path = multi_result_check(file_path=save_path, result_list=result_list,
                                    dataset_name=dataset_name, save_id_with_qid_count=True)
        with open(checked_result_file_path, "r", encoding="utf-8") as f:
            checked_result = json.load(f)

        acc = 0.0
        for id in checked_result:
            line = checked_result[id]
            if line["generation_check"][0][2]:
                acc += 1

        print(f">>>>> {model_path}")
        print(f">>>>> Accuracy: {acc / len(checked_result) * 100:.2f}%")











if __name__ == '__main__':
    args = parse_args()
    task_name = args.task_name
    dataset_name = args.dataset_name
    model_name = args.model_name
    gpus = args.gpus

    pipeline = Pipeline(model_name=model_name,
                        dataset_name=dataset_name,
                        gpus=gpus)
    if task_name == "train_ce":
        pipeline.start_ce_train_pipeline()
    elif task_name == "train_cdpo":
        pipeline.start_cdpo_train_pipeline()
    elif task_name == "prepare_startup_data":
        pipeline.prepare_startup_data()
    elif task_name == "evaluation":
        pipeline.evaluation(lora_path=args.lora_path, model_path=args.model_path)
