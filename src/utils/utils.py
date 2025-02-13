import json
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm

from utils.parser import extract_answer_str_by_answer_pattern, extract_answer_by_question_source
from utils.grader import math_equal

def load_jsonl(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_merged_model(base_model, lora_path, save_path):
    print(f"Saving merged model in {save_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    lora_model = PeftModel.from_pretrained(
        model=model,
        model_id=lora_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    merge_model = lora_model.merge_and_unload()
    merge_model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(save_path)


def multi_result_check(file_path, result_list, dataset_name, save_id_with_qid_count):
    dataset_name = dataset_name.upper()
    checked_result_file_path = (file_path.replace(".jsonl", ".checked.json"))
    print(">>>>> Check result saved at: ", checked_result_file_path)
    if os.path.exists(checked_result_file_path):
        with open(checked_result_file_path, "r", encoding="utf-8") as f:
            result_by_same_id = json.load(f)
    else:
        result_by_same_id = {}
    id_count = defaultdict(int)
    save_count = -1
    result_already_check_dict = {}
    for sample in tqdm(result_list, "check result: "):
        q_id = str(sample["id"])
        if save_id_with_qid_count:
            final_id = f"{q_id}_{id_count[q_id]}"
        else:
            final_id = q_id
        if final_id in result_by_same_id:
            continue
        result_by_same_id[final_id] = {**sample}
        result_by_same_id[final_id][f"generation_check"] = []
        golden_answers = sample["answer"]
        pred_str_list = sample[f"generation_result"]
        if isinstance(pred_str_list, str):
            pred_str_list = [pred_str_list]
        for pred_str in pred_str_list:
            answer_pattern = ""
            answer_str = extract_answer_str_by_answer_pattern(pred_str=pred_str, answer_pattern=answer_pattern)
            answer_str = answer_str.split("\n\nQuestion:")[0].split("\n\nProblem:")[0]
            clean_pred = extract_answer_by_question_source(pred_str=answer_str, question_source=dataset_name)
            label = False
            for g in golden_answers:
                if isinstance(g, str) and dataset_name != "MATH":
                    g = g.lower()
                if isinstance(clean_pred, str) and dataset_name != "MATH":
                    clean_pred = clean_pred.lower()
                if f"{clean_pred}_{g}" in result_already_check_dict.keys():
                    label = result_already_check_dict[f"{clean_pred}_{g}"]
                elif f"{g}_{clean_pred}" in result_already_check_dict.keys():
                    label = result_already_check_dict[f"{g}_{clean_pred}"]
                else:
                    if math_equal(clean_pred, g):
                        label = True
                        result_already_check_dict[f"{g}_{clean_pred}"] = True
                        result_already_check_dict[f"{clean_pred}_{g}"] = True
                    else:
                        result_already_check_dict[f"{g}_{clean_pred}"] = False
                        result_already_check_dict[f"{clean_pred}_{g}"] = False
            assert clean_pred == clean_pred.strip()
            result_by_same_id[final_id][f"generation_check"].append([pred_str, clean_pred, label])
        save_count += 1
        if save_count % 2000 == 0:
            with open(checked_result_file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(result_by_same_id, indent=2))
        id_count[q_id] += 1
    if save_count > -1:
        with open(checked_result_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result_by_same_id, indent=2))
    return checked_result_file_path