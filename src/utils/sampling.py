import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from configs.prompt import MATH_PROMPT, GSM8K_PROMPT


def load_model(model_name, num_gpus=-1, start_id=0):
    device = "cuda"
    max_gpu_memory = 35
    if device == "cuda":
        kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_name}/offload"}
        if num_gpus == -1:
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if torch.cuda.device_count() != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: f"{max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
                })
    elif device == "cpu":
        kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    if device == "cuda" and num_gpus == 1:
        model.cuda()
    return model, tokenizer


def load_vllm_model(model_path, gpus):
    available_gpus = gpus.split(',')
    llm = LLM(model=model_path,
              tensor_parallel_size=len(available_gpus),
              trust_remote_code=True,
              swap_space=32)
    return llm


def vllm_model_sampling(llm, dataset, prompt_key, tokenizer, temperature, save_path, dataset_name, sampling_n, stop):
    print(">>>>> save path:", save_path)
    existed_unfinished_result_list = None
    if os.path.exists(save_path.replace(".finished", ".unfinished")):
        existed_unfinished_result_list = []
        with open(save_path.replace(".finished", ".unfinished"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                existed_unfinished_result_list.append(json.loads(line))
        existed_result_ids = [line["id"] for line in existed_unfinished_result_list]
        clean_dataset = []
        for line in dataset:
            if line["id"] not in existed_result_ids:
                clean_dataset.append(line)
        print(f"Existing sampling data({len(dataset) - len(clean_dataset)}/{len(dataset)})")
        dataset = clean_dataset
        if len(dataset) == 0:
            return save_path

    print(f"temperature: {temperature}")
    if temperature == 0:
        sampling_n = 1
    sampling_params = SamplingParams(temperature=temperature,
                                     top_p=0.95,
                                     max_tokens=1024,
                                     n=sampling_n,
                                     stop=stop)
    inputs = []
    if dataset_name == "MATH":
        few_shot_prompt = MATH_PROMPT
    elif dataset_name == "GSM8K":
        few_shot_prompt = GSM8K_PROMPT
    else:
        raise ValueError(f"Dataset {dataset_name}'s prompt not supported")
    for sample in dataset:
        final_prompt = few_shot_prompt + sample[prompt_key]
        inputs.append(final_prompt.strip())
        sample['final_prompt'] = final_prompt

    inputs = [tokenizer.encode(i, add_special_tokens=False) for i in inputs]

    # Save each batch to prevent result loss in case of a crash.
    batch_size = 128
    outputs = []
    for i in tqdm(range(0, len(inputs), batch_size)):
        input_batch = inputs[i: i + batch_size]
        output_batch = llm.generate(prompt_token_ids=input_batch, sampling_params=sampling_params)
        outputs = outputs + output_batch
        temp_outputs = sorted(outputs, key=lambda x: int(x.request_id))
        temp_outputs = [[response.text for response in output.outputs] for output in temp_outputs]
        result_list = [{**sample, 'generation_result': output} for sample, output in zip(dataset, temp_outputs)]
        if existed_unfinished_result_list is not None:
            result_list += existed_unfinished_result_list
        print(f">>>>> save batch {i // batch_size} in {save_path}")
        with open(save_path.replace(".finished", ".unfinished"), "w", encoding="utf-8") as f:
            for line in result_list:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    outputs = [[response.text for response in output.outputs] for output in outputs]
    result_list = [{**sample, 'generation_result': output} for sample, output in zip(dataset, outputs)]
    if existed_unfinished_result_list is not None:
        result_list += existed_unfinished_result_list
    with open(save_path, "w", encoding="utf-8") as f:
        for line in result_list:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return save_path
