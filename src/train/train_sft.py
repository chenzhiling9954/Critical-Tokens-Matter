import json
import random
from transformers import TrainingArguments
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.sampling import load_model


def load_data(file_path, train_type, dataset_name, use_hf_dataset=True, only_question=False, num_val=300):
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    raw_data = raw_data[train_type]
    all_data = []
    valid_data = []
    for q_id in raw_data:
        line = raw_data[q_id]
        question = line["question"]
        answer = line["response"]
        if dataset_name == "MATH":
            question_start = "Problem: "
            response_start = "\nSolution: "
        elif dataset_name == "GSM8K":
            question_start = "Question: "
            response_start = "\nAnswer: "
        if only_question:
            text = question_start + question + response_start
        else:
            text = question_start + question + response_start + answer
        all_data.append({
            "text": text
        })

    if num_val != 0:
        train_data = all_data[:-num_val]
        valid_data = all_data[-num_val:]
    else:
        train_data = all_data
    random.shuffle(train_data)
    if use_hf_dataset:
        valid_data = Dataset.from_dict({key: [dic[key] for dic in valid_data] for key in valid_data[0]})
        train_data = Dataset.from_dict({key: [dic[key] for dic in train_data] for key in train_data[0]})

    return train_data, valid_data


def sft_model_with_lora(model_path, model_name, dataset_path, dataset_name,
                        train_type, lora_path, hyperparameters):
    peft_config = LoraConfig(
        r=hyperparameters['r'],
        lora_alpha=hyperparameters['r'] * 2,
        target_modules=hyperparameters['target_modules'],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM")
    train_data, val_data = load_data(file_path=dataset_path, train_type=train_type, dataset_name=dataset_name)
    print(f"-" * 50)
    print(f">>>>> Training CE model, type: {train_type}")
    if "70B" in model_name:
        batch_size = hyperparameters['per_device_train_batch_size_for_accumulation']
        gradient_accumulation_steps = hyperparameters['gradient_accumulation_steps']
    else:
        batch_size = hyperparameters['per_device_train_batch_size']
        gradient_accumulation_steps = 1
    training_arguments = (
        TrainingArguments(output_dir=lora_path,
                          per_device_train_batch_size=batch_size,
                          optim="adamw_torch",
                          learning_rate=hyperparameters['learning_rate'],
                          eval_steps=hyperparameters['eval_steps'],
                          save_steps=hyperparameters['save_steps'],
                          logging_steps=hyperparameters['logging_steps'],
                          evaluation_strategy="steps",
                          group_by_length=False,
                          num_train_epochs=hyperparameters['num_train_epoch'],
                          bf16=True,
                          lr_scheduler_type="cosine",
                          warmup_steps=hyperparameters['warmup_steps'],
                          load_best_model_at_end=True,
                          gradient_accumulation_steps=gradient_accumulation_steps
                          ))
    base_model, tokenizer = load_model(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    if dataset_name == "GSM8K":
        response_template = "Answer:"
        print(f">>>>> response is after: {response_template}")
    elif dataset_name == "MATH":
        response_template = "Solution:"
        print(f">>>>> response is after: {response_template}")
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    print("train data examples: ", train_data[0])
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=lora_path)
    print(f">>>>> Save model at: {lora_path}")
    print(f"-" * 50)
