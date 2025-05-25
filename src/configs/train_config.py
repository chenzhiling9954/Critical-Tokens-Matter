def get_model_path(model_name):
    if model_name == "Meta-Llama-3-8B":
        model_path = {
            "base": "meta-llama/Meta-Llama-3-8B",
        }
    elif model_name == "Meta-Llama-3-70B":
        model_path = {
            "base": "meta-llama/Meta-Llama-3-70B",
        }
    elif model_name == "deepseek-math-7b-base":
        model_path = {
            "base": "deepseek-ai/deepseek-math-7b-base",
        }
    elif model_name == "Qwen2.5-7B":
        model_path = {
            "base": "Qwen/Qwen2.5-7B",
        }
    elif model_name == "Qwen2.5-32B":
        model_path = {
            "base": "Qwen/Qwen2.5-32B",
        }
    else:
        raise ValueError("Model name should be Meta-Llama-3-8B, Meta-Llama-3-70B, deepseek-math-7b-base, Qwen2.5-7B or Qwen2.5-32B")

    return model_path

def get_sft_config(model_name):
    hyperparameters = {
        "learning_rate": 3e-4,
        "per_device_train_batch_size": 6,
        "warmup_steps": 100,
        "eval_steps": 200,
        "save_steps": 200,
        "logging_steps": 20,
        "num_train_epoch": 1,
        "gradient_accumulation_steps": 3,
        "per_device_train_batch_size_for_accumulation": 2,
        "target_modules": ["gate_proj", "down_proj", "up_proj"],
        "r": 8
    }

    model_path = get_model_path(model_name)

    return {
        "model_path": model_path,
        "hyperparameters": hyperparameters
    }


def get_cdpo_config(model_name):
    hyperparameters = {
        "learning_rate": 4e-5,
        "num_train_epochs": 3,
        "beta": 0.1,
        "logging_steps": 10,
        "warmup_ratio": 0.1,
        "max_length": 1500,
        "max_prompt_length": 512,
        "weight_decay": 0.01,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "only_neg": True,
        "r": 16,
        "lora_alpha": 16 * 2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }

    model_path = get_model_path(model_name)

    return {
        "model_path": model_path,
        "hyperparameters": hyperparameters
    }
