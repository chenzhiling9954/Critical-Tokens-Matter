DATASET_PATH = {
    "GSM8K": {"train": "./input/GSM8K.jsonl", "test": "./input/GSM8K_test.jsonl"},
    "MATH": {"train": "./input/MATH.jsonl", "test": "./input/MATH_test.jsonl"}
}

TEMPERATURE_BY_DATASET_AND_MODEL = {
    "GSM8K": {
        "Meta-Llama-3-8B": 0.5,
        "Meta-Llama-3-70B": 0.85,
        "deepseek-math-7b-base": 1.0
    },
    "MATH": {
        "Meta-Llama-3-8B": 0.1,
        "Meta-Llama-3-70B": 0.15,
        "deepseek-math-7b-base": 0.3
    }
}


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
    else:
        raise ValueError("Model name should be Meta-Llama-3-8B, Meta-Llama-3-70B or deepseek-math-7b-base")

    return model_path


def get_model_sampling_config(model_name, dataset_name):
    with_sampling_n = 64
    without_sampling_n = 1
    startup_data_sampling_n = 64
    """
    For the subsequent model training used in Contrastive Estimation,
    we control the temperature so that each question in the dataset
    has an average of approximately 10 incorrect answers.
    """
    temperature = TEMPERATURE_BY_DATASET_AND_MODEL[dataset_name][model_name]
    model_path = get_model_path(model_name)

    return {
        "temperature": temperature,
        "model_path": model_path,
        "startup_data_sampling_n": startup_data_sampling_n,
        "with_sampling_n": with_sampling_n,
        "without_sampling_n": without_sampling_n,
        "stop": ["\n\nQuestion:", ".\n\nQuestion:", "\n\nProblem:", ".\n\nProblem:"]
    }


def get_data_selection_config():
    """
    The selection criteria are based on the frequency of generated answers.
    Answers are sorted in descending order of occurrence.
    'top-k': Selects the highest k answers, each corresponding to one response.
    'top-p': Selects answers whose cumulative occurrence percentage reaches p%, each corresponding to one response.
    'pos' (positive) and 'neg' (negative) indicate the correctness of the response.

    For example, suppose there are 10 responses in total, all with the same correctness:
    'a' appears 3 times, 'b' appears 2 times. The remaining 5 responses are unique answers: "c", "d", "e", "f", and "g".
    Selecting top-p=0.5 in this scenario means choosing one instance of "a" and one instance of "b,"
    as these are the most frequently occurring responses that collectively account for 50% of the total occurrences.
    Selecting top-k=1 means choosing one instance of "a" the most frequent response.
    """
    selection_config = {
        "sft_pos": {"top-k": 1, "top-p": None},
        "sft_neg": {"top-k": None, "top-p": 0.5},
        "dpo": {"top-k": 1, "top-p": None},
    }

    for k in selection_config.keys():
        assert selection_config[k]["top-k"] is None or selection_config[k]["top-p"] is None
        if selection_config[k]["top-k"] is not None:
            selection_config[k]["name"] = f"{k}_top_k{selection_config[k]['top-k']}"
        if selection_config[k]["top-p"] is not None:
            selection_config[k]["name"] = f"{k}_top_p{selection_config[k]['top-p']}"
    return selection_config
