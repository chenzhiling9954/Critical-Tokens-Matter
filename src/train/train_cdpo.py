# Modified from trl/trl/trainer/dpo_trainer.py
import json
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer
from typing import Dict, List, Literal, Tuple, Union
from peft import LoraConfig
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.sampling import load_model


class CDPOTrainer(DPOTrainer):
    def __init__(self, only_neg, **kwargs):
        self.only_neg = only_neg
        super().__init__(**kwargs)

    @staticmethod
    def get_batch_logps(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask), loss_mask, (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
    torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps, size_completion, original_all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        original_chosen_logps = original_all_logps[:len_chosen]
        original_rejected_logps = original_all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, original_chosen_logps,
                original_rejected_logps)

    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            reference_original_chosen_logps: torch.FloatTensor,
            reference_original_rejected_logps: torch.FloatTensor,
            policy_original_chosen_logps: torch.FloatTensor,
            policy_original_rejected_logps: torch.FloatTensor,
            prob_result: list
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_size = policy_chosen_logps.size(0)

        chosen_list = [torch.tensor(line["pos"]["ce_prob"] + [1]).to(policy_chosen_logps.device) for line in
                       prob_result]
        rejected_list = [1 - torch.tensor(line["neg"]["ce_prob"] + [0]).to(policy_chosen_logps.device) for line in
                         prob_result]

        all_list = chosen_list + rejected_list
        padding_sequences = pad_sequence(all_list, batch_first=True, padding_value=0)
        chosen_rewards = padding_sequences[:batch_size]
        rejected_rewards = padding_sequences[batch_size:]

        if self.only_neg:
            chosen_logratio = (policy_chosen_logps - reference_chosen_logps).to(self.accelerator.device)
        else:
            chosen_logratio = chosen_rewards * (policy_chosen_logps - reference_chosen_logps).to(
                self.accelerator.device)
        rejected_logratio = rejected_rewards * (policy_rejected_logps - reference_rejected_logps).to(
            self.accelerator.device)
        logits = chosen_logratio - rejected_logratio

        losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )
        chosen_rewards = (
                self.beta * (policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(
            self.accelerator.device)
                             ).detach()
        )
        rejected_rewards = (
                self.beta * (policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(
            self.accelerator.device)
                             ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
            policy_original_chosen_logps,
            policy_original_rejected_logps,
        ) = forward_output[:7]

        if (
                "reference_chosen_logps" in batch
                and "reference_rejected_logps" in batch
                and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                            reference_original_chosen_logps,
                            reference_original_rejected_logps,
                        ) = self.concatenated_forward(self.model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_original_chosen_logps=reference_original_chosen_logps,
            reference_original_rejected_logps=reference_original_rejected_logps,
            policy_original_chosen_logps=policy_original_chosen_logps,
            policy_original_rejected_logps=policy_original_rejected_logps,
            prob_result=batch["prob_result"]
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()

        metrics[f"{prefix}logps/rejected"] = policy_original_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_original_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics


def get_cdpo_train_dataset(dpo_data_with_ce_score_path, tokenizer):
    print(f">>>>> open file {dpo_data_with_ce_score_path}")
    with open(dpo_data_with_ce_score_path, "r", encoding="utf-8") as f:
        data_with_ce_score = json.load(f)
        clean_data = {"prompt": [], "chosen": [], "rejected": [], "prob_result": []}
    for question, pos_response, neg_response, probs in tqdm(zip(data_with_ce_score["prompt"],
                                                                data_with_ce_score["chosen"],
                                                                data_with_ce_score["rejected"],
                                                                data_with_ce_score["prob_result"])):
        question_ids = tokenizer(question).input_ids
        neg_response_ids = tokenizer(question + neg_response).input_ids
        pos_response_ids = tokenizer(question + pos_response).input_ids
        if len(question_ids) < 512 and len(neg_response_ids) < 1024 and len(pos_response_ids) < 1024:
            clean_data["prob_result"].append(probs)
            clean_data["prompt"].append(question)
            clean_data["chosen"].append(pos_response)
            clean_data["rejected"].append(neg_response)

    num_data = len(clean_data["prompt"])
    clean_data = Dataset.from_dict(clean_data).shuffle(seed=33)
    data_with_ce_score = Dataset.from_dict(data_with_ce_score)

    return clean_data, data_with_ce_score, num_data


def cdpo_model_with_lora(model_path, lora_path, dpo_data_with_ce_score_path, hyperparameters):
    print(f"=" * 50)
    base_model, tokenizer = load_model(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    train_dataset, train_dataset_with_ce_score, num_data = (
        get_cdpo_train_dataset(dpo_data_with_ce_score_path=dpo_data_with_ce_score_path, tokenizer=tokenizer))
    save_steps = int(num_data / (hyperparameters["batch_size"] * hyperparameters["gradient_accumulation_steps"]) / 2)
    epochs = hyperparameters["num_train_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    print(f">>>>> Train cDPO")
    print(f">>>>> Num of epochs: {epochs}")
    print(f">>>>> Checkpoints will be saved to {lora_path}")
    training_args = DPOConfig(
        output_dir=lora_path,
        beta=hyperparameters["beta"],
        save_steps=save_steps,
        logging_steps=hyperparameters["logging_steps"],
        warmup_ratio=hyperparameters["warmup_ratio"],
        max_length=hyperparameters["max_length"],
        max_prompt_length=hyperparameters["max_prompt_length"],
        num_train_epochs=hyperparameters["num_train_epochs"],
        per_device_train_batch_size=hyperparameters["batch_size"],
        learning_rate=learning_rate,
        weight_decay=hyperparameters["weight_decay"],
        optim="adamw_torch",
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"]
    )
    peft_config = LoraConfig(
        r=hyperparameters["r"],
        lora_alpha=hyperparameters["lora_alpha"],
        target_modules=hyperparameters["target_modules"],
        lora_dropout=hyperparameters["lora_dropout"],
        bias=hyperparameters["bias"],
        task_type=hyperparameters["task_type"])

    dpo_trainer = CDPOTrainer(
        only_neg=hyperparameters["only_neg"],
        model=base_model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config)
    dpo_trainer.train()
    print(f"=" * 50)
