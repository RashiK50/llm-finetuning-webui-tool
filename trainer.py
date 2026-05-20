import os
import torch
import traceback
import platform

from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

import shared_state as ss


class StateUpdateCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs and "loss" in logs:
            ss.state["training_loss"].append(logs["loss"])
            ss.state["current_step"] = state.global_step

            if state.epoch:
                ss.state["current_epoch"] = round(state.epoch, 2)


def _push_lora_to_hf(model, tokenizer, hf_repo_id: str):
    """
    Push the trained LoRA adapters and tokenizer to a Hugging Face repo.
    Reads HF_TOKEN from the environment.  Creates the repo if it does not exist.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Add it to your .env file to enable Hugging Face push."
        )

    ss.state["hf_push_status"]  = "pushing"
    ss.state["hf_push_message"] = f"Pushing LoRA adapters to {hf_repo_id}..."

    print(f"[HF Push] Uploading LoRA adapters to: {hf_repo_id}")
    model.push_to_hub(hf_repo_id, token=hf_token, private=False)

    ss.state["hf_push_message"] = f"Pushing tokenizer to {hf_repo_id}..."
    print(f"[HF Push] Uploading tokenizer to: {hf_repo_id}")
    tokenizer.push_to_hub(hf_repo_id, token=hf_token, private=False)

    repo_url = f"https://huggingface.co/{hf_repo_id}"
    ss.state["hf_push_status"]  = "pushed"
    ss.state["hf_push_message"] = f"Pushed successfully → {repo_url}"
    ss.state["hf_pushed_repo"]  = repo_url
    print(f"[HF Push] Done: {repo_url}")


def run_training(**kwargs):

    try:

        ss.state["status"] = "training"
        ss.state["training_loss"] = []
        ss.state["current_step"] = 0
        ss.state["current_epoch"] = 0
        # Reset any previous push state
        ss.state["hf_push_status"]  = ""
        ss.state["hf_push_message"] = ""
        ss.state["hf_pushed_repo"]  = ""

        model = ss.state["model"]
        tokenizer = ss.state["tokenizer"]
        dataset = ss.state["train_dataset"]

        # Optional: HF repo to push LoRA adapters after training
        hf_repo_id = kwargs.get("hf_repo_id", "").strip()

        is_mac = platform.system() == "Darwin"
        is_cuda = torch.cuda.is_available()

        # safer optimizer
        opt_choice = "adamw_torch"

        # WINDOWS STABILITY FIX
        use_fp16 = False
        use_bf16 = False

        l_rank = kwargs.get("lora_rank", 8)
        l_alpha = kwargs.get("lora_alpha", 16)
        lr = kwargs.get("learning_rate", 1e-4)
        batch_size = kwargs.get("batch_size", 1)
        epochs = kwargs.get("epochs", 5)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        def tokenize_function(examples):

            outputs = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256
            )

            outputs["labels"] = outputs["input_ids"].copy()

            return outputs

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True
        )

        # only for cuda
        if is_cuda and not is_mac:
            model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=int(l_rank),
            lora_alpha=int(l_alpha),
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, config)

        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir="./tiny_outputs",

            per_device_train_batch_size=int(batch_size),

            gradient_accumulation_steps=4,

            num_train_epochs=int(epochs),

            learning_rate=float(lr),

            logging_steps=1,

            save_strategy="no",

            report_to="none",

            optim=opt_choice,

            fp16=use_fp16,

            bf16=use_bf16,

            gradient_checkpointing=False,

            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            callbacks=[StateUpdateCallback()]
        )

        trainer.train()

        # VERY IMPORTANT FIX
        ss.state["model"] = model

        ss.state["status"] = "finished"

        # ── Push LoRA adapters to Hugging Face (if repo ID was provided) ──
        if hf_repo_id:
            try:
                _push_lora_to_hf(model, tokenizer, hf_repo_id)
            except Exception as push_err:
                # Training succeeded — don't mark overall status as error
                ss.state["hf_push_status"]  = "error"
                ss.state["hf_push_message"] = f"HF push failed: {str(push_err)}"
                print(f"[HF Push] ERROR: {push_err}")
                traceback.print_exc()

    except Exception as e:

        ss.state["status"] = "error"
        ss.state["error_message"] = str(e)

        traceback.print_exc()