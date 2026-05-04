# trainer.py — Team 2
# SFT + LoRA training pipeline using HuggingFace trl + peft

from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import shared_state as ss


# Custom callback to stream loss + epoch into shared_state
# This is what GET /api/progress reads from
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            ss.state["training_loss"].append(round(logs["loss"], 4))
            ss.state["current_step"] = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        ss.state["current_epoch"] = int(state.epoch)


def run_training(
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    batch_size: int = 2,
):
    try:
        ss.state["status"] = "training"
        ss.state["training_loss"] = []
        ss.state["current_epoch"] = 0
        ss.state["current_step"] = 0
        ss.state["error_message"] = None

        model = ss.state["model"]
        tokenizer = ss.state["tokenizer"]
        train_dataset = ss.state["train_dataset"]
        test_dataset = ss.state["test_dataset"]

        if model is None:
            raise ValueError("Model not loaded. Call /api/load-model first.")
        if train_dataset is None:
            raise ValueError("Dataset not loaded. Call /api/upload-dataset first.")

        # Inject LoRA adapters
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj",
                "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        ss.state["model"] = model
        ss.state["lora_config"] = {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size
        }

        trainable, total = model.get_nb_trainable_parameters()
        print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        training_args = SFTConfig(
            output_dir="./outputs/checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=5,
            save_steps=100,
            eval_strategy="epoch",
            warmup_steps=10,
            lr_scheduler_type="cosine",
            report_to="none",
            dataset_text_field="text",
            max_length=512
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            args=training_args,
            callbacks=[ProgressCallback()]
        )

        ss.state["trainer"] = trainer
        trainer.train()

        ss.state["status"] = "done"
        print("Training complete.")

    except Exception as e:
        ss.state["status"] = "error"
        ss.state["error_message"] = str(e)
        raise e