# trainer.py
import torch
import os
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import shared_state as ss

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
        
        # Pull from shared state
        model = ss.state.get("model")
        tokenizer = ss.state.get("tokenizer")
        train_dataset = ss.state.get("train_dataset")

        if model is None or train_dataset is None:
            raise ValueError("State is empty. Please call /load-model and /upload-dataset again.")

        # Lora Setup - Simplified target modules for TinyLlama stability
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        
        training_args = SFTConfig(
            output_dir="./outputs/checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            logging_steps=1,
            # M4 specific settings
            bf16=False, 
            fp16=False,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=512
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            processing_class=tokenizer,
            callbacks=[ProgressCallback()]
        )

        trainer.train()
        
        # Save results
        model.save_pretrained("./outputs/final_adapter")
        ss.state["status"] = "done"
        print("Training complete.")

    except Exception as e:
        ss.state["status"] = "error"
        ss.state["error_message"] = str(e)
        print(f"Error during training: {e}")
        raise e