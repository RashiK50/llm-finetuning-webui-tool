import torch
import traceback
from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import shared_state as ss

class StateUpdateCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            ss.state["training_loss"].append(logs["loss"])
            ss.state["current_step"] = state.global_step

# We add **kwargs here to "catch" anything the UI sends (like lora_rank)
def run_training(**kwargs):
    try:
        ss.state["status"] = "training"
        model = ss.state["model"]
        tokenizer = ss.state["tokenizer"]
        dataset = ss.state["train_dataset"]

        # 1. Capture UI values or use defaults
        # This handles the 'lora_rank' error specifically
        l_rank = kwargs.get('lora_rank', 16)
        l_alpha = kwargs.get('lora_alpha', 32)
        lr = kwargs.get('learning_rate', 1e-4)
        batch_size = kwargs.get('batch_size', 2)

        # 2. TOKENIZER
        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            # Tokenize the text
            outputs = tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
            
            # CRITICAL FIX: The model needs 'labels' to calculate loss.
            # For Causal LM, labels are identical to input_ids.
            outputs["labels"] = [list(ids) for ids in outputs["input_ids"]]
            
            return outputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # 3. PREP FOR LORA
        model = prepare_model_for_kbit_training(model)
        
        config = LoraConfig(
            r=int(l_rank), 
            lora_alpha=int(l_alpha),
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

        # 4. TRAINING ARGS
        training_args = TrainingArguments(
            output_dir="./tiny_outputs",
            per_device_train_batch_size=int(batch_size),
            gradient_accumulation_steps=8,
            warmup_steps=10,
            max_steps=50,                       
            learning_rate=float(lr),
            fp16=True,
            logging_steps=1,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            callbacks=[StateUpdateCallback()]
        )

        trainer.train()
        ss.state["status"] = "finished"

    except Exception as e:
        ss.state["status"] = "error"
        ss.state["error_message"] = str(e)
        traceback.print_exc()