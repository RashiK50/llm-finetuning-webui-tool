import torch
import traceback
import platform  # Needed to detect Mac vs Windows
from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import shared_state as ss

class StateUpdateCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            ss.state["training_loss"].append(logs["loss"])
            ss.state["current_step"] = state.global_step

def run_training(**kwargs):
    try:
        ss.state["status"] = "training"
        model = ss.state["model"]
        tokenizer = ss.state["tokenizer"]
        dataset = ss.state["train_dataset"]

        # Hardware Detection
        is_mac = platform.system() == "Darwin"
        is_cuda = torch.cuda.is_available()

        # Dynamic Configuration
        # Mac cannot use paged_8bit; Windows/NVIDIA should use it for VRAM efficiency
        opt_choice = "adamw_torch" if (is_mac or not is_cuda) else "paged_adamw_8bit"
        
        # Mac prefers bf16; Windows is standard with fp16
        use_bf16 = True if is_mac else False
        use_fp16 = False if is_mac else True

        # 1. Capture UI values
        l_rank = kwargs.get('lora_rank', 16)
        l_alpha = kwargs.get('lora_alpha', 32)
        lr = kwargs.get('learning_rate', 1e-4)
        batch_size = kwargs.get('batch_size', 2)

        # 2. Tokenizer logic
        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            outputs = tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
            outputs["labels"] = [list(ids) for ids in outputs["input_ids"]]
            return outputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # 3. Prep for LoRA
        # Only prepare for kbit if using CUDA (bitsandbytes requirement)
        if is_cuda and not is_mac:
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

        # 4. Training Args (Dynamic Hardware Mapping)
        training_args = TrainingArguments(
            output_dir="./tiny_outputs",
            per_device_train_batch_size=int(batch_size),
            gradient_accumulation_steps=8,
            warmup_steps=10,
            max_steps=50,                       
            learning_rate=float(lr),
            logging_steps=1,
            report_to="none",
            
            # --- Hardware Adaptive Settings ---
            optim=opt_choice,
            bf16=use_bf16,
            fp16=use_fp16,
            # Gradient checkpointing can be unstable on some MPS versions, 
            # but usually okay for TinyLlama.
            gradient_checkpointing=True if not is_mac else False, 
            # ----------------------------------
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