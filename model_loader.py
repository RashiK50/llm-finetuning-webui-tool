import os
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from tqdm import tqdm
import threading
import shared_state as ss

# Persistent local storage
MODEL_DIR = "./model_weights"

def _format_bytes(num_bytes: float) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes:.1f} B"


class AggregateProgressTqdm(tqdm):
    _total_bytes = 0.0
    _downloaded_bytes = 0.0
    _lock = threading.Lock()

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._total_bytes = 0.0
            cls._downloaded_bytes = 0.0

    @classmethod
    def _update_shared_state(cls):
        with cls._lock:
            total = cls._total_bytes
            done = min(cls._downloaded_bytes, total) if total > 0 else 0.0

        if total <= 0:
            return

        percent = int((done / total) * 100)
        mapped_percent = min(90, max(1, int(percent * 0.9)))
        ss.state["loading_progress"] = max(ss.state.get("loading_progress", 0), mapped_percent)
        ss.state["loading_message"] = (
            f"Downloading model weights... {percent}% "
            f"({_format_bytes(done)} / {_format_bytes(total)})"
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_n = 0.0
        total = float(self.total or 0.0)
        if total > 0:
            with self.__class__._lock:
                self.__class__._total_bytes += total
            self.__class__._update_shared_state()

    def update(self, n=1):
        super().update(n)
        current_n = float(self.n or 0.0)
        delta = max(0.0, current_n - self._last_n)
        self._last_n = current_n
        if delta > 0:
            with self.__class__._lock:
                self.__class__._downloaded_bytes += delta
            self.__class__._update_shared_state()

def load_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> dict:
    hf_token = os.getenv("HF_TOKEN")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ss.state["status"] = "loading_model"
    ss.state["model_loaded"] = False
    ss.state["error_message"] = ""
    ss.state["loading_progress"] = 0
    ss.state["loading_message"] = "Preparing model source..."
    ss.state["cancel_requested"] = False
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        print(f"Loading {model_name} onto {device.upper()}...")

        local_path = os.path.expanduser(str(model_name).strip())
        is_local_path = os.path.isdir(local_path)

        if is_local_path:
            ss.state["loading_progress"] = 25
            ss.state["loading_message"] = "Using local model directory..."
            snapshot_dir = os.path.abspath(local_path)
        else:
            AggregateProgressTqdm.reset()
            ss.state["loading_progress"] = 1
            ss.state["loading_message"] = "Resolving Hugging Face files..."
            snapshot_dir = snapshot_download(
                repo_id=model_name,
                cache_dir=MODEL_DIR,
                token=hf_token,
                tqdm_class=AggregateProgressTqdm,
            )

        if ss.state.get("cancel_requested"):
            ss.state["status"] = "idle"
            ss.state["loading_progress"] = 0
            ss.state["loading_message"] = "Model loading canceled."
            return {"success": False, "message": "Canceled by user."}

        ss.state["loading_progress"] = 92
        ss.state["loading_message"] = "Initializing tokenizer..."
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_dir,
            local_files_only=True,
        )

        if ss.state.get("cancel_requested"):
            ss.state["status"] = "idle"
            ss.state["loading_progress"] = 0
            ss.state["loading_message"] = "Model loading canceled."
            return {"success": False, "message": "Canceled by user."}

        ss.state["loading_progress"] = 96
        ss.state["loading_message"] = "Initializing model..."
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto", # Automatically handles RTX memory
            local_files_only=True,
            trust_remote_code=True
        )

        print(f"Model stored locally in {MODEL_DIR}. Ready to train.")

        # Update state on success
        ss.state.update({
            "model": model,
            "tokenizer": tokenizer,
            "model_name": model_name,
            "model_loaded": True,
            "status": "model_ready",
            "loading_progress": 100,
            "loading_message": "Model loaded successfully.",
            "cancel_requested": False,
        })

        return {"success": True, "message": f"Loaded on {device.upper()}. Weights saved locally."}

    except Exception as e:
        # Catch errors so the background task doesn't fail silently
        ss.state["status"] = "error"
        ss.state["model_loaded"] = False
        ss.state["error_message"] = f"Failed to load model: {str(e)}"
        ss.state["loading_progress"] = 0
        ss.state["loading_message"] = ""
        traceback.print_exc()
        # We don't raise the error here anymore, otherwise the background task crashes
        # and the frontend never finds out why.
        return {"success": False, "message": str(e)}


def load_finetuned_model(base_model_name: str, lora_repo_id: str) -> dict:
    """
    Load a base model and then apply previously pushed LoRA adapters from
    a Hugging Face repo.  This allows continuing training from where you
    left off instead of always starting from the raw base model.

    Args:
        base_model_name: HF repo ID (or local path) of the original base model
                         e.g. "meta-llama/Meta-Llama-3-8B"
        lora_repo_id:    HF repo ID that holds the saved LoRA adapter weights
                         e.g. "your-username/my-finetuned-lora"
    """
    hf_token = os.getenv("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ss.state["status"] = "loading_model"
    ss.state["model_loaded"] = False
    ss.state["error_message"] = ""
    ss.state["loading_progress"] = 0
    ss.state["loading_message"] = "Preparing to reload fine-tuned model..."
    ss.state["cancel_requested"] = False

    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        # ── Step 1: Download / locate base model ──────────────────────────
        print(f"[Reload] Loading base model: {base_model_name}")
        local_base = os.path.expanduser(str(base_model_name).strip())
        is_local = os.path.isdir(local_base)

        if is_local:
            ss.state["loading_progress"] = 10
            ss.state["loading_message"] = "Using local base model directory..."
            base_snapshot = os.path.abspath(local_base)
        else:
            AggregateProgressTqdm.reset()
            ss.state["loading_progress"] = 1
            ss.state["loading_message"] = "Downloading base model weights..."
            base_snapshot = snapshot_download(
                repo_id=base_model_name,
                cache_dir=MODEL_DIR,
                token=hf_token,
                tqdm_class=AggregateProgressTqdm,
            )

        if ss.state.get("cancel_requested"):
            ss.state["status"] = "idle"
            ss.state["loading_progress"] = 0
            ss.state["loading_message"] = "Model loading canceled."
            return {"success": False, "message": "Canceled by user."}

        # ── Step 2: Download LoRA adapter repo from HF ────────────────────
        ss.state["loading_progress"] = 50
        ss.state["loading_message"] = f"Downloading LoRA adapters from {lora_repo_id}..."
        print(f"[Reload] Downloading LoRA adapters: {lora_repo_id}")

        lora_snapshot = snapshot_download(
            repo_id=lora_repo_id,
            cache_dir=MODEL_DIR,
            token=hf_token,
        )

        if ss.state.get("cancel_requested"):
            ss.state["status"] = "idle"
            ss.state["loading_progress"] = 0
            ss.state["loading_message"] = "Model loading canceled."
            return {"success": False, "message": "Canceled by user."}

        # ── Step 3: Load tokenizer ─────────────────────────────────────────
        ss.state["loading_progress"] = 70
        ss.state["loading_message"] = "Initializing tokenizer..."
        tokenizer = AutoTokenizer.from_pretrained(
            base_snapshot,
            local_files_only=True,
        )

        # ── Step 4: Load base model ────────────────────────────────────────
        ss.state["loading_progress"] = 80
        ss.state["loading_message"] = "Initializing base model weights..."
        base_model = AutoModelForCausalLM.from_pretrained(
            base_snapshot,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

        # ── Step 5: Wrap with LoRA adapters ───────────────────────────────
        ss.state["loading_progress"] = 92
        ss.state["loading_message"] = "Attaching LoRA adapters..."
        print(f"[Reload] Wrapping base model with LoRA from {lora_snapshot}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, lora_snapshot)

        print(f"[Reload] Fine-tuned model ready on {device.upper()}.")

        ss.state.update({
            "model":            model,
            "tokenizer":        tokenizer,
            "model_name":       f"{base_model_name} + {lora_repo_id}",
            "model_loaded":     True,
            "status":           "model_ready",
            "loading_progress": 100,
            "loading_message":  "Fine-tuned model reloaded successfully.",
            "cancel_requested": False,
        })

        return {
            "success": True,
            "message": (
                f"Base model '{base_model_name}' loaded with LoRA adapters "
                f"from '{lora_repo_id}' on {device.upper()}."
            ),
        }

    except Exception as e:
        ss.state["status"] = "error"
        ss.state["model_loaded"] = False
        ss.state["error_message"] = f"Failed to reload fine-tuned model: {str(e)}"
        ss.state["loading_progress"] = 0
        ss.state["loading_message"] = ""
        traceback.print_exc()
        return {"success": False, "message": str(e)}
