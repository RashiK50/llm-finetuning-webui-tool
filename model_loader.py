import os
import platform
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import shared_state as ss

# Renamed to signify permanent storage
MODEL_DIR = "./model_weights"

def load_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> dict:
    hf_token = os.getenv("HF_TOKEN")
    is_mac = platform.system() == "Darwin"
    device = "mps" if is_mac and torch.backends.mps.is_available() else "cpu"

    ss.state["status"] = "loading_model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        print(f"Loading {model_name} from {MODEL_DIR}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token, 
            cache_dir=MODEL_DIR
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
            device_map={"": device},
            token=hf_token,
            cache_dir=MODEL_DIR,
            trust_remote_code=True
        )
        model.eval()

        # REMOVED: _flush_hf_cache call to keep files on disk
        print(f"Model loaded and preserved in {MODEL_DIR}.")

    except Exception as e:
        ss.state["status"] = "error"
        ss.state["error_message"] = str(e)
        raise e

    ss.state.update({
        "model": model,
        "tokenizer": tokenizer,
        "model_name": model_name,
        "model_loaded": True,
        "status": "model_ready",
    })

    return {"success": True, "message": f"Loaded. Running on {device.upper()}. Files kept in {MODEL_DIR}."}