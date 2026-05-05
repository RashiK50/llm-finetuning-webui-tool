import os
import shutil
import platform
import torch
from huggingface_hub import scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer
import shared_state as ss

# Define a local temporary path for model files
TEMP_MODEL_DIR = "./temp_model_cache"

def load_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> dict:
    # 1. Environment & Device Detection
    hf_token = os.getenv("HF_TOKEN")
    is_mac = platform.system() == "Darwin"
    device = "mps" if is_mac and torch.backends.mps.is_available() else "cpu"

    ss.state["status"] = "loading_model"
    
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

    try:
        print(f"Downloading {model_name} to temporary storage...")
        
        # 2. Tokenizer Setup (pointed to temp dir)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token, 
            cache_dir=TEMP_MODEL_DIR
        )

        # 3. Model Loading (Loads into RAM/MPS, then we can wipe disk)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
            device_map={"": device},
            token=hf_token,
            cache_dir=TEMP_MODEL_DIR,
            trust_remote_code=True
        )
        model.eval()

        # 4. THE FLUSH: Wipe the downloaded files from disk
        print("Model loaded into memory. Flushing disk storage...")
        _flush_hf_cache(model_name)

    except Exception as e:
        # If it fails, still try to clean up
        if os.path.exists(TEMP_MODEL_DIR):
            shutil.rmtree(TEMP_MODEL_DIR)
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

    return {"success": True, "message": f"Loaded and disk flushed. Running on {device.upper()}."}

def _flush_hf_cache(repo_id: str):
    """Scan the temp directory and delete the cached revision of the model."""
    try:
        cache_info = scan_cache_dir(TEMP_MODEL_DIR)
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                # Get the strategy to delete all revisions for this model
                delete_strategy = cache_info.delete_revisions(*[r.commit_hash for r in repo.revisions])
                delete_strategy.execute()
        
        # Final safety: remove the actual folder structure if empty
        if os.path.exists(TEMP_MODEL_DIR):
            shutil.rmtree(TEMP_MODEL_DIR)
            os.makedirs(TEMP_MODEL_DIR, exist_ok=True)
    except Exception as e:
        print(f"Cleanup warning: {e}")