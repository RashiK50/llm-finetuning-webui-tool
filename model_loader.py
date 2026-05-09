import os
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
import shared_state as ss

# Persistent local storage
MODEL_DIR = "./model_weights"

def load_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> dict:
    hf_token = os.getenv("HF_TOKEN")
    
    # Force CUDA for your RTX GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set explicit loading states for the frontend
    ss.state["status"] = "loading_model"
    ss.state["model_loaded"] = False
    ss.state["error_message"] = ""
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        print(f"Loading {model_name} onto {device.upper()}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token, 
            cache_dir=MODEL_DIR
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto", # Automatically handles RTX memory
            token=hf_token,
            cache_dir=MODEL_DIR,
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
        })

        return {"success": True, "message": f"Loaded on {device.upper()}. Weights saved locally."}

    except Exception as e:
        # Catch errors so the background task doesn't fail silently
        ss.state["status"] = "error"
        ss.state["model_loaded"] = False
        ss.state["error_message"] = f"Failed to load model: {str(e)}"
        traceback.print_exc()
        # We don't raise the error here anymore, otherwise the background task crashes
        # and the frontend never finds out why.
        return {"success": False, "message": str(e)}