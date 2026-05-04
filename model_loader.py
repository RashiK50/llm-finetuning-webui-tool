import os
import time
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import shared_state as ss


def load_model(model_name: str = "mistralai/Mistral-7B-v0.3") -> dict:
    # Already loaded
    if ss.state["model"] is not None:
        return {
            "success": True,
            "message": "Model already loaded — skipping reload.",
            "model_name": ss.state["model_name"],
            "gpu_memory_used_gb": _gpu_mem(),
            "quantization_config": ss.state["quantization_config"],
        }

    # HF token check
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Add it to your .env file.")

    # CUDA check
    if not torch.cuda.is_available():
        raise EnvironmentError("No CUDA GPU detected. Need 6GB+ VRAM.")

    # VRAM check
    free_vram = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)
    ) / 1e9
    if free_vram < 5.5:
        raise EnvironmentError(f"Only {free_vram:.2f}GB VRAM free. Need at least 6GB.")

    ss.state["status"] = "loading_model"
    ss.state["error_message"] = None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,   # bfloat16 — not float16
        bnb_4bit_use_double_quant=True,
    )
    quant_summary = {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
    }

    start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        model.eval()

    except OSError as e:
        _set_error(str(e))
        msg = "Invalid or expired HF_TOKEN." if "401" in str(e) else str(e)
        raise OSError(msg)
    except RuntimeError as e:
        _set_error(str(e))
        msg = "CUDA out of memory." if "out of memory" in str(e).lower() else str(e)
        raise RuntimeError(msg)
    except Exception:
        tb = traceback.format_exc()
        _set_error(tb)
        raise

    ss.state.update({
        "model": model,
        "tokenizer": tokenizer,
        "model_name": model_name,
        "model_loaded": True,
        "quantization_config": quant_summary,
        "status": "model_ready",
    })

    return {
        "success": True,
        "message": f"Model '{model_name}' loaded successfully.",
        "model_name": model_name,
        "load_time_seconds": round(time.perf_counter() - start, 2),
        "gpu_memory_used_gb": _gpu_mem(),
        "quantization_config": quant_summary,
    }


def _gpu_mem() -> float:
    try:
        return round(torch.cuda.memory_allocated() / 1e9, 3)
    except Exception:
        return 0.0


def _set_error(msg: str):
    ss.state["status"] = "error"
    ss.state["error_message"] = msg