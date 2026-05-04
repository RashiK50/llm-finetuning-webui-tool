import os
import json
import shutil

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

load_dotenv()

import shared_state as ss
from model_loader import load_model
from data_pipeline import validate_and_load
from trainer import run_training
from evaluator import run_evaluation

app = FastAPI(
    title="LLM Fine-Tuning WebUI Tool",
    description="Fine-tune Mistral 7B on CA legislature QA data via SFT + LoRA. Use /docs to test all endpoints.",
    version="1.0.0",
)


# ── Request bodies ────────────────────────────────────────────────────────────

class LoadModelRequest(BaseModel):
    model_name: Optional[str] = "mistralai/Mistral-7B-v0.3"

class TrainRequest(BaseModel):
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    lora_dropout: Optional[float] = 0.05
    learning_rate: Optional[float] = 2e-4
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 2

class EvaluateRequest(BaseModel):
    num_samples: Optional[int] = 10


# ── Team 1 — Data & Model ─────────────────────────────────────────────────────

@app.get("/api/status", tags=["Team 1 — Data & Model"])
def get_status():
    """Single source of truth. Check before calling anything else."""
    return {
        "status": ss.state["status"],
        "model_loaded": ss.state["model_loaded"],
        "model_name": ss.state["model_name"],
        "dataset_ready": ss.state["dataset"] is not None,
        "dataset_size": ss.state["dataset_size"],
        "current_epoch": ss.state["current_epoch"],
        "current_step": ss.state["current_step"],
        "error_message": ss.state["error_message"],
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_used_gb": _gpu_mem(),
    }


@app.post("/api/load-model", tags=["Team 1 — Data & Model"])
def load_model_endpoint(request: LoadModelRequest):
    """
    Loads model in 4-bit NF4 + bfloat16.
    For local testing: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    For production: mistralai/Mistral-7B-v0.3
    """
    try:
        return load_model(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-dataset", tags=["Team 1 — Data & Model"])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a .json file of QA pairs.
    Format: [{"instruction": "...", "output": "..."}]
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files accepted.")
    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    try:
        return validate_and_load(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Team 2 — Training & Evaluation ───────────────────────────────────────────

@app.post("/api/train", tags=["Team 2 — Training & Evaluation"])
def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Kicks off SFT + LoRA as a background task.
    Poll GET /api/progress every 2-3 seconds to watch loss.
    """
    if not ss.state["model_loaded"]:
        raise HTTPException(status_code=400, detail="Load the model first via /api/load-model.")
    if ss.state["dataset"] is None:
        raise HTTPException(status_code=400, detail="Upload a dataset first via /api/upload-dataset.")
    if ss.state["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress.")

    background_tasks.add_task(run_training, **request.model_dump())
    return {"success": True, "message": "Training started.", "config": request.model_dump()}


@app.get("/api/progress", tags=["Team 2 — Training & Evaluation"])
def get_progress():
    """Poll this every 2-3 seconds during training to get live loss updates."""
    return {
        "status": ss.state["status"],
        "current_epoch": ss.state["current_epoch"],
        "current_step": ss.state["current_step"],
        "latest_loss": ss.state["training_loss"][-1] if ss.state["training_loss"] else None,
        "all_losses": ss.state["training_loss"],
        "error_message": ss.state["error_message"],
    }


@app.post("/api/evaluate", tags=["Team 2 — Training & Evaluation"])
def evaluate(request: EvaluateRequest):
    """
    Evaluates the fine-tuned model on the test split.
    Returns ROUGE-L, BERTScore, Token F1, and sample outputs.
    """
    if ss.state["model"] is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    if ss.state["test_dataset"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    try:
        return run_evaluation(num_samples=request.num_samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export", tags=["Team 2 — Training & Evaluation"])
def export_model(type: str = "lora"):
    """
    Export the fine-tuned model.
    ?type=lora    → LoRA adapter zip (~100MB)
    ?type=merged  → Full merged model zip (~14GB)
    """
    if ss.state["model"] is None:
        raise HTTPException(status_code=400, detail="No model to export.")

    os.makedirs("./outputs", exist_ok=True)

    if type == "lora":
        path = "./outputs/lora-adapter"
        os.makedirs(path, exist_ok=True)
        ss.state["model"].save_pretrained(path)
        ss.state["tokenizer"].save_pretrained(path)
        _write_model_card(path)
        zip_path = shutil.make_archive("./outputs/lora-adapter", "zip", path)
        return FileResponse(zip_path, filename="lora-adapter.zip", media_type="application/zip")

    elif type == "merged":
        try:
            merged = ss.state["model"].merge_and_unload()
            path = "./outputs/merged-model"
            os.makedirs(path, exist_ok=True)
            merged.save_pretrained(path)
            ss.state["tokenizer"].save_pretrained(path)
            _write_model_card(path)
            zip_path = shutil.make_archive("./outputs/merged-model", "zip", path)
            return FileResponse(zip_path, filename="merged-model.zip", media_type="application/zip")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")

    raise HTTPException(status_code=400, detail="type must be 'lora' or 'merged'.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_mem() -> float:
    try:
        return round(torch.cuda.memory_allocated() / 1e9, 3)
    except Exception:
        return 0.0


def _write_model_card(path: str):
    card = {
        "base_model": ss.state["model_name"],
        "fine_tuned_on": "CA Legislature QA",
        "dataset_size": ss.state["dataset_size"],
        "lora_config": ss.state["lora_config"],
        "eval_results": ss.state["eval_results"],
    }
    with open(f"{path}/model_card.json", "w") as f:
        json.dump(card, f, indent=2)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)