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


class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7

# ── Team 1 — Data & Model ─────────────────────────────────────────────────────

@app.get("/api/status", tags=["Team 1 — Data & Model"])
def get_status():
    """Check this to verify the dataset is actually in memory."""
    return {
        "status": ss.state.get("status", "idle"),
        "model_loaded": ss.state.get("model_loaded", False),
        "model_name": ss.state.get("model_name", None),
        "dataset_ready": ss.state.get("train_dataset") is not None, 
        "dataset_size": ss.state.get("dataset_size", 0),
        "current_epoch": ss.state.get("current_epoch", 0),
        "current_step": ss.state.get("current_step", 0),
        "error_message": ss.state.get("error_message", ""),
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_used_gb": _gpu_mem(),
    }


@app.post("/api/generate", tags=["Team 2 — Training & Evaluation"])
async def generate_response(request: ChatRequest):
    """
    Hit this to chat with your fine-tuned model.
    """
    if not ss.state.get("model_loaded") or ss.state.get("model") is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")
    
    model = ss.state["model"]
    tokenizer = ss.state["tokenizer"]

    try:
        # Prepare the prompt (matching the training format is key!)
        prompt = f"Instruction: {request.message}\nOutput: "
        
        # Move inputs to the correct device (MPS for Mac, CUDA for Windows)
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and clean up the output
        full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        response = full_text.split("Output:")[-1].strip()

        return {"success": True, "response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load-model", tags=["Team 1 — Data & Model"])
def load_model_endpoint(request: LoadModelRequest, background_tasks: BackgroundTasks):
    """
    Loads model in background to prevent frontend timeout.
    """
    if ss.state.get("status") == "loading_model":
        raise HTTPException(status_code=400, detail="A model is already loading.")
        
    # FIX: Move load_model to a background task
    background_tasks.add_task(load_model, request.model_name)
    
    return {
        "success": True, 
        "message": f"Started loading {request.model_name} in the background. Poll /api/status for updates."
    }

@app.post("/api/upload-dataset", tags=["Team 1 — Data & Model"])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a .json file of QA pairs, saves it to disk, then loads it into memory.
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files accepted.")
    
    try:
        content = await file.read()
        data = json.loads(content)
        
        # FIX: Actually save the dataset to disk so it isn't lost
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", "latest_uploaded_dataset.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON.")
        
    try:
        # Assuming validate_and_load populates ss.state["train_dataset"]
        result = validate_and_load(data)
        return {"success": True, "message": f"Dataset uploaded and saved to {save_path}", "details": result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

# ── Team 2 — Training & Evaluation ───────────────────────────────────────────

@app.post("/api/train", tags=["Team 2 — Training & Evaluation"])
def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    if not ss.state.get("model_loaded"):
        raise HTTPException(status_code=400, detail="Load the model first via /api/load-model.")
    
    if ss.state.get("train_dataset") is None: 
        raise HTTPException(status_code=400, detail="Upload a dataset first via /api/upload-dataset.")
        
    if ss.state.get("status") == "training":
        raise HTTPException(status_code=400, detail="Training already in progress.")

    background_tasks.add_task(run_training, **request.model_dump())
    return {"success": True, "message": "Training started in background.", "config": request.model_dump()}

@app.get("/api/progress", tags=["Team 2 — Training & Evaluation"])
def get_progress():
    """Poll this every 2-3 seconds during training to get live loss updates."""
    losses = ss.state.get("training_loss", [])
    return {
        "status": ss.state.get("status"),
        "current_epoch": ss.state.get("current_epoch"),
        "current_step": ss.state.get("current_step"),
        "latest_loss": losses[-1] if losses else None,
        "all_losses": losses,
        "error_message": ss.state.get("error_message"),
    }

@app.post("/api/evaluate", tags=["Team 2 — Training & Evaluation"])
def evaluate(request: EvaluateRequest):
    if ss.state.get("model") is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    if ss.state.get("test_dataset") is None:
        raise HTTPException(status_code=400, detail="No test dataset loaded.")
    try:
        return run_evaluation(num_samples=request.num_samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export", tags=["Team 2 — Training & Evaluation"])
async def export_model(type: str = "lora", background_tasks: BackgroundTasks = None):
    if ss.state.get("model") is None:
        raise HTTPException(status_code=400, detail="No model to export.")

    # Create a unique output directory
    output_base = "./outputs"
    export_dir = os.path.join(output_base, f"export_{type}")
    os.makedirs(export_dir, exist_ok=True)

    try:
        ss.state["status"] = "exporting"
        
        if type == "lora":
            # LoRA is small (~50MB-200MB), zipping is usually okay
            ss.state["model"].save_pretrained(export_dir)
            ss.state["tokenizer"].save_pretrained(export_dir)
            _write_model_card(export_dir)
            
            # Use a faster zip approach or return path
            return {"success": True, "message": f"Model saved to {export_dir}", "path": os.path.abspath(export_dir)}

        elif type == "merged":
            # WARNING: Merging on an 8GB/16GB Mac will likely cause a crash/freeze
            # We suggest returning the LoRA path instead, but if you want to try:
            merged_model = ss.state["model"].merge_and_unload()
            merged_model.save_pretrained(export_dir)
            ss.state["tokenizer"].save_pretrained(export_dir)
            
            ss.state["status"] = "model_ready"
            return {"success": True, "message": f"Full model merged and saved to {export_dir}"}

    except Exception as e:
        ss.state["status"] = "error"
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_mem() -> float:
    try:
        return round(torch.cuda.memory_allocated() / 1e9, 3)
    except Exception:
        return 0.0

def _write_model_card(path: str):
    card = {
        "base_model": ss.state.get("model_name"),
        "fine_tuned_on": "CA Legislature QA",
        "dataset_size": ss.state.get("dataset_size"),
        "lora_config": ss.state.get("lora_config"),
        "eval_results": ss.state.get("eval_results"),
    }
    with open(f"{path}/model_card.json", "w") as f:
        json.dump(card, f, indent=2)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)