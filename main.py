import os
import json
import shutil
import platform
import zipfile
from urllib.parse import quote_plus


import torch
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
import asyncio
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

load_dotenv()

import shared_state as ss
from model_loader import load_model, load_finetuned_model
from data_pipeline import validate_and_load, validate_and_load_eval_only
from trainer import run_training
from evaluator import run_evaluation
from gguf_exporter import convert_and_push


app = FastAPI(
    title="LLM Fine-Tuning WebUI Tool",
    description="Fine-tune Mistral 7B on CA legislature QA data via SFT + LoRA. Use /docs to test all endpoints.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request bodies ────────────────────────────────────────────────────────────

class LoadModelRequest(BaseModel):
    model_name: Optional[str] = "meta-llama/Meta-Llama-3-8B"

class LoadFinetunedModelRequest(BaseModel):
    base_model_name: str                    # e.g. "meta-llama/Meta-Llama-3-8B"
    lora_repo_id: str                       # e.g. "your-username/my-finetuned-lora"

class TrainRequest(BaseModel):
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    lora_dropout: Optional[float] = 0.05
    learning_rate: Optional[float] = 2e-4
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 2
    hf_repo_id: Optional[str] = ""         # NEW: push LoRA adapters here after training

class ExportGGUFRequest(BaseModel):
    merged_model_dir: str                   # local path to saved merged model
    hf_repo_id: str                         # HF repo to push the .gguf file
    quantisation: Optional[str] = "f16"    # "f16", "q8_0", "q4_k_m" etc.

class EvaluateRequest(BaseModel):
    num_samples: Optional[int] = 10


class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = None
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7

class CancelRequest(BaseModel):
    operation: Optional[str] = "loading_model"

# ── Team 1 — Data & Model ─────────────────────────────────────────────────────

@app.get("/api/status", tags=["Team 1 — Data & Model"])
def get_status():
    """Check this to verify the dataset is actually in memory."""
    ram = _get_ram_info()
    gpu = _get_gpu_info()
    model_type = _get_model_type(ss.state.get("model"))
    return {
        "status": ss.state.get("status", "idle"),
        "model_loaded": ss.state.get("model_loaded", False),
        "model_name": ss.state.get("model_name", None),
        "model_type": model_type,
        "dataset_ready": ss.state.get("train_dataset") is not None, 
        "dataset_size": ss.state.get("dataset_size", 0),
        "dataset_name": ss.state.get("dataset_name", ""),
        "dataset_preview": ss.state.get("dataset_preview", []),
        "current_epoch": ss.state.get("current_epoch", 0),
        "current_step": ss.state.get("current_step", 0),
        "error_message": ss.state.get("error_message", ""),
        "loading_progress": ss.state.get("loading_progress", 0),
        "loading_message": ss.state.get("loading_message", ""),
        "export_download_url": ss.state.get("export_download_url", ""),
        "export_artifact_path": ss.state.get("export_artifact_path", ""),
        "export_type": ss.state.get("export_type", ""),
        "export_error": ss.state.get("export_error", ""),
        "gpu_available": gpu["available"],
        "gpu_name": gpu["name"],
        "gpu_memory_used_gb": gpu["used_gb"],
        "gpu_memory_total_gb": gpu["total_gb"],
        "ram_used_gb": ram["used"],
        "ram_total_gb": ram["total"],
        "eval_results": ss.state.get("eval_results"),
        # HF push status (new)
        "hf_push_status":  ss.state.get("hf_push_status", ""),
        "hf_push_message": ss.state.get("hf_push_message", ""),
        "hf_pushed_repo":  ss.state.get("hf_pushed_repo", ""),
        "gguf_push_status":  ss.state.get("gguf_push_status", ""),
        "gguf_push_message": ss.state.get("gguf_push_message", ""),
        "gguf_pushed_repo":  ss.state.get("gguf_pushed_repo", ""),
    }

@app.get("/api/status-stream", tags=["Team 1 — Data & Model"])
async def status_stream():
    async def event_generator():
        last_state = None
        while True:
            ram = _get_ram_info()
            gpu = _get_gpu_info()
            model_type = _get_model_type(ss.state.get("model"))
            current_state = {
                "status": ss.state.get("status", "idle"),
                "model_loaded": ss.state.get("model_loaded", False),
                "model_name": ss.state.get("model_name", None),
                "model_type": model_type,
                "dataset_ready": ss.state.get("train_dataset") is not None, 
                "dataset_size": ss.state.get("dataset_size", 0),
                "dataset_name": ss.state.get("dataset_name", ""),
                "dataset_preview": ss.state.get("dataset_preview", []),
                "current_epoch": ss.state.get("current_epoch", 0),
                "current_step": ss.state.get("current_step", 0),
                "error_message": ss.state.get("error_message", ""),
                "loading_progress": ss.state.get("loading_progress", 0),
                "loading_message": ss.state.get("loading_message", ""),
                "export_download_url": ss.state.get("export_download_url", ""),
                "export_artifact_path": ss.state.get("export_artifact_path", ""),
                "export_type": ss.state.get("export_type", ""),
                "export_error": ss.state.get("export_error", ""),
                "gpu_available": gpu["available"],
                "gpu_name": gpu["name"],
                "gpu_memory_used_gb": gpu["used_gb"],
                "gpu_memory_total_gb": gpu["total_gb"],
                "ram_used_gb": ram["used"],
                "ram_total_gb": ram["total"],
                "eval_results": ss.state.get("eval_results"),
                # HF push status (new)
                "hf_push_status":  ss.state.get("hf_push_status", ""),
                "hf_push_message": ss.state.get("hf_push_message", ""),
                "hf_pushed_repo":  ss.state.get("hf_pushed_repo", ""),
                "gguf_push_status":  ss.state.get("gguf_push_status", ""),
                "gguf_push_message": ss.state.get("gguf_push_message", ""),
                "gguf_pushed_repo":  ss.state.get("gguf_pushed_repo", ""),
            }
            if current_state != last_state:
                last_state = current_state.copy()
                yield f"data: {json.dumps(current_state)}\n\n"
            await asyncio.sleep(0.5)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
        system_prompt = (
            "You are a precise assistant in a fine-tuning playground. "
            "Answer directly, avoid repetition, and keep responses practical."
        )

        normalized_history = []
        for item in (request.history or [])[-12:]:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role in {"model", "bot"}:
                role = "assistant"
            if role not in {"user", "assistant"}:
                continue
            normalized_history.append({"role": role, "content": content})

        messages = [{"role": "system", "content": system_prompt}, *normalized_history, {"role": "user", "content": request.message}]

        prompt = None
        chat_template = getattr(tokenizer, "chat_template", None)
        if hasattr(tokenizer, "apply_chat_template") and chat_template:
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = None

        if prompt is None:
            turns = []
            for msg in normalized_history:
                prefix = "User" if msg["role"] == "user" else "Assistant"
                turns.append(f"{prefix}: {msg['content']}")
            turns.append(f"User: {request.message}")
            prompt = (
                f"System: {system_prompt}\n\n"
                + "\n".join(turns[-10:])
                + "\nAssistant:"
            )
        
        first_param = next(model.parameters(), None)
        input_device = first_param.device if first_param is not None else torch.device("cpu")
        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

        max_new_tokens = max(32, min(int(request.max_tokens or 200), 384))
        temperature = 0.0
        do_sample = False

        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = output_tokens[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        for marker in ["<|assistant|>", "assistant:", "Assistant:"]:
            if response.lower().startswith(marker.lower()):
                response = response[len(marker):].strip()
        for marker in ["User:", "System:", "### Instruction:", "### Response:"]:
            if marker in response:
                response = response.split(marker)[0].strip()

        if not response:
            response = "I’m not sure how to answer that clearly yet."

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

    ss.state["cancel_requested"] = False
        
    # FIX: Move load_model to a background task
    background_tasks.add_task(load_model, request.model_name)
    
    return {
        "success": True, 
        "message": f"Started loading {request.model_name} in the background. Poll /api/status for updates."
    }


@app.post("/api/load-finetuned-model", tags=["Team 1 — Data & Model"])
def load_finetuned_model_endpoint(request: LoadFinetunedModelRequest, background_tasks: BackgroundTasks):
    """
    Load a base model and then attach previously pushed LoRA adapters from HF.
    Use this to continue training from a checkpoint instead of starting from scratch.

    Example body:
        {
          "base_model_name": "meta-llama/Meta-Llama-3-8B",
          "lora_repo_id": "your-username/my-finetuned-lora"
        }
    """
    if ss.state.get("status") == "loading_model":
        raise HTTPException(status_code=400, detail="A model is already loading.")

    ss.state["cancel_requested"] = False
    background_tasks.add_task(
        load_finetuned_model,
        request.base_model_name,
        request.lora_repo_id,
    )

    return {
        "success": True,
        "message": (
            f"Started loading base model '{request.base_model_name}' "
            f"with LoRA adapters from '{request.lora_repo_id}'. "
            "Poll /api/status for updates."
        ),
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
        ss.state["dataset_name"] = file.filename or "dataset.json"
        ss.state["status"] = "dataset_ready"
        return {
            "success": True, 
            "message": f"Dataset uploaded and saved to {save_path}", 
            "size": result["size"],
            "preview": result["preview"]
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/api/clear-dataset", tags=["Team 1 — Data & Model"])
def clear_dataset():
    ss.state["train_dataset"] = None
    ss.state["test_dataset"] = None
    ss.state["dataset_size"] = 0
    ss.state["dataset_preview"] = []
    ss.state["dataset_name"] = ""
    if ss.state.get("status") == "dataset_ready":
        ss.state["status"] = "model_ready" if ss.state.get("model_loaded") else "idle"
    return {"success": True, "message": "Dataset cleared."}


@app.post("/api/upload-eval-dataset", tags=["Team 1 — Data & Model"])
async def upload_eval_dataset(file: UploadFile = File(...)):
    """
    Upload a SEPARATE evaluation-only JSON file with differently-phrased questions.
    This replaces ONLY the test split used by /api/evaluate — training data is untouched.

    Expected JSON format: same as the main dataset:
        [{"instruction": "...", "output": "..."}, ...]
    OR Group 2 format:
        [{"Question": "...", "Analysis": "...", "Answer": "..."}, ...]
    """
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files accepted.")

    try:
        content = await file.read()
        data = json.loads(content)

        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", "latest_eval_dataset.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON.")

    try:
        result = validate_and_load_eval_only(data)
        return {
            "success": True,
            "message": f"Eval dataset uploaded ({result['size']} entries). Training data unchanged.",
            "size":    result["size"],
            "preview": result["preview"],
        }
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
    """Poll this every 2-3 seconds during training to get live loss and accuracy updates."""
    losses = ss.state.get("training_loss", [])
    accuracies = ss.state.get("training_accuracy", [])
    return {
        "status": ss.state.get("status"),
        "current_epoch": ss.state.get("current_epoch"),
        "current_step": ss.state.get("current_step"),
        "latest_loss": losses[-1] if losses else None,
        "all_losses": losses,
        "training_loss": losses,
        "latest_accuracy": accuracies[-1] if accuracies else None,
        "all_accuracies": accuracies,
        "training_accuracy": accuracies,
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
def export_model(type: str = "lora", hf_repo_id: str = "", background_tasks: BackgroundTasks = None):
    if ss.state.get("model") is None:
        raise HTTPException(status_code=400, detail="No model to export.")

    if ss.state.get("status") == "exporting":
        raise HTTPException(status_code=400, detail="An export is already in progress.")

    if type not in {"lora", "merged"}:
        raise HTTPException(status_code=400, detail="Invalid export type. Use 'lora' or 'merged'.")

    previous_status = ss.state.get("status", "finished")
    ss.state["status"] = "exporting"
    ss.state["loading_progress"] = 1
    ss.state["loading_message"] = f"Queued {type} export..."
    ss.state["export_download_url"] = ""
    ss.state["export_artifact_path"] = ""
    ss.state["export_type"] = type
    ss.state["export_error"] = ""
    # Reset HF push state for this export cycle
    ss.state["hf_push_status"]  = ""
    ss.state["hf_push_message"] = ""
    ss.state["hf_pushed_repo"]  = ""

    background_tasks.add_task(_run_export_job, type, previous_status, hf_repo_id.strip())
    return {"success": True, "message": f"{type} export started."}


@app.post("/api/export-gguf", tags=["Team 2 — Training & Evaluation"])
def export_gguf(request: ExportGGUFRequest, background_tasks: BackgroundTasks):
    """
    Convert a locally saved merged model to GGUF format and push it to Hugging Face.

    Steps performed in the background:
      1. Clone llama.cpp (once, cached after first run)
      2. Run convert_hf_to_gguf.py on the merged model directory
      3. Upload the resulting .gguf file to your HF repo

    Requires HF_TOKEN in your .env with write permission.

    Example body:
        {
          "merged_model_dir": "./outputs/export_merged_dir",
          "hf_repo_id": "your-username/my-model-gguf",
          "quantisation": "f16"
        }
    """
    if not os.path.isdir(request.merged_model_dir):
        raise HTTPException(
            status_code=400,
            detail=f"merged_model_dir not found: {request.merged_model_dir}",
        )

    if not os.getenv("HF_TOKEN"):
        raise HTTPException(
            status_code=400,
            detail="HF_TOKEN is not set. Add it to your .env file.",
        )

    # Reset GGUF push state
    ss.state["gguf_push_status"]  = "converting"
    ss.state["gguf_push_message"] = "GGUF export queued..."
    ss.state["gguf_pushed_repo"]  = ""

    background_tasks.add_task(
        convert_and_push,
        request.merged_model_dir,
        request.hf_repo_id,
        request.quantisation,
    )

    return {
        "success": True,
        "message": (
            f"GGUF conversion started for '{request.merged_model_dir}'. "
            f"Will push to '{request.hf_repo_id}'. Poll /api/status for progress."
        ),
    }


def _run_export_job(type: str, previous_status: str = "finished", hf_repo_id: str = ""):
    # Create unique temp export directory and final zip destination.
    output_base = "./outputs"
    os.makedirs(output_base, exist_ok=True)
    export_dir = os.path.join(output_base, f"_tmp_export_{type}")
    zip_path = os.path.abspath(os.path.join(output_base, f"export_{type}.zip"))

    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir, ignore_errors=True)
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    os.makedirs(export_dir, exist_ok=True)

    try:
        ss.state["loading_progress"] = 3
        ss.state["loading_message"] = "Initializing export..."
        
        if type == "lora":
            from peft import PeftModel
            model = ss.state.get("model")
            if model is None:
                raise ValueError("No model loaded for export.")
            if not isinstance(model, PeftModel):
                raise ValueError(
                    "LoRA export requires a PEFT/LoRA model. "
                    "Current model appears to be a full base/merged model."
                )

            ss.state["loading_progress"] = 12
            ss.state["loading_message"] = "Saving LoRA adapter weights..."
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(export_dir)
            ss.state["loading_progress"] = 42
            ss.state["loading_message"] = "Saving tokenizer files..."
            if ss.state.get("tokenizer") is not None and hasattr(ss.state["tokenizer"], "save_pretrained"):
                ss.state["tokenizer"].save_pretrained(export_dir)

            # ── Push LoRA to HF if repo ID provided ───────────────────────
            if hf_repo_id:
                ss.state["loading_progress"] = 50
                ss.state["loading_message"] = f"Pushing LoRA adapters to {hf_repo_id}..."
                try:
                    hf_token = os.getenv("HF_TOKEN")
                    if not hf_token:
                        raise EnvironmentError("HF_TOKEN not set in environment.")
                    ss.state["hf_push_status"]  = "pushing"
                    ss.state["hf_push_message"] = f"Uploading LoRA to {hf_repo_id}..."
                    model.push_to_hub(hf_repo_id, token=hf_token, private=False)
                    ss.state["tokenizer"].push_to_hub(hf_repo_id, token=hf_token, private=False)
                    repo_url = f"https://huggingface.co/{hf_repo_id}"
                    ss.state["hf_push_status"]  = "pushed"
                    ss.state["hf_push_message"] = f"Pushed → {repo_url}"
                    ss.state["hf_pushed_repo"]  = repo_url
                    print(f"[HF Push] LoRA adapter pushed: {repo_url}")
                except Exception as push_err:
                    ss.state["hf_push_status"]  = "error"
                    ss.state["hf_push_message"] = f"HF push failed: {str(push_err)}"
                    print(f"[HF Push] ERROR: {push_err}")

            _write_model_card(export_dir)
            ss.state["loading_progress"] = 55
            ss.state["loading_message"] = "Packaging ZIP artifact (fast mode)..."
            _zip_directory_with_progress(export_dir, zip_path, start=55, end=98, compression=zipfile.ZIP_STORED)

            shutil.rmtree(export_dir, ignore_errors=True)
            ss.state["loading_progress"] = 100
            ss.state["loading_message"] = "Export complete."
            ss.state["status"] = "finished"
            ss.state["export_artifact_path"] = zip_path
            ss.state["export_download_url"] = f"/api/download-export?path={quote_plus(zip_path)}"
            return

        elif type == "merged":
            from peft import PeftModel
            model = ss.state.get("model")
            if model is None:
                raise HTTPException(status_code=400, detail="No model to merge.")
            
            ss.state["loading_progress"] = 10
            ss.state["loading_message"] = "Preparing merged export..."
            if isinstance(model, PeftModel):
                # Merge the weights and unload LoRA config
                ss.state["loading_progress"] = 30
                ss.state["loading_message"] = "Merging LoRA weights into base model..."
                merged_model = model.merge_and_unload()
                ss.state["loading_progress"] = 62
                ss.state["loading_message"] = "Saving merged model files..."
                merged_model.save_pretrained(export_dir)
            else:
                # The model is not a PEFT model (maybe just base model loaded), just save it
                ss.state["loading_progress"] = 55
                ss.state["loading_message"] = "Saving model files..."
                model.save_pretrained(export_dir)
                merged_model = model

            if ss.state.get("tokenizer"):
                ss.state["loading_progress"] = 74
                ss.state["loading_message"] = "Saving tokenizer files..."
                ss.state["tokenizer"].save_pretrained(export_dir)

            # ── Push merged model to HF if repo ID provided ───────────────
            if hf_repo_id:
                ss.state["loading_progress"] = 78
                ss.state["loading_message"] = f"Pushing merged model to {hf_repo_id}..."
                try:
                    hf_token = os.getenv("HF_TOKEN")
                    if not hf_token:
                        raise EnvironmentError("HF_TOKEN not set in environment.")
                    ss.state["hf_push_status"]  = "pushing"
                    ss.state["hf_push_message"] = f"Uploading merged model to {hf_repo_id}..."
                    merged_model.push_to_hub(hf_repo_id, token=hf_token, private=False)
                    ss.state["tokenizer"].push_to_hub(hf_repo_id, token=hf_token, private=False)
                    repo_url = f"https://huggingface.co/{hf_repo_id}"
                    ss.state["hf_push_status"]  = "pushed"
                    ss.state["hf_push_message"] = f"Pushed → {repo_url}"
                    ss.state["hf_pushed_repo"]  = repo_url
                    print(f"[HF Push] Merged model pushed: {repo_url}")
                except Exception as push_err:
                    ss.state["hf_push_status"]  = "error"
                    ss.state["hf_push_message"] = f"HF push failed: {str(push_err)}"
                    print(f"[HF Push] ERROR: {push_err}")
            
            _write_model_card(export_dir)
            ss.state["loading_progress"] = 82
            ss.state["loading_message"] = "Packaging ZIP artifact (fast mode)..."
            _zip_directory_with_progress(export_dir, zip_path, start=82, end=98, compression=zipfile.ZIP_STORED)

            shutil.rmtree(export_dir, ignore_errors=True)
            ss.state["loading_progress"] = 100
            ss.state["loading_message"] = "Export complete."
            ss.state["status"] = "finished"
            ss.state["export_artifact_path"] = zip_path
            ss.state["export_download_url"] = f"/api/download-export?path={quote_plus(zip_path)}"
            return

    except Exception as e:
        # Keep export failure scoped to export UI; do not poison global pipeline state.
        restore_status = previous_status if previous_status not in {"exporting", "error"} else "finished"
        ss.state["status"] = restore_status
        ss.state["loading_message"] = f"Export failed: {str(e)}"
        ss.state["loading_progress"] = 0
        ss.state["export_error"] = str(e)
        if os.path.isdir(export_dir):
            shutil.rmtree(export_dir, ignore_errors=True)
        import traceback; traceback.print_exc()

@app.post("/api/unload-model", tags=["Team 1 — Data & Model"])
def unload_model_endpoint():
    """Unloads model to free GPU memory."""
    if ss.state.get("model") is not None:
        del ss.state["model"]
    if ss.state.get("tokenizer") is not None:
        del ss.state["tokenizer"]
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    ss.state["model"] = None
    ss.state["tokenizer"] = None
    ss.state["model_name"] = None
    ss.state["model_loaded"] = False
    ss.state["eval_results"] = None
    ss.state["status"] = "idle"
    ss.state["loading_progress"] = 0
    ss.state["loading_message"] = ""
    ss.state["export_download_url"] = ""
    ss.state["export_artifact_path"] = ""
    ss.state["export_type"] = ""
    ss.state["export_error"] = ""
    return {"success": True, "message": "Model unloaded."}

@app.post("/api/cancel", tags=["Team 1 — Data & Model"])
def cancel_operation(request: CancelRequest):
    op = (request.operation or "").strip().lower()
    if op == "loading_model" and ss.state.get("status") == "loading_model":
        ss.state["cancel_requested"] = True
        ss.state["status"] = "idle"
        ss.state["loading_message"] = "Cancel requested..."
        return {"success": True, "message": "Cancellation requested for model loading."}
    raise HTTPException(status_code=400, detail=f"No active cancellable operation for '{op}'.")

@app.get("/api/download-export", tags=["Team 2 — Training & Evaluation"])
def download_export(path: str):
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Export artifact not found.")
    if not abs_path.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only zip artifacts can be downloaded.")
    return FileResponse(path=abs_path, filename=os.path.basename(abs_path), media_type="application/zip")

# ── Helpers ───────────────────────────────────────────────────────────────────

import psutil

def _get_gpu_info() -> dict:
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True,
                "name": props.name,
                "used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
                "total_gb": round(props.total_memory / 1e9, 3),
            }
        if torch.backends.mps.is_available():
            return {
                "available": True,
                "name": f"Apple Silicon GPU ({platform.machine()})",
                "used_gb": round(psutil.virtual_memory().used / 1e9, 3),
                "total_gb": round(psutil.virtual_memory().total / 1e9, 3),
            }
        return {"available": False, "name": "CPU Mode", "used_gb": 0.0, "total_gb": 0.0}
    except Exception:
        return {"available": False, "name": "CPU Mode", "used_gb": 0.0, "total_gb": 0.0}

def _get_ram_info() -> dict:
    try:
        mem = psutil.virtual_memory()
        return {
            "used": round(mem.used / 1e9, 1),
            "total": round(mem.total / 1e9, 1)
        }
    except:
        return {"used": 0.0, "total": 0.0}

def _write_model_card(path: str):
    card = {
        "base_model": ss.state.get("model_name"),
        "fine_tuned_on": "CA Legislature QA",
        "dataset_size": ss.state.get("dataset_size"),
        "lora_config": ss.state.get("lora_config"),
        "eval_results": ss.state.get("eval_results"),
    }
    with open(f"{path}/model_card.json", "w") as f:
        json.dump(card, f, indent=2, default=str)


def _get_model_type(model) -> str:
    if model is None:
        return "none"
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            return "peft_lora"
    except Exception:
        pass
    return "merged_base"


def _zip_directory_with_progress(
    source_dir: str,
    zip_path: str,
    start: int = 60,
    end: int = 98,
    compression: int = zipfile.ZIP_STORED,
):
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for name in files:
            file_paths.append(os.path.join(root, name))
    total_bytes = sum(max(0, os.path.getsize(p)) for p in file_paths)
    processed_bytes = 0
    total_files = max(1, len(file_paths))

    with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
        for idx, abs_file in enumerate(file_paths, start=1):
            arcname = os.path.relpath(abs_file, source_dir)
            zf.write(abs_file, arcname)
            processed_bytes += max(0, os.path.getsize(abs_file))
            if total_bytes > 0:
                frac = processed_bytes / total_bytes
                done = f"{processed_bytes / (1024 * 1024):.1f}MB/{total_bytes / (1024 * 1024):.1f}MB"
            else:
                frac = idx / total_files
                done = f"{idx}/{total_files} files"
            ss.state["loading_progress"] = int(start + (end - start) * frac)
            ss.state["loading_message"] = f"Zipping files ({done})..."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)