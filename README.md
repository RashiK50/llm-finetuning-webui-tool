# TCS FineTuner

End-to-end web UI + API for LoRA fine-tuning, evaluation, chat testing, and export of LLMs.

## What This Project Does

- Loads a Hugging Face model for local fine-tuning.
- Uploads and validates JSON datasets with:
  - `instruction`
  - `reasoning`
  - `output`
- Trains with LoRA using `transformers` + `peft`.
- Evaluates with ROUGE-L and Exact Match, including reasoning/output extraction.
- Exports model artifacts as ZIP downloads from the UI:
  - LoRA adapter export (only valid for PEFT/LoRA model state)
  - Full merged model export
- Streams live status to frontend via SSE (`/api/status-stream`) for:
  - model/dataset state
  - training state
  - export progress + messages

---

## Architecture

- **Backend:** FastAPI (`main.py`) + shared in-memory state (`shared_state.py`)
- **Frontend:** React + Vite (`tcs-frontend/`)
- **Model stack:** `transformers`, `datasets`, `peft`, `torch`

---

## Current UI Workflow

1. **Model Setup**: load base model.
2. **Dataset**: upload JSON dataset.
3. **Training**: configure hyperparameters and run LoRA training.
4. **Evaluation**: run sample evaluation with reasoning/response display.
5. **Export**:
   - Shows model type (`PEFT LoRA` or `Merged/Base`).
   - Disables incompatible LoRA export automatically.
   - Shows live export percentage + step message.
   - Auto-downloads resulting ZIP artifact when ready.
6. **Chat Playground**: interact with current loaded model.

---

## API Endpoints (Main)

### Status / Control
- `GET /api/status`
- `GET /api/status-stream` (SSE)
- `POST /api/cancel`

### Model / Data
- `POST /api/load-model`
- `POST /api/unload-model`
- `POST /api/upload-dataset`
- `POST /api/clear-dataset`

### Training / Eval / Export
- `POST /api/train`
- `GET /api/progress`
- `POST /api/evaluate`
- `GET /api/export?type=lora|merged` (starts background export job)
- `GET /api/download-export?path=...` (serves ZIP artifact)
- `POST /api/generate`

---

## Export Behavior (Important)

- Export runs in background and updates:
  - `status` (`exporting`, `finished`, etc.)
  - `loading_progress` (percentage)
  - `loading_message` (current export stage)
  - `export_download_url`, `export_artifact_path`, `export_error`
- **LoRA export requires PEFT/LoRA model state.**
  - If current model is merged/base, LoRA export is blocked by design.
- ZIP packaging is in fast mode (`ZIP_STORED`) for better speed.

---

## Setup

## 1) Backend

```bash
pip install -r requirements.txt
python main.py
```

Backend runs on `http://localhost:8000`.

If needed, set `.env`:

```env
HF_TOKEN=your_huggingface_token
```

## 2) Frontend

```bash
cd tcs-frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.

---

## Dataset Format

Upload JSON array like:

```json
[
  {
    "instruction": "What does California AB 2930 regulate?",
    "reasoning": "AB 2930 focuses on algorithmic discrimination...",
    "output": "California AB 2930 requires businesses using automated decision tools..."
  }
]
```

---

## Notes

- Offline backend detection is surfaced in UI and persistent toast until reconnect.
- System Resource Overview reflects backend online/offline state.
- Export UI shows model type and compatibility to prevent invalid export actions.
