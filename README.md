# LLM Fine-Tuning WebUI Tool

A lightweight FastAPI-based backend tool for fine-tuning Large Language Models (like Mistral 7B or TinyLlama) using Supervised Fine-Tuning (SFT) and LoRA (Low-Rank Adaptation). It is designed to fine-tune an LLM on California Legislature QA data utilizing a specialized Chain of Thought strategy.

## Features

- **Model Loading:** Supports loading models onto GPUs in optimized formats using Hugging Face `transformers` and `bitsandbytes` (4-bit NF4 + bfloat16 representation when deployed appropriately). By default, uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for local testing and `mistralai/Mistral-7B-v0.3` for production.
- **Dataset Pipeline:** Upload datasets in JSON format (`instruction`, `reasoning`, `output`). The pipeline automatically formats prompts into a predefined structure to prompt Chain of Thought (CoT).
- **Training (LoRA):** Configure LoRA hyperparameters (rank, alpha, dropout) directly through the API. Training utilizes `peft` and Hugging Face's `Trainer` for efficient, distributed training with status polling logic.
- **Evaluation:** Evaluates the fine-tuned model against a test data split, tracking Exact Match and ROUGE-L scores using `rouge-score`. Examines the generated Chain of Thought reasoning vs. the final response.
- **Exporting Options:** Export either just the LoRA adapter (~100MB zip) or the fully merged model zip (~14GB).

## API Endpoints

The API is divided into two conceptual operations. You can access the Swagger UI directly at `/docs` (e.g., `http://localhost:8000/docs`).

### Data & Model
- `GET /api/status` - Check the current application state, model readiness, dataset size, and GPU usage.
- `POST /api/load-model` - Load a base Hugging Face model onto the device into persistent caching.
- `POST /api/upload-dataset` - Upload the JSON task dataset.

### Training & Evaluation
- `POST /api/train` - Kick off the background training task with custom training and LoRA parameters.
- `GET /api/progress` - Long-poll endpoint to retrieve live training loss and epochs.
- `POST /api/evaluate` - Test the fine-tuned model using a test dataset split to visualize text generation mapping and retrieve ROUGE/Exact Match metrics.
- `GET /api/export` - Package model checkpoints into zip maps. Download LoRA adapters or a fully merged model.

## Installation

Make sure your environment is configured for GPU access if you plan to train on-device (CUDA strongly recommended).

1. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your `.env` file (requires Hugging Face token to download certain models like Mistral).
   ```env
   HF_TOKEN=your_hf_token_here
   ```

## Usage

1. **Start the API server:**
   ```bash
   python main.py
   ```
   The sever will start on `http://0.0.0.0:8000`.

2. **Access Swagger UI:**
   Navigate to `http://localhost:8000/docs` to test out the endpoints. Ensure you call the functions sequentially: 
   1. `/api/load-model` 
   2. `/api/upload-dataset` 
   3. `/api/train`

## Tech Stack
- **FastAPI / Uvicorn** - Web Server
- **Hugging Face (`transformers`, `datasets`, `peft`, `trl`)** - Model pipelines, datasets, and LoRA
- **PyTorch / Accelerate** - Base computational graph
- **Rouge Score** - Response Evaluation metric
