# llm-finetuning-webui-tool

A web-based fine-tuning platform for training Mistral 7B v0.3 on GenAI regulatory Q&A data on California state legislation. Built as part of a TCS industry project.

---

## Overview

This tool allows users to upload a JSON dataset of legislature Q&A pairs, configure training parameters, and fine-tune Mistral 7B using SFT + LoRA — all through a simple web interface without writing any code. The fine-tuned model targets 80-90% accuracy on regulatory questions.

---

## Features

- Upload a JSON file of legislature Q&A pairs via the UI
- Configure LoRA hyperparameters (rank, alpha, learning rate, epochs)
- Train Mistral 7B using Supervised Fine-Tuning and QLoRA
- Monitor live training loss and evaluation metrics
- Export the fine-tuned LoRA adapter or merged model

---

## Project Structure

### this is subject to change

```
llm-finetuning-webui-tool/
├── app/
│   ├── gradio_app.py        # Main Gradio UI
│   ├── trainer.py           # SFT + LoRA training pipeline
│   ├── model_loader.py      # Loads Mistral 7B in 4-bit
│   ├── evaluator.py         # ROUGE, BERTScore, accuracy
│   └── utils.py             # JSON validation and formatting
├── data/
│   ├── sample_ca.json       # Sample California Q&A
│   ├── sample_ny.json       # Sample New York Q&A
│   └── sample_fl.json       # Sample Florida Q&A
├── notebooks/
│   └── training_pipeline.ipynb
├── outputs/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

Clone the repo

```bash
git clone https://github.com/your-username/llm-finetuning-webui-tool.git
cd llm-finetuning-webui-tool
```

Install dependencies

```bash
pip install -r requirements.txt
```

Set up HuggingFace token

```bash
cp .env.example .env
# Add your HF_TOKEN to .env
```

Launch the UI

```bash
python app/gradio_app.py
```

Open your browser at `http://localhost:7860`

---

## Input Data Format

```json
[
  {
    "instruction": "What does California AB 2930 regulate?",
    "output": "California AB 2930 requires businesses using automated decision tools to conduct impact assessments and notify individuals when such tools are used in consequential decisions."
  },
  {
    "instruction": "What are New York's requirements for AI in hiring?",
    "output": "New York Local Law 144 requires employers using AI hiring tools to conduct annual bias audits and publicly disclose results before deploying the tool."
  }
]
```

---

## UI Tabs

| Tab | Description |
|---|---|
| Load Model | Select and load Mistral 7B v0.3 in 4-bit |
| Dataset | Upload legislature JSON and preview Q&A pairs |
| Fine-Tune | Set LoRA rank, epochs, learning rate and start training |
| Evaluate | Run test questions and view accuracy score |
| Export | Download LoRA adapter or merged model |

---

## Model and Training

| Component | Details |
|---|---|
| Base Model | mistralai/Mistral-7B-v0.3 |
| Fine-Tuning Method | Supervised Fine-Tuning (SFT) |
| Parameter Efficiency | LoRA / QLoRA via peft |
| Training Framework | trl SFTTrainer |
| Quantization | 4-bit NF4 via bitsandbytes |
| Evaluation | ROUGE-L, BERTScore, Exact Match |

---

## Cloud Setup

Recommended GCP instance:

```
Machine : n1-standard-8
GPU     : NVIDIA T4 (16GB) or A100 (40GB)
Disk    : 100GB SSD
OS      : Ubuntu 20.04 + CUDA 11.8
```

---

## Tech Stack

- transformers
- trl
- peft
- bitsandbytes
- unsloth
- datasets
- gradio
- wandb
- rouge-score
- bert-score


