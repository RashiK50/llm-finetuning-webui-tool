state = {
    # Model
    "model": None,
    "tokenizer": None,
    "model_name": None,
    "model_loaded": False,
    "train_dataset": None,
    "status": "idle",             # 'idle', 'loading_model', 'training', 'error'
    "is_model_loaded": False,
    "is_dataset_loaded": False,
    "current_step": 0,
    "training_loss": [],
    "error_message": "",
    "loading_progress": 0,
    "loading_message": "",
    "cancel_requested": False,
    "export_download_url": "",
    "export_artifact_path": "",
    "export_type": "",
    "export_error": "",

    # Dataset
    "dataset": None,
    "train_dataset": None,
    "test_dataset": None,
    "dataset_size": 0,
    "dataset_name": "",
    "dataset_preview": [],

    # Training
    "trainer": None,
    "training_loss": [],
    "current_epoch": 0,
    "current_step": 0,
    "lora_config": None,

    # Evaluation
    "eval_results": None,

    # HF Push status (new)
    "hf_push_status": "",         # '', 'pushing', 'pushed', 'error'
    "hf_push_message": "",
    "hf_pushed_repo": "",         # repo URL after successful push
    "gguf_push_status": "",       # '', 'converting', 'pushing', 'pushed', 'error'
    "gguf_push_message": "",
    "gguf_pushed_repo": "",

    # Status
    "status": "idle",
    "error_message": None,
}
