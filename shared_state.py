state = {
    # Model
    "model": None,
    "tokenizer": None,
    "train_dataset": None,
    "status": "idle",             # 'idle', 'loading_model', 'training', 'error'
    "is_model_loaded": False,     # Fix 1: Explicit model status
    "is_dataset_loaded": False,   # Fix 1: Explicit dataset status
    "current_step": 0,
    "training_loss": [],
    "error_message": "",

    # Dataset
    "dataset": None,
    "train_dataset": None,
    "test_dataset": None,
    "dataset_size": 0,

    # Training
    "trainer": None,
    "training_loss": [],
    "current_epoch": 0,
    "current_step": 0,
    "lora_config": None,

    # Evaluation
    "eval_results": None,

    # Status
    "status": "idle",
    "error_message": None,
}