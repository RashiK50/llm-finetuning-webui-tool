state = {
    # Model
    "model": None,
    "tokenizer": None,
    "model_name": None,
    "model_loaded": False,
    "quantization_config": None,

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