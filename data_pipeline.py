# data_pipeline.py
from datasets import Dataset
import shared_state as ss  # Ensure this is also present if not already

def format_prompt(row):
    # This string now includes the new 'reasoning' field
    formatted_string = f"### Instruction:\n{row['instruction']}\n\n### Reasoning:\n{row['reasoning']}\n\n### Response:\n{row['output']}"
    
    # FIX: Return a dictionary, not just the string
    return {"text": formatted_string}

def validate_and_load(data):
    # Ensure all rows have the necessary keys including the new 'reasoning'
    clean = []
    for item in data:
        if all(key in item for key in ["instruction", "reasoning", "output"]):
            clean.append(item)
    
    if not clean:
        raise ValueError("No valid entries found. Ensure 'instruction', 'reasoning', and 'output' keys exist.")

    # Convert to HuggingFace Dataset and apply the fixed mapping function
    dataset = Dataset.from_list(clean).map(format_prompt)
    
    # Split and save to shared state
    split = dataset.train_test_split(test_size=0.2)
    ss.state["train_dataset"] = split["train"]
    ss.state["test_dataset"] = split["test"]
    
    return {"status": "success", "samples": len(clean)}