from datasets import Dataset
import shared_state as ss


def format_prompt(row):
    formatted_string = (
        f"### Instruction:\n{row['instruction']}\n\n"
        f"### Response:\n{row['output']}"
    )

    return {"text": formatted_string}


def _normalise_item(item):
    """
    Accept both the original schema (instruction / reasoning / output)
    and Group 2's schema (Question / Analysis / Answer).
    Returns a normalised dict or None if the item is unusable.
    """
    # Group 2 schema takes priority when present
    if "Question" in item and "Answer" in item:
        return {
            "instruction": str(item.get("Question", "")).strip(),
            "reasoning":   str(item.get("Analysis", "")).strip(),
            "output":      str(item.get("Answer", "")).strip(),
        }
    # Original schema
    if "instruction" in item and "output" in item:
        return {
            "instruction": str(item.get("instruction", "")).strip(),
            "reasoning":   str(item.get("reasoning", "")).strip(),
            "output":      str(item.get("output", "")).strip(),
        }
    return None


def validate_and_load(data):
    """
    Validates, normalises, and loads a JSON dataset into shared state.
    Splits 90 / 10 into train and test.
    Supports both original (instruction/output) and Group 2 (Question/Analysis/Answer) schemas.
    """
    clean = []

    for item in data:
        normalised = _normalise_item(item)
        if normalised:
            clean.append(normalised)

    if not clean:
        raise ValueError(
            "No valid entries found. "
            "Ensure keys are either 'instruction'+'output' OR 'Question'+'Answer'."
        )

    dataset = Dataset.from_list(clean).map(format_prompt)

    print("\n===== DATASET DEBUG =====")
    print("FIRST DATASET ENTRY:")
    print(dataset[0])
    print("=========================\n")

    # Use 90/10 split — proper held-out evaluation set
    split = dataset.train_test_split(test_size=0.1, seed=42)

    print("\n===== TEST DATASET DEBUG =====")
    print(split["test"][0])
    print("==============================\n")

    ss.state["train_dataset"] = split["train"]
    ss.state["test_dataset"]  = split["test"]
    ss.state["dataset_size"]  = len(clean)
    ss.state["dataset_preview"] = clean[:5]

    return {
        "status": "success",
        "size":    len(clean),
        "preview": clean[:5],
    }


def validate_and_load_eval_only(data):
    """
    Loads a SEPARATE evaluation-only dataset (different question phrasing).
    Only populates ss.state['test_dataset'] — never touches train_dataset.
    """
    clean = []

    for item in data:
        normalised = _normalise_item(item)
        if normalised:
            clean.append(normalised)

    if not clean:
        raise ValueError(
            "No valid entries found in eval dataset. "
            "Ensure keys are either 'instruction'+'output' OR 'Question'+'Answer'."
        )

    dataset = Dataset.from_list(clean).map(format_prompt)

    print("\n===== EVAL-ONLY DATASET DEBUG =====")
    print("FIRST EVAL ENTRY:")
    print(dataset[0])
    print("===================================\n")

    ss.state["test_dataset"] = dataset

    return {
        "status":  "success",
        "size":    len(clean),
        "preview": clean[:5],
    }