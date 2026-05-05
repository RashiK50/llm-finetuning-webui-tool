from datasets import Dataset
import shared_state as ss

REQUIRED_FIELDS = {"instruction", "output"}


def format_prompt(row):
    return f"### Instruction:\n{row['instruction']}\n\n### Reasoning:\n{row['reasoning']}\n\n### Response:\n{row['output']}"


def validate_and_load(data: list) -> dict:
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("JSON must be a non-empty list of QA pairs.")

    warnings = []
    seen = set()
    clean = []

    for i, item in enumerate(data):
        if not REQUIRED_FIELDS.issubset(item.keys()):
            warnings.append(f"Entry {i}: missing 'instruction' or 'output'. Skipped.")
            continue
        if not item["instruction"].strip() or not item["output"].strip():
            warnings.append(f"Entry {i}: empty field. Skipped.")
            continue
        key = item["instruction"].strip().lower()
        if key in seen:
            warnings.append(f"Entry {i}: duplicate. Skipped.")
            continue
        seen.add(key)
        clean.append(item)

    if not clean:
        raise ValueError("No valid QA pairs found after validation.")

    dataset = Dataset.from_list(clean).map(format_prompt)
    split = dataset.train_test_split(test_size=0.2, seed=42)

    ss.state.update({
        "dataset": split,
        "train_dataset": split["train"],
        "test_dataset": split["test"],
        "dataset_size": len(clean),
        "status": "dataset_ready",
    })

    return {
        "success": True,
        "total": len(clean),
        "train": len(split["train"]),
        "test": len(split["test"]),
        "warnings": warnings,
        "preview": clean[:5],
    }