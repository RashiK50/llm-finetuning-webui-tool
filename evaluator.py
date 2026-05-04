# evaluator.py — Team 2
# Runs evaluation on the fine-tuned model
# Returns ROUGE-L, Exact Match, and sample outputs

import torch
import numpy as np
from rouge_score import rouge_scorer as rouge_lib
import shared_state as ss


def run_evaluation(num_samples: int = 10):
    try:
        ss.state["status"] = "evaluating"

        model = ss.state["model"]
        tokenizer = ss.state["tokenizer"]
        test_dataset = ss.state["test_dataset"]

        if model is None:
            raise ValueError("Model not loaded.")
        if test_dataset is None:
            raise ValueError("Dataset not loaded.")

        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

        rouge_scores = []
        exact_matches = []
        samples = []

        # Cap at available test samples
        eval_data = test_dataset.select(range(min(num_samples, len(test_dataset))))

        for item in eval_data:
            instruction = item["instruction"]
            expected = item["output"]

            prompt = f"### Instruction:\n{instruction}\n\n### Response:"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    repetition_penalty=1.1
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            model_answer = generated.split("### Response:")[-1].strip()

            # Exact match
            exact = 1 if model_answer.strip() == expected.strip() else 0
            exact_matches.append(exact)

            # ROUGE-L
            score = scorer.score(expected, model_answer)
            rouge_scores.append(score["rougeL"].fmeasure)

            samples.append({
                "question": instruction,
                "expected": expected,
                "model_output": model_answer,
                "rouge_l": round(score["rougeL"].fmeasure, 4),
                "exact_match": exact
            })

        avg_rouge = round(float(np.mean(rouge_scores)), 4)
        avg_exact = round(float(np.mean(exact_matches)) * 100, 2)

        result = {
            "rouge_l": avg_rouge,
            "exact_match_percent": avg_exact,
            "samples_evaluated": len(eval_data),
            "samples": samples
        }

        ss.state["eval_results"] = result
        ss.state["status"] = "done"

        return result

    except Exception as e:
        ss.state["status"] = "error"
        ss.state["error_message"] = str(e)
        raise e