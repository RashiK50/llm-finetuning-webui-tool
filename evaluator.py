# evaluator.py — Team 2
# Runs evaluation on the fine-tuned model
# Returns ROUGE-L, Exact Match, and sample outputs including Chain of Thought

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

        if model is None or test_dataset is None:
            raise ValueError("Model or Dataset not loaded.")

        # --- Dynamic Device Detection ---
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        model.to(device) # Ensure model is on the correct device

        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores, exact_matches, samples = [], [], []

        eval_data = test_dataset.select(range(min(num_samples, len(test_dataset))))

        for item in eval_data:
            instruction = item["instruction"]
            expected = item["output"]

            # Changed prompt to trigger Chain of Thought reasoning FIRST
            prompt = f"### Instruction:\n{instruction}\n\n### Reasoning:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300, # Increased token limit to accommodate reasoning + response
                    do_sample=False,
                    repetition_penalty=1.1
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract everything generated after the prompt
            generated_content = generated[len(prompt):].strip()
            
            # Split the reasoning from the final response
            if "### Response:" in generated_content:
                model_reasoning, model_answer = generated_content.split("### Response:", 1)
                model_reasoning = model_reasoning.strip()
                model_answer = model_answer.strip()
            else:
                # Fallback if the model fails to output the Response tag
                model_reasoning = "Model failed to format Chain of Thought properly."
                model_answer = generated_content.strip()

            # Metric Calculation (Only grading the final answer, not the reasoning)
            exact = 1 if model_answer.strip() == expected.strip() else 0
            exact_matches.append(exact)
            score = scorer.score(expected, model_answer)
            rouge_scores.append(score["rougeL"].fmeasure)

            samples.append({
                "question": instruction,
                "expected": expected,
                "model_reasoning": model_reasoning,
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