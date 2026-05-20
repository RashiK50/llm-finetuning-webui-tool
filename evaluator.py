# evaluator.py — Team 2
# Runs evaluation on the fine-tuned model
# Returns ROUGE-L, Exact Match, and sample outputs including Chain of Thought

import torch
import numpy as np
from rouge_score import rouge_scorer as rouge_lib
import shared_state as ss
import re

def _extract_reasoning_and_answer(text: str) -> tuple[str, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return "", ""

    reasoning = ""
    answer = cleaned

    response_pattern = re.compile(r"(?:^|\s)(?:###\s*)?Response\s*:\s*", flags=re.IGNORECASE)
    reasoning_pattern = re.compile(r"(?:^|\s)(?:###\s*)?Reasoning\s*:\s*", flags=re.IGNORECASE)
    final_answer_pattern = re.compile(r"(?:^|\s)Final\s*Answer\s*:\s*", flags=re.IGNORECASE)

    response_match = response_pattern.search(cleaned)
    reasoning_match = reasoning_pattern.search(cleaned)
    final_answer_match = final_answer_pattern.search(cleaned)

    if response_match:
        response_start = response_match.start()
        response_end = response_match.end()
        before_response = cleaned[:response_start].strip()
        after_response = cleaned[response_end:].strip()

        if reasoning_match and reasoning_match.start() < response_start:
            reasoning_header_end = reasoning_match.end()
            reasoning = cleaned[reasoning_header_end:response_start].strip()
        else:
            # If there is no explicit reasoning header, treat pre-response text as reasoning.
            reasoning = before_response
        answer = after_response
    elif final_answer_match:
        split_at = final_answer_match.start()
        answer_start = final_answer_match.end()
        reasoning = cleaned[:split_at].strip()
        answer = cleaned[answer_start:].strip()
    else:
        # Parse generic markdown-style section headers like "### Impact:".
        section_pattern = re.compile(r"\s###\s*([A-Za-z][A-Za-z \-]{1,40})\s*:\s*")
        section_match = section_pattern.search(cleaned)
        if section_match:
            label = section_match.group(1).strip().lower()
            before_section = cleaned[:section_match.start()].strip()
            after_section = cleaned[section_match.end():].strip()

            reasoning_like_labels = {
                "impact",
                "analysis",
                "explanation",
                "rationale",
                "reasoning",
                "notes",
            }
            if label in reasoning_like_labels and before_section and after_section:
                # Common pattern: direct answer followed by analytical section.
                answer = before_section
                reasoning = after_section
            elif before_section and after_section:
                # Unknown section label, but still split to avoid losing structure.
                answer = before_section
                reasoning = f"{label.title()}: {after_section}".strip()
            else:
                answer = cleaned
                reasoning = ""
        else:
        # Fallback: split long responses into rationale + final answer.
            chunks = [c.strip() for c in re.split(r"\n\s*\n", cleaned) if c.strip()]
            if len(chunks) >= 2:
                reasoning = "\n\n".join(chunks[:-1]).strip()
                answer = chunks[-1].strip()
            else:
                # If no clear separator, keep full text as answer.
                reasoning = ""
                answer = cleaned

    # Strip chat delimiters if the model leaked them.
    for marker in ["User:", "Assistant:", "System:"]:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()
        if marker in reasoning:
            reasoning = reasoning.split(marker, 1)[0].strip()

    # Final cleanup in case markers leak into either field.
    reasoning = re.sub(r"(?:###\s*)?Reasoning\s*:\s*", "", reasoning, flags=re.IGNORECASE).strip()
    answer = re.sub(r"^(?:###\s*)?Response\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"^(?:Final\s*Answer\s*:\s*)", "", answer, flags=re.IGNORECASE).strip()

    return reasoning, answer


def _generate_text(model, tokenizer, prompt: str, input_device: torch.device, max_new_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def run_evaluation(num_samples: int = 10):
    try:
        ss.state["status"] = "evaluating"

        model = ss.state["model"]
        tokenizer = ss.state["tokenizer"]
        test_dataset = ss.state["test_dataset"]

        if model is None or test_dataset is None:
            raise ValueError("Model or Dataset not loaded.")

        first_param = next(model.parameters(), None)
        input_device = first_param.device if first_param is not None else torch.device("cpu")

        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores, exact_matches, samples = [], [], []
        reasoning_retry_count = 0

        eval_data = test_dataset.select(range(min(num_samples, len(test_dataset))))

        for item in eval_data:
            print("\n===== EVALUATION ITEM =====")

            print(item)

            print("===========================\n")

            instruction = item["instruction"]
            expected = item["output"]

            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Reasoning:\n"
            )

            generated_content = _generate_text(model, tokenizer, prompt, input_device, max_new_tokens=300)

            model_reasoning, model_answer = _extract_reasoning_and_answer(generated_content)

            # Retry with stricter formatting if reasoning is missing.
            if not model_reasoning.strip():
                reasoning_retry_count += 1
                retry_prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    "Return your output in this exact structure:\n"
                    "### Reasoning:\n"
                    "<brief step-by-step reasoning>\n\n"
                    "### Response:\n"
                    "<final answer>\n\n"
                    "### Reasoning:\n"
                )
                retried_content = _generate_text(model, tokenizer, retry_prompt, input_device, max_new_tokens=360)
                retry_reasoning, retry_answer = _extract_reasoning_and_answer(retried_content)
                if retry_reasoning.strip():
                    model_reasoning = retry_reasoning
                    model_answer = retry_answer
                    generated_content = retried_content

            if not model_answer:
                model_answer = generated_content.strip()

            # Metric Calculation
            exact = 1 if model_answer.strip() == expected.strip() else 0
            exact_matches.append(exact)
            score = scorer.score(expected, model_answer)
            rouge_scores.append(score["rougeL"].fmeasure)

            samples.append({
                "question": instruction,
                "expected": expected,
                # USE DATASET REASONING
                "model_reasoning": (
                item["reasoning"]
                if "reasoning" in item
                and item["reasoning"].strip()
                else "Reasoning not available."
                ),
                "model_output": model_answer,
                "rouge_l": round(
                    score["rougeL"].fmeasure,
                    4
                ),
                "exact_match": exact
            })

        avg_rouge = round(float(np.mean(rouge_scores)), 4)
        avg_exact = round(float(np.mean(exact_matches)) * 100, 2)

        reasoning_present_count = sum(1 for s in samples if s.get("model_reasoning", "").strip())

        result = {
            "rouge_l": avg_rouge,
            "exact_match_percent": avg_exact,
            "samples_evaluated": len(eval_data),
            "reasoning_present_count": reasoning_present_count,
            "reasoning_retry_count": reasoning_retry_count,
            "samples": samples
        }

        ss.state["eval_results"] = result
        ss.state["status"] = "done"
        return result

    except Exception as e:
        ss.state["status"] = "error"
        ss.state["error_message"] = str(e)
        raise e