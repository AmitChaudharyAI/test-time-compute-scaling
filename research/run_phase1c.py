#!/usr/bin/env python3
"""Run the frozen five-example Phase 1C extraction validation."""

import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DATASET_ID = "HuggingFaceH4/MATH-500"
INDICES = [0, 1, 2, 3, 4]
SEED = 42
GENERATION = {
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "max_new_tokens": 1024,
}


def balanced_braced_content(text: str, open_pos: int):
    if open_pos >= len(text) or text[open_pos] != "{":
        return None, None
    depth = 0
    for pos in range(open_pos, len(text)):
        if text[pos] == "{" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1
        elif text[pos] == "}" and (pos == 0 or text[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[open_pos + 1 : pos], pos + 1
    return None, None


def extract_answer(generation: str):
    marker = "Final Answer:"
    marker_pos = generation.rfind(marker)
    if marker_pos >= 0:
        trailing_lines = generation[marker_pos + len(marker) :].splitlines()
        line = trailing_lines[0].strip() if trailing_lines else ""
        if line:
            return line, "final_answer_marker", "OK"
        return None, None, "INVALID_EXTRACTION"

    candidates = []
    start = 0
    while True:
        boxed_pos = generation.find("\\boxed{", start)
        if boxed_pos < 0:
            break
        content, end = balanced_braced_content(generation, boxed_pos + len("\\boxed"))
        if content is not None and content.strip():
            candidates.append(content.strip())
            start = end
        else:
            start = boxed_pos + len("\\boxed{")
    if candidates:
        return candidates[-1], "boxed_fallback", "OK"
    return None, None, "INVALID_EXTRACTION"


def unwrap_complete(value: str, command: str):
    prefix = command + "{"
    if not value.startswith(prefix):
        return value
    content, end = balanced_braced_content(value, len(command))
    return content if content is not None and end == len(value) else value


def normalize_answer(answer):
    if answer is None:
        return None
    value = answer.strip()
    while len(value) >= 2 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    for wrapper in ("\\boxed", "\\text", "\\mathrm", "\\operatorname"):
        value = unwrap_complete(value, wrapper).strip()
    value = re.sub(r"\\(?:,|!|;|:| )", "", value)
    return re.sub(r"\s+", " ", value).strip()


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Phase 1C requires GPU 0")

    prompt_template = (ROOT / "research/pilot_prompt_template_v2.txt").read_text()
    dataset = load_dataset(DATASET_ID, split="test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    set_seed(SEED)

    output_path = ROOT / "results/raw/phase1c_extraction_validation.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output_path.open("w") as handle:
        for dataset_index in INDICES:
            row = dataset[dataset_index]
            prompt = prompt_template.format(problem=row["problem"])
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            input_tokens = encoded.input_ids.shape[1]
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    **GENERATION,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize(0)
            latency = time.perf_counter() - started
            continuation = generated[0, input_tokens:]
            raw_generation = tokenizer.decode(continuation, skip_special_tokens=True)
            raw_answer, source, status = extract_answer(raw_generation)
            normalized = normalize_answer(raw_answer)
            reference_normalized = normalize_answer(row["answer"])
            extraction_success = status == "OK"
            record = {
                "dataset_index": dataset_index,
                "question": row["problem"],
                "reference_answer": row["answer"],
                "prompt": prompt,
                "seed": SEED,
                "raw_generation": raw_generation,
                "extraction_source": source,
                "raw_extracted_answer": raw_answer,
                "normalized_answer": normalized,
                "extraction_success": extraction_success,
                "status": status,
                "correct": extraction_success and normalized == reference_normalized,
                "input_tokens": input_tokens,
                "output_tokens": int(continuation.shape[0]),
                "latency_seconds": round(latency, 6),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated(0) / 2**20, 2),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            print(f"index={dataset_index} extraction={status} source={source}", flush=True)

    marker = sum(r["extraction_source"] == "final_answer_marker" for r in records)
    fallback = sum(r["extraction_source"] == "boxed_fallback" for r in records)
    success = sum(r["extraction_success"] for r in records)
    correct = sum(r["correct"] for r in records)
    mean_tokens = sum(r["output_tokens"] for r in records) / len(records)
    mean_latency = sum(r["latency_seconds"] for r in records) / len(records)
    peak_vram = max(r["peak_vram_mb"] for r in records)
    gpu = torch.cuda.get_device_name(0)
    disk_free = __import__("shutil").disk_usage("/workspace").free / 2**30
    status = "READY FOR PHASE 2" if success == 5 else "NEEDS FIX"
    issue_lines = []
    if marker < 5:
        issue_lines.append(
            f"- The required `Final Answer:` marker appeared in only {marker}/5 generations; "
            f"the boxed fallback recovered {fallback}/5."
        )
    if success < 5:
        issue_lines.append("- Extraction failed for one or more preserved examples.")
    issues = "\n".join(issue_lines) if issue_lines else "- None."
    action = "- Human review of Phase 1C artifacts before Phase 2."
    summary = f"""PHASE 1C SUMMARY

MODEL:
{MODEL_ID}

BENCHMARK:
MATH-500

EXAMPLES:
5 (dataset indices: {', '.join(map(str, INDICES))})

GPU:
GPU 0 — {gpu}, bfloat16

DECODING CONFIGURATION:
temperature={GENERATION['temperature']}, top_p={GENERATION['top_p']}, do_sample={GENERATION['do_sample']}, max_new_tokens={GENERATION['max_new_tokens']}, seed={SEED}

FINAL ANSWER MARKER COMPLIANCE:
{marker}/5

BOXED FALLBACK USED:
{fallback}/5

EXTRACTION SUCCESS:
{success}/5

INVALID EXTRACTIONS:
{5 - success}/5

CORRECT ANSWERS:
{correct}/5

MEAN OUTPUT TOKENS:
{mean_tokens:.1f}

MEAN LATENCY:
{mean_latency:.2f} seconds

PEAK VRAM:
{peak_vram:.2f} MiB

SCORING METHOD:
Conservative normalized exact match against the MATH-500 canonical answer field (see protocol; formatting-equivalent answers may be undercounted).

DISK FREE:
{disk_free:.2f} GiB on /workspace after Phase 1C

STATUS:
{status}

ISSUES:
{issues}

RECOMMENDED NEXT ACTION:
{action}
"""
    report = "# Phase 1C Extraction Validation Report\n\n" + summary
    (ROOT / "results/phase1c_extraction_validation_report.md").write_text(report)
    print("\n" + summary)


if __name__ == "__main__":
    main()
