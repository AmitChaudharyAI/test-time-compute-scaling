#!/usr/bin/env python3
"""Resumable full MATH-500 x 16 generation experiment."""

import json
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from math_scoring import math_equal, vote_key
from run_phase1c import extract_answer, normalize_answer


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_REVISION = "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
DATASET_ID = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SAMPLES = 16
BASE_SEED = 20260902
GENERATION = {"temperature": 0.7, "top_p": 0.9, "do_sample": True, "max_new_tokens": 1024}
RAW_PATH = ROOT / "results/raw/generations.jsonl"
STATE_PATH = ROOT / "results/raw/main_experiment_state.json"
MIN_FREE_BYTES = 5 * 2**30


def derived_seed(question_index, sample_index):
    return BASE_SEED + question_index * SAMPLES + (sample_index - 1)


@contextmanager
def independent_multinomial_streams(seeds):
    """Give each batch row its own seeded CUDA RNG without changing sampling."""
    original = torch.multinomial
    generators = [torch.Generator(device="cuda:0").manual_seed(seed) for seed in seeds]

    def per_row(input, num_samples, replacement=False, *, generator=None, out=None):
        if input.ndim == 2 and input.shape[0] == len(generators) and generator is None and out is None:
            return torch.stack([
                original(row, num_samples, replacement, generator=row_generator)
                for row, row_generator in zip(input, generators)
            ])
        return original(input, num_samples, replacement, generator=generator, out=out)

    torch.multinomial = per_row
    try:
        yield
    finally:
        torch.multinomial = original


def load_completed():
    completed = set()
    if not RAW_PATH.exists():
        return completed
    with RAW_PATH.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed checkpoint line {line_number}; refusing unsafe resume") from error
            key = (row["question_index"], row["sample_index"])
            if key in completed:
                raise RuntimeError(f"Duplicate checkpoint key {key}")
            completed.add(key)
    return completed


def write_state(started_at, completed, status, peak_vram_mb=0.0):
    state = {
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_generations": len(completed),
        "target_generations": 500 * SAMPLES,
        "peak_vram_mb": peak_vram_mb,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "decoding": GENERATION,
        "seed_policy": f"seed={BASE_SEED}+question_index*{SAMPLES}+(sample_index-1)",
    }
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, STATE_PATH)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 0 is required")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed()
    previous_state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    started_at = previous_state.get("started_at", datetime.now(timezone.utc).isoformat())
    write_state(started_at, completed, "RUNNING")
    print(f"Resuming with {len(completed)}/8000 generations complete", flush=True)

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="test")
    prompt_template = (ROOT / "research/pilot_prompt_template_v2.txt").read_text()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    torch.cuda.reset_peak_memory_stats(0)

    tokenizer.padding_side = "left"
    with RAW_PATH.open("a", buffering=1) as output:
        for chunk_start in range(0, 500, 4):
            jobs = []
            for question_index in range(chunk_start, min(chunk_start + 4, 500)):
                row = dataset[question_index]
                prompt = prompt_template.format(problem=row["problem"])
                for sample_index in range(1, SAMPLES + 1):
                    if (question_index, sample_index) not in completed:
                        jobs.append((question_index, sample_index, row, prompt))
            if not jobs:
                continue
            if shutil.disk_usage("/workspace").free < MIN_FREE_BYTES:
                output.flush(); os.fsync(output.fileno())
                write_state(started_at, completed, "STOPPED_LOW_DISK", torch.cuda.max_memory_allocated(0) / 2**20)
                print("Clean stop: /workspace free disk is below 5 GiB", flush=True)
                return
            prompts = [job[3] for job in jobs]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
            input_width = int(encoded.input_ids.shape[1])
            input_lengths = encoded.attention_mask.sum(dim=1).tolist()
            seeds = [derived_seed(question_index, sample_index) for question_index, sample_index, _, _ in jobs]
            set_seed(seeds[0])
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            with torch.inference_mode():
                with independent_multinomial_streams(seeds):
                    generated = model.generate(
                        **encoded, **GENERATION, pad_token_id=tokenizer.eos_token_id
                    )
            torch.cuda.synchronize(0)
            batch_latency = time.perf_counter() - started
            for batch_row, ((question_index, sample_index, row, prompt), seed) in enumerate(zip(jobs, seeds)):
                continuation = generated[batch_row, input_width:]
                eos_positions = (continuation == tokenizer.eos_token_id).nonzero(as_tuple=False)
                output_tokens = int(eos_positions[0, 0]) + 1 if len(eos_positions) else int(continuation.shape[0])
                continuation = continuation[:output_tokens]
                raw_generation = tokenizer.decode(continuation, skip_special_tokens=True)
                extracted, source, extraction_status = extract_answer(raw_generation)
                normalized = normalize_answer(extracted)
                result = {
                    "question_index": question_index, "sample_index": sample_index, "seed": seed,
                    "question": row["problem"], "reference_answer": row["answer"], "prompt": prompt,
                    "raw_generation": raw_generation, "extraction_source": source,
                    "raw_extracted_answer": extracted, "normalized_answer": normalized,
                    "vote_key": vote_key(extracted) if extraction_status == "OK" else None,
                    "extraction_success": extraction_status == "OK", "status": extraction_status,
                    "evaluator_correct": math_equal(extracted, row["answer"]) if extraction_status == "OK" else False,
                    "input_tokens": int(input_lengths[batch_row]), "output_tokens": output_tokens,
                    "latency_seconds": round(batch_latency / len(jobs), 6),
                    "batch_wall_latency_seconds": round(batch_latency, 6), "batch_size": len(jobs),
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated(0) / 2**20, 2),
                    "temperature": GENERATION["temperature"], "top_p": GENERATION["top_p"],
                    "do_sample": GENERATION["do_sample"], "max_new_tokens": GENERATION["max_new_tokens"],
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed.add((question_index, sample_index))
            output.flush(); os.fsync(output.fileno())
            peak = torch.cuda.max_memory_allocated(0) / 2**20
            write_state(started_at, completed, "RUNNING", peak)
            print(f"checkpoint questions_through={min(chunk_start + 4, 500)}/500 generations={len(completed)}/8000", flush=True)

    write_state(started_at, completed, "GENERATIONS_COMPLETE", torch.cuda.max_memory_allocated(0) / 2**20)
    print("All 8000 generations complete", flush=True)


if __name__ == "__main__":
    main()
