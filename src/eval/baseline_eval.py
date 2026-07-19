"""Zero-shot baseline evaluation of Meta-Llama-3-8B-Instruct on MTS-Dialog.

Loads the base instruct model in 4-bit (bitsandbytes NF4), generates a clinical note for
each test-set dialogue using the exact same prompt template that Phase 3 trains against
(src/eval/prompts.py), scores the outputs with ROUGE-1/2/L against the reference notes, and
saves both per-example and aggregate results to outputs/metrics/baseline_results.json.

This script requires a CUDA GPU (bitsandbytes 4-bit quantization is not available on CPU or
Apple Silicon/MPS). It is meant to be run on a rented/Colab GPU, e.g. a free-tier T4.

Run as:
    python -m src.eval.baseline_eval
    python -m src.eval.baseline_eval --sample_size 20      # quick smoke test
    python -m src.eval.baseline_eval --no_wandb            # skip W&B logging
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
from dotenv import load_dotenv
from rouge_score import rouge_scorer

from src.eval.prompts import build_messages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "baseline_results.json"
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
WANDB_PROJECT = "ambient-clinical-scribe"


def chunked(items: list, batch_size: int) -> Iterator[list]:
    """Yield successive batch_size-sized chunks of items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_model_and_tokenizer(model_name: str, hf_token: str | None):
    """Load Meta-Llama-3-8B-Instruct 4-bit quantized via bitsandbytes.

    Args:
        model_name: HF Hub model id.
        hf_token: HF access token (required -- Llama 3 is a gated model).

    Returns:
        (model, tokenizer) ready for generation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


def generate_notes(
    model,
    tokenizer,
    dialogues: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate a clinical note for each dialogue using greedy decoding, in batches.

    Args:
        model: Loaded causal LM.
        tokenizer: Matching tokenizer (left-padded, pad token set).
        dialogues: Raw dialogue transcripts.
        batch_size: Number of dialogues to generate for at once.
        max_new_tokens: Generation length cap per note.

    Returns:
        Generated note text per dialogue, in the same order as `dialogues`.
    """
    import torch

    predictions: list[str] = []
    for batch in chunked(dialogues, batch_size):
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(dialogue), tokenize=False, add_generation_prompt=True
            )
            for dialogue in batch
        ]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(text.strip() for text in texts)

    return predictions


def compute_rouge(predictions: list[str], references: list[str]) -> tuple[dict, list[dict]]:
    """Compute ROUGE-1/2/L F-measure per example and averaged over the whole set.

    Args:
        predictions: Generated notes.
        references: Ground-truth reference notes.

    Returns:
        (aggregate_scores, per_example_scores) where aggregate_scores has keys
        "rouge1"/"rouge2"/"rougeL" mapped to mean F-measure, and per_example_scores is a
        list of the same three keys per example.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_example: list[dict] = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        per_example.append({k: round(v.fmeasure, 4) for k, v in scores.items()})

    aggregate = {
        key: round(sum(ex[key] for ex in per_example) / len(per_example), 4)
        for key in ("rouge1", "rouge2", "rougeL")
    }
    return aggregate, per_example


def run_baseline_eval(
    model_name: str = DEFAULT_MODEL_NAME,
    test_path: Path = DEFAULT_TEST_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sample_size: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    seed: int = 42,
    use_wandb: bool = True,
    wandb_run_name: str = "baseline-zero-shot",
) -> dict:
    """Run zero-shot generation + ROUGE scoring on the test set and save results.

    Args:
        model_name: HF Hub model id for the base instruct model.
        test_path: Path to the cleaned test split (from src/data/load_data.py).
        output_path: Where to write the JSON results.
        sample_size: If set, evaluate on a random sample of this many rows instead of
            the full test set (for quick smoke testing).
        batch_size: Generation batch size.
        max_new_tokens: Max tokens to generate per note.
        seed: Random seed used for sampling.
        use_wandb: Whether to log a summary to Weights & Biases.
        wandb_run_name: W&B run name.

    Returns:
        The results dict that was also written to output_path.
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None
    wandb_api_key = os.environ.get("WANDB_API_KEY") or None

    df = pd.read_csv(test_path)
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)

    run = None
    if use_wandb:
        import wandb

        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        run = wandb.init(
            project=WANDB_PROJECT,
            name=wandb_run_name,
            config={
                "model_name": model_name,
                "n_examples": len(df),
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "quantization": "4-bit NF4 (bitsandbytes)",
                "phase": "baseline_zero_shot",
            },
        )

    model, tokenizer = load_model_and_tokenizer(model_name, hf_token)

    start = time.time()
    predictions = generate_notes(
        model, tokenizer, df["dialogue"].tolist(), batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - start

    aggregate, per_example = compute_rouge(predictions, df["note"].tolist())

    results = {
        "model_name": model_name,
        "phase": "baseline_zero_shot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_examples": len(df),
        "generation_seconds": round(elapsed, 1),
        "aggregate_rouge": aggregate,
        "examples": [
            {
                "id": row.id,
                "section_type": row.section_type,
                "dialogue": row.dialogue,
                "reference_note": row.note,
                "generated_note": pred,
                **scores,
            }
            for row, pred, scores in zip(df.itertuples(), predictions, per_example)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    if run is not None:
        import wandb

        wandb.log({f"rouge/{k}": v for k, v in aggregate.items()})
        wandb.log({"generation_seconds": elapsed})
        sample_table = wandb.Table(
            columns=["id", "section_type", "reference_note", "generated_note"],
            data=[
                [ex["id"], ex["section_type"], ex["reference_note"], ex["generated_note"]]
                for ex in results["examples"][:20]
            ],
        )
        wandb.log({"sample_generations": sample_table})
        run.finish()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot baseline evaluation of Meta-Llama-3-8B-Instruct on MTS-Dialog."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test_path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Evaluate on a random sample of this many test rows (for a quick test run).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true", help="Skip Weights & Biases logging.")
    parser.add_argument("--wandb_run_name", default="baseline-zero-shot")
    args = parser.parse_args()

    results = run_baseline_eval(
        model_name=args.model_name,
        test_path=args.test_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        wandb_run_name=args.wandb_run_name,
    )
    print(json.dumps(results["aggregate_rouge"], indent=2))


if __name__ == "__main__":
    main()
