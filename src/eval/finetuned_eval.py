"""Evaluation of the fine-tuned (merged) model on MTS-Dialog -- apples-to-apples with Phase 2.

This is deliberately a thin wrapper around src/eval/baseline_eval.py rather than a parallel
reimplementation: it imports the exact same generate_notes(), compute_rouge(), and chunked()
functions, and the exact same prompt template (src/eval/prompts.py, via generate_notes()). The
ONLY thing that differs from the baseline run is which model gets loaded -- the merged
fine-tuned model (src/train/merge_adapter.py's output) instead of the base instruct model, and
it is loaded from a local directory rather than the HF Hub. Everything downstream (prompting,
decoding, scoring) is byte-for-byte the same code path, so any ROUGE difference between
baseline_results.json and finetuned_results.json reflects what the model learned, not a harness
difference.

Loads the merged model 4-bit quantized (same BitsAndBytesConfig as the baseline, via
src/model_utils.py) so the two runs are also apples-to-apples on quantization, not just prompt.

This script requires a CUDA GPU, same as baseline_eval.py, and cannot run on this Mac.

Run as (on a CUDA box, after src/train/merge_adapter.py has produced models/merged/):
    python -m src.eval.finetuned_eval
    python -m src.eval.finetuned_eval --sample_size 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.eval.baseline_eval import (
    DEFAULT_TEST_PATH,
    WANDB_PROJECT,
    compute_rouge,
    generate_notes,
)
from src.model_utils import get_bnb_config, load_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_MODEL_DIR = PROJECT_ROOT / "models" / "merged"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "finetuned_results.json"


def load_finetuned_model_and_tokenizer(model_dir: Path, hf_token: str | None):
    """Load the merged fine-tuned model 4-bit quantized, from a local directory.

    Uses the same BitsAndBytesConfig as baseline_eval.load_model_and_tokenizer, via
    src/model_utils.py, so the baseline and fine-tuned runs are comparable on quantization.

    Args:
        model_dir: Local directory produced by src/train/merge_adapter.py.
        hf_token: HF access token. Not required for a local directory, but accepted for
            interface symmetry with baseline_eval's loader.

    Returns:
        (model, tokenizer) ready for generation.
    """
    from transformers import AutoModelForCausalLM

    model_dir_str = str(model_dir)
    tokenizer = load_tokenizer(model_dir_str, hf_token, padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir_str,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


def run_finetuned_eval(
    merged_model_dir: Path = DEFAULT_MERGED_MODEL_DIR,
    test_path: Path = DEFAULT_TEST_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sample_size: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    seed: int = 42,
    use_wandb: bool = True,
    wandb_run_name: str = "finetuned-eval",
) -> dict:
    """Run generation + ROUGE scoring on the test set using the merged fine-tuned model.

    Same signature and same output schema as baseline_eval.run_baseline_eval, so
    src/eval/generate_report.py can treat the two result files identically.

    Args:
        merged_model_dir: Local directory with the merged model (see merge_adapter.py).
        test_path: Path to the cleaned test split.
        output_path: Where to write the JSON results.
        sample_size: If set, evaluate on a random sample of this many rows.
        batch_size: Generation batch size.
        max_new_tokens: Max tokens to generate per note.
        seed: Random seed used for sampling.
        use_wandb: Whether to log a summary to Weights & Biases.
        wandb_run_name: W&B run name (distinct from baseline_eval.py's run names).

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
                "model_dir": str(merged_model_dir),
                "n_examples": len(df),
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "quantization": "4-bit NF4 (bitsandbytes)",
                "phase": "finetuned_eval",
            },
        )

    model, tokenizer = load_finetuned_model_and_tokenizer(merged_model_dir, hf_token)

    start = time.time()
    predictions = generate_notes(
        model, tokenizer, df["dialogue"].tolist(), batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - start

    aggregate, per_example = compute_rouge(predictions, df["note"].tolist())

    results = {
        "model_name": str(merged_model_dir),
        "phase": "finetuned_eval",
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
        description="Evaluate the merged fine-tuned model on MTS-Dialog (apples-to-apples with baseline_eval.py)."
    )
    parser.add_argument("--merged_model_dir", type=Path, default=DEFAULT_MERGED_MODEL_DIR)
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
    parser.add_argument("--wandb_run_name", default="finetuned-eval")
    args = parser.parse_args()

    results = run_finetuned_eval(
        merged_model_dir=args.merged_model_dir,
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
