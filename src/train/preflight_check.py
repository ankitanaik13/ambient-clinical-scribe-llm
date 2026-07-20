"""Pre-flight check: confirm DataCollatorForCompletionOnlyLM actually finds the response
template in every formatted training example, before spending GPU time on a real run.

Why this exists: trl's DataCollatorForCompletionOnlyLM searches for the tokenized
response-template subsequence inside each example's tokenized input_ids. If it can't find
that subsequence for a given example, it does NOT raise -- it emits a warning and sets that
example's entire label sequence to -100 (ignore_index), which silently gives the example
zero loss for the whole training run. Whether the template is found depends on how the
tokenizer merges tokens around the template boundary in context, which can differ per
example (BPE merges are context-sensitive) -- so a decoded-string substring check is not a
faithful test of what the real collator will do. This script instead runs the REAL
DataCollatorForCompletionOnlyLM, with the REAL Llama 3 tokenizer, against every row of
data/processed/train.csv, using build_training_example() from src/train/finetune.py
unchanged, so the check exercises exactly the code path finetune.py itself uses.

Earlier validation of the prompt-formatting logic (during Phase 3) only spot-checked 2 rows
with a mock tokenizer that doesn't replicate real BPE merge behavior -- that was enough to
validate message ordering, not template-matching against every real training example. This
script covers the latter, and is meant to run before finetune.py:

    python -m src.train.preflight_check && python -m src.train.finetune

This cannot be validated on a CUDA-less machine: it needs trl and transformers installed,
plus an HF_TOKEN with access to the gated Llama 3 tokenizer -- none of which are available
on the Mac this repo was otherwise developed on. It does not strictly require a GPU
(tokenization is CPU-bound), but it needs the same environment finetune.py needs, so in
practice it runs as a Colab pre-training step alongside everything else GPU-dependent in
this project. All imports beyond argparse/pandas/dotenv are lazy, consistent with the rest
of this repo, so --help still works without those installed.

Run as (on the same box you'll run finetune.py on):
    python -m src.train.preflight_check
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.model_utils import BASE_MODEL_NAME, load_tokenizer
from src.train.finetune import ASSISTANT_RESPONSE_TEMPLATE, DEFAULT_TRAIN_PATH, build_training_example


def run_preflight_check(
    train_path: Path = DEFAULT_TRAIN_PATH,
    model_name: str = BASE_MODEL_NAME,
) -> dict:
    """Check every row of train_path against the real completion-only-loss collator.

    Args:
        train_path: Cleaned training split CSV (needs "dialogue" and "note" columns, plus
            "id" for reporting which rows fail).
        model_name: HF Hub id of the base model whose tokenizer/chat template to use --
            must match what finetune.py will actually load.

    Returns:
        {"n_total": int, "n_failed": int, "failed_ids": list[str]}. A row's id lands in
        failed_ids if the response template was not found in its formatted text, meaning
        DataCollatorForCompletionOnlyLM would mask that example's labels entirely and it
        would contribute zero loss during training.
    """
    from trl import DataCollatorForCompletionOnlyLM

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None

    tokenizer = load_tokenizer(model_name, hf_token, padding_side="right")
    collator = DataCollatorForCompletionOnlyLM(ASSISTANT_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    df = pd.read_csv(train_path)
    failed_ids: list[str] = []

    for row in df.itertuples():
        text = build_training_example({"dialogue": row.dialogue, "note": row.note}, tokenizer)["text"]
        input_ids = tokenizer(text)["input_ids"]
        batch = collator([{"input_ids": input_ids}])
        labels = batch["labels"][0].tolist()
        if all(label == -100 for label in labels):
            failed_ids.append(row.id)

    n_total = len(df)
    n_failed = len(failed_ids)

    print(f"Checked {n_total} training examples against the real completion-only collator.")
    if n_failed:
        print(
            f"FAILED: {n_failed}/{n_total} example(s) would get ZERO loss -- response "
            f"template not found in their tokenized form:"
        )
        for failed_id in failed_ids[:20]:
            print(f"  - {failed_id}")
        if n_failed > 20:
            print(f"  ... and {n_failed - 20} more")
    else:
        print(f"OK: response template found in all {n_total} examples.")

    return {"n_total": n_total, "n_failed": n_failed, "failed_ids": failed_ids}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-flight check: confirm the completion-only loss mask finds the response "
            "template in every training example before running finetune.py."
        )
    )
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    args = parser.parse_args()

    result = run_preflight_check(train_path=args.train_path, model_name=args.model_name)
    if result["n_failed"] > 0:
        raise SystemExit(
            f"Pre-flight check failed: {result['n_failed']} training example(s) would "
            f"silently get zero loss. Fix ASSISTANT_RESPONSE_TEMPLATE in "
            f"src/train/finetune.py or the affected data before running finetune.py."
        )


if __name__ == "__main__":
    main()
