"""Merge a trained LoRA adapter into the base model's weights.

Loads the base model in full precision (NOT 4-bit -- PEFT's merge_and_unload() needs real
weights to add the LoRA delta into; merging into still-quantized weights is unsupported/
lossy), loads the adapter on top, merges, and saves a single standalone set of weights to
models/merged/. Phase 4 evaluation and Phase 5 serving then load this directly with
AutoModelForCausalLM, with no PEFT/adapter-loading logic needed at inference time.

Memory note: an 8B model in fp16/bf16 is ~16GB, which typically does NOT fit on a free
Colab T4 (16GB total, and the training run before it will have left activations/optimizer
state resident too). Run this as a separate step, either:
  - on CPU (--device_map cpu; needs ~32GB+ system RAM, will be slow but works), or
  - on a GPU with more headroom than a T4 (e.g. a rented A10/A100/L4, 24GB+).

Requires torch/transformers/peft; all imports are lazy so --help and argument parsing work
without them installed (this machine has neither CUDA nor these packages installed).

Run as:
    python -m src.train.merge_adapter
    python -m src.train.merge_adapter --device_map cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model_utils import BASE_MODEL_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / "models" / "adapter"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "merged"


def merge_adapter(
    base_model_name: str,
    adapter_path: Path,
    output_dir: Path,
    device_map: str = "auto",
) -> Path:
    """Merge a LoRA adapter into the base model and save the standalone merged weights.

    Args:
        base_model_name: HF Hub id of the base instruct model the adapter was trained on.
        adapter_path: Directory containing the trained adapter (from finetune.py's
            adapter_output_dir), including its saved tokenizer.
        output_dir: Directory to save the merged model + tokenizer to.
        device_map: "auto" to load onto GPU(s), "cpu" for a CPU-only merge (needs ~32GB+
            system RAM but no GPU memory).

    Returns:
        output_dir, containing the merged model's weights and tokenizer.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(output_dir))

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a trained LoRA adapter into the base model's weights."
    )
    parser.add_argument("--base_model_name", default=BASE_MODEL_NAME)
    parser.add_argument("--adapter_path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device_map",
        default="auto",
        help='"auto" to load onto GPU(s), "cpu" for a CPU-only merge (needs ~32GB+ RAM).',
    )
    args = parser.parse_args()

    output_dir = merge_adapter(
        args.base_model_name, args.adapter_path, args.output_dir, args.device_map
    )
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
