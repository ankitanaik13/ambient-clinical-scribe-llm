"""Shared model-loading building blocks used by both eval and train scripts.

Centralizing the quantization config here guarantees the QLoRA fine-tune (Phase 3) loads
the base model under the exact same 4-bit quantization the zero-shot baseline (Phase 2) was
scored under, and that the fine-tuned eval (Phase 4) does too -- so any score difference
measured downstream reflects what the model learned, not a quantization mismatch.

All heavy imports (torch, transformers) are lazy, inside the functions, so this module can
be imported on a machine with no CUDA/no ML libs installed (e.g. for CLI/config validation).
"""

from __future__ import annotations

BASE_MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
# Ungated community mirror of meta-llama/Meta-Llama-3-8B-Instruct -- identical weights and
# architecture, just hosted without Meta's access-request gate. Switched to this while
# Meta's gated-access approval was pending; see README.md's "Base model note" (in the
# Setup section) for the full explanation. Swap back to the meta-llama/ repo id here if/
# when that approval clears -- everything else (tokenizer, chat template, HF_TOKEN usage)
# is unaffected either way.


def get_bnb_config():
    """4-bit NF4, double-quantized, fp16-compute BitsAndBytesConfig.

    Used identically to load the base model for zero-shot baseline generation (Phase 2)
    and for QLoRA fine-tuning (Phase 3), so the frozen base weights are read through the
    same quantization in both.

    Compute dtype is float16, not bfloat16: a T4 GPU (Turing, sm_75 -- the free-tier Colab
    GPU this project targets) has no bf16 tensor-core support, so bf16 compute silently
    falls back to a slower path there for no numerical benefit. float16 is what T4 actually
    accelerates. On an Ampere+ GPU (A10/A100/L4/24GB+) this can be switched to
    torch.bfloat16 for better numerical stability with no speed cost.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str, hf_token: str | None, padding_side: str = "right"):
    """Load a tokenizer with a pad token guaranteed to be set.

    Args:
        model_name: HF Hub model id.
        hf_token: HF access token. Not required for the ungated NousResearch mirror this
            project defaults to (see BASE_MODEL_NAME), but still worth setting -- HF rate-
            limits/throttles anonymous downloads more aggressively than authenticated ones,
            and an authenticated token is required if BASE_MODEL_NAME is ever pointed back
            at the gated meta-llama/ repo.
        padding_side: "left" for batched generation (Phase 2/4 eval, so new tokens are
            generated contiguously at the end of every sequence in a batch), "right" for
            SFT training (Phase 3, so padding doesn't get interleaved into the label
            sequence the loss is computed over).

    Returns:
        A tokenizer with `pad_token` set (falls back to `eos_token` if the base model
        doesn't define one, which is the case for Llama 3).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
