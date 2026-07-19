"""QLoRA fine-tune of Meta-Llama-3-8B-Instruct on MTS-Dialog for SOAP-note generation.

Loads the base model 4-bit quantized (same BitsAndBytesConfig as src/eval/baseline_eval.py,
via src/model_utils.py) and trains a LoRA adapter with trl's SFTTrainer. Training examples
are built with src.eval.prompts.build_messages() UNCHANGED, with the reference note appended
as the assistant turn -- this is what makes the Phase 4 baseline-vs-fine-tuned comparison
apples-to-apples: the fine-tuned model is trained on exactly the prompt structure the
zero-shot baseline was scored against.

Hyperparameters are sized for a free-tier Colab T4 (16GB VRAM); see the comments on
TrainConfig and LORA_CONFIG_KWARGS below for the reasoning behind each choice, and what to
change on a bigger GPU.

This targets trl>=0.9,<0.12's SFTTrainer API (tokenizer=/dataset_text_field=/max_seq_length=/
packing= passed directly to SFTTrainer). trl>=0.12 moved those into a separate SFTConfig
class -- see the pin + comment in requirements.txt.

Requires a CUDA GPU: bitsandbytes 4-bit quantization and the paged 8-bit optimizer used here
are CUDA-only, so this cannot actually run on this Mac (Apple Silicon/MPS or CPU). All heavy
imports are lazy (inside functions) so the CLI, config, and data-formatting logic can still
be validated locally without CUDA or any of torch/transformers/peft/trl/bitsandbytes
installed.

Run as (on a CUDA box):
    python -m src.train.finetune
    python -m src.train.finetune --no_wandb
    python -m src.train.finetune --resume_from_checkpoint models/checkpoints/checkpoint-100
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.eval.prompts import build_messages
from src.model_utils import BASE_MODEL_NAME, get_bnb_config, load_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEFAULT_VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.csv"
DEFAULT_ADAPTER_OUTPUT_DIR = PROJECT_ROOT / "models" / "adapter"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
WANDB_PROJECT = "ambient-clinical-scribe"

# Llama 3's chat template renders an assistant turn's header as exactly this string. We give
# it to DataCollatorForCompletionOnlyLM so the loss is masked to only the note tokens (the
# assistant's response), not the system/user prompt tokens (the dialogue) -- the model is
# trained to *write notes*, not to reproduce transcripts it was already given.
ASSISTANT_RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"


# =========================================================================================
# CONFIG -- hyperparameters sized for a free-tier Colab T4 (16GB VRAM). Each field explains
# its reasoning and what to change if you move to a bigger GPU (e.g. a rented A10/A100/L4).
# =========================================================================================
@dataclass
class TrainConfig:
    num_train_epochs: float = 3.0
    # MTS-Dialog's train split is only ~1200 examples; 3 epochs is enough for the adapter to
    # pick up the note style/structure without badly overfitting such a small set. Bigger
    # GPU: unchanged -- epoch count is a data-size decision, not a hardware one. Watch the
    # W&B val-loss curve and stop earlier via resume/checkpoint if it starts climbing.

    learning_rate: float = 2e-4
    # A LoRA-typical learning rate, 10-100x higher than a full fine-tune would use. This is
    # safe specifically because only the small LoRA matrices are being updated (the 4-bit
    # base model is frozen) -- they can tolerate a much higher LR without the instability a
    # full-parameter update at this LR would cause. Bigger GPU: unchanged, this is driven by
    # LoRA rank/scale, not compute.

    per_device_train_batch_size: int = 1
    # A T4 has 16GB VRAM; the 4-bit base model alone takes ~5-6GB, and Llama 3 8B's
    # activations for even modest sequence lengths eat most of the rest. batch_size=1 is the
    # safe ceiling to avoid OOM. Bigger GPU (24GB+): try 4-8 directly.

    gradient_accumulation_steps: int = 16
    # Compensates for the tiny per-device batch size: effective batch size = 1 * 16 = 16,
    # a reasonable size for stable gradients on ~1200 examples. Bigger GPU: lower this
    # roughly in proportion to how much you raise per_device_train_batch_size, to keep the
    # *effective* batch size in the same ballpark (e.g. batch_size=8, accum=2 -> still 16).

    max_seq_length: int = 1024
    # MTS-Dialog dialogues average ~105 words (roughly 150-200 tokens) and notes ~40 words;
    # 1024 tokens covers prompt + note for the large majority of examples with headroom,
    # without wasting T4 memory padding every batch to a much longer max. Bigger GPU: raise
    # to 2048 to stop truncating the small number of longer outlier dialogues.

    gradient_checkpointing: bool = True
    # Recomputes activations during the backward pass instead of caching them -- trades
    # compute for memory, and is close to mandatory to fit an 8B model's training state on a
    # T4 at all. Bigger GPU (40GB+): can disable for faster (but more memory-hungry) steps.

    optim: str = "paged_adamw_8bit"
    # 8-bit paged AdamW (bitsandbytes): keeps optimizer states in 8-bit and pages them out to
    # CPU RAM on memory spikes. This is what keeps gradient-checkpointing's
    # activation-recomputation boundaries from OOM-ing on a T4. Bigger GPU: plain
    # "adamw_torch" is simpler and a bit faster if VRAM isn't the binding constraint.

    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    # Checkpoint every 50 steps (a few minutes on a T4), not just at the end -- a Colab
    # disconnect mid-run then loses at most a few minutes of progress, not the whole job.
    save_total_limit: int = 3
    # Keep only the 3 most recent checkpoints. Free-tier Colab disk is small, and even
    # LoRA-adapter-only checkpoints (plus optimizer state) add up fast otherwise.

    seed: int = 42


TRAIN_CONFIG = TrainConfig()


# =========================================================================================
# LoRA config -- sized to fit a T4 while targeting the layers that matter most here.
# =========================================================================================
LORA_CONFIG_KWARGS = dict(
    r=16,
    # Rank of the low-rank update matrices. r=16 is a common middle ground: enough capacity
    # for the adapter to genuinely learn the SOAP note style/structure, while keeping
    # trainable params a small fraction of the 8B base (roughly ~0.1-0.2% of total params
    # with these target modules) so it trains fast and fits T4 memory. r=8 would be leaner
    # but risks underfitting a structural/formatting task; r=32-64 buys more capacity at the
    # cost of memory and higher overfitting risk on only ~1200 training examples.
    lora_alpha=32,
    # LoRA scales its update by (alpha / r). alpha=32 with r=16 gives a scaling factor of 2 --
    # the "alpha = 2*r" heuristic popularized by the original LoRA paper, which keeps the
    # effective update magnitude roughly stable if r is later changed, instead of needing to
    # retune the learning rate every time.
    lora_dropout=0.05,
    # Light dropout on the LoRA path as a regularizer. With only ~1200 training examples, an
    # adapter can start memorizing rather than generalizing without some regularization.
    bias="none",
    # Don't train bias terms -- matches the original LoRA/QLoRA papers' setup. Bias-only
    # gains are marginal for this task, and skipping them keeps the trainable parameter
    # count (and memory/compute) minimal.
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # Attention projections only -- not the MLP layers (gate_proj/up_proj/down_proj). This
    # task is primarily a structural/stylistic transformation (dialogue -> SOAP note), which
    # attention adapters capture well per the original LoRA paper's own ablations; skipping
    # the much larger MLP layers roughly halves trainable params and memory versus
    # targeting "all-linear", which matters on a 16GB card. If note quality plateaus below
    # what you need, adding the MLP layers is the natural next lever on a bigger GPU.
)


def build_training_example(example: dict, tokenizer) -> dict:
    """Format one (dialogue, note) row into a single chat-templated training string.

    Uses src.eval.prompts.build_messages() UNCHANGED, then appends the reference note as
    the assistant turn -- so the model trains on exactly the prompt structure Phase 2
    scored the zero-shot baseline against.

    Args:
        example: Dict with "dialogue" and "note" keys (one row of the processed dataset).
        tokenizer: Tokenizer whose chat template renders the Llama 3 Instruct format.

    Returns:
        {"text": <full chat-templated string, prompt + reference note>}
    """
    messages = build_messages(example["dialogue"]) + [
        {"role": "assistant", "content": example["note"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def load_dataset_split(csv_path: Path, tokenizer):
    """Load a processed MTS-Dialog CSV and format it into chat-templated training text.

    Args:
        csv_path: Path to a cleaned split CSV from src/data/load_data.py (needs at least
            "dialogue" and "note" columns).
        tokenizer: Tokenizer used to render each example via its chat template.

    Returns:
        A datasets.Dataset with a "text" column ready for SFTTrainer.
    """
    from datasets import Dataset

    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df[["dialogue", "note"]], preserve_index=False)
    return dataset.map(lambda example: build_training_example(example, tokenizer))


def run_finetune(
    model_name: str = BASE_MODEL_NAME,
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VAL_PATH,
    adapter_output_dir: Path = DEFAULT_ADAPTER_OUTPUT_DIR,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    config: TrainConfig = TRAIN_CONFIG,
    use_wandb: bool = True,
    wandb_run_name: str = "qlora-finetune-r16",
    resume_from_checkpoint: str | None = None,
) -> Path:
    """Run the QLoRA SFT fine-tune and save the trained LoRA adapter.

    Requires a CUDA GPU -- bitsandbytes 4-bit loading and the paged 8-bit optimizer are
    CUDA-only. Intended for a Colab T4 or rented GPU, not local execution on this Mac.

    Args:
        model_name: HF Hub id of the base instruct model.
        train_path: Cleaned train split CSV.
        val_path: Cleaned val split CSV, used for in-loop eval-loss logging.
        adapter_output_dir: Where to save the final trained LoRA adapter.
        checkpoint_dir: Where to save periodic training checkpoints (for resuming).
        config: Hyperparameters; defaults to TRAIN_CONFIG above.
        use_wandb: Whether to log loss curves to Weights & Biases.
        wandb_run_name: W&B run name (distinct from baseline_eval.py's run names).
        resume_from_checkpoint: Path to a checkpoint dir to resume from, if any.

    Returns:
        Path the final adapter was saved to.
    """
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None
    wandb_api_key = os.environ.get("WANDB_API_KEY") or None

    if use_wandb:
        import wandb

        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        # HF Trainer's WandbCallback reads the target project from this env var.
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    tokenizer = load_tokenizer(model_name, hf_token, padding_side="right")

    train_dataset = load_dataset_split(train_path, tokenizer)
    eval_dataset = load_dataset_split(val_path, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=hf_token,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )

    lora_config = LoraConfig(**LORA_CONFIG_KWARGS)

    collator = DataCollatorForCompletionOnlyLM(ASSISTANT_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        fp16=True,
        bf16=False,
        # Training precision (this flag) is intentionally fp16, not bf16, because a T4
        # (Turing, sm_75) has no bf16 tensor-core support. This is separate from the bf16
        # *compute dtype* used to dequantize the frozen 4-bit base weights (get_bnb_config(),
        # matched to baseline_eval.py) -- that dequant op runs fine on T4 regardless. Bigger
        # GPU (Ampere+, e.g. A10/A100/L4): flip to bf16=True, fp16=False here -- bf16 is
        # numerically more stable and those GPUs accelerate it natively.
        report_to=["wandb"] if use_wandb else [],
        run_name=wandb_run_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        packing=False,
    )
    trainer.model.print_trainable_parameters()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_output_dir))
    tokenizer.save_pretrained(str(adapter_output_dir))

    if use_wandb:
        import wandb

        if wandb.run is not None:
            wandb.finish()

    return adapter_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Meta-Llama-3-8B-Instruct on MTS-Dialog."
    )
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--adapter_output_dir", type=Path, default=DEFAULT_ADAPTER_OUTPUT_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--no_wandb", action="store_true", help="Skip Weights & Biases logging.")
    parser.add_argument("--wandb_run_name", default="qlora-finetune-r16")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint dir under --checkpoint_dir to resume from.",
    )
    # Hyperparameter overrides -- default to the sized-for-T4 TrainConfig above, but can be
    # tweaked from the CLI (e.g. on a bigger GPU) without editing the file.
    parser.add_argument("--num_train_epochs", type=float, default=TRAIN_CONFIG.num_train_epochs)
    parser.add_argument("--learning_rate", type=float, default=TRAIN_CONFIG.learning_rate)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=TRAIN_CONFIG.per_device_train_batch_size,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=TRAIN_CONFIG.gradient_accumulation_steps,
    )
    parser.add_argument("--max_seq_length", type=int, default=TRAIN_CONFIG.max_seq_length)
    args = parser.parse_args()

    config = TrainConfig(
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
    )

    adapter_path = run_finetune(
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        adapter_output_dir=args.adapter_output_dir,
        checkpoint_dir=args.checkpoint_dir,
        config=config,
        use_wandb=not args.no_wandb,
        wandb_run_name=args.wandb_run_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
