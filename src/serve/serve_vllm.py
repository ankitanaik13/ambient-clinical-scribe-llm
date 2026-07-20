"""FastAPI serving app for the Ambient Clinical Scribe -- vLLM-backed path.

Same POST /generate-note request/response contract as src/serve/app.py (this module
imports those exact Pydantic models, so there is only one schema definition) -- but
generation runs through vLLM instead of a transformers pipeline, for real throughput/
latency numbers worth citing on a rented GPU.

Why this is a SEPARATE script rather than a `--vllm` flag on app.py:
  - Dependency footprint: vLLM pins its own compatible torch/CUDA build and pulls in a
    large, fast-moving dependency tree (its own CUDA kernels, etc.) that commonly conflicts
    with the transformers + bitsandbytes + peft stack the rest of this repo (train/eval)
    depends on. Keeping it out of requirements.txt means `pip install -r requirements.txt`
    never has to resolve that conflict for people who only want to train/evaluate.
  - Deployment reality: vLLM is CUDA-only and only makes sense on a real GPU box (the free
    Colab T4 used for training/eval in this project is undersized for it); app.py is meant
    to run anywhere the merged model can be loaded, including modest local hardware. A
    runtime flag would make app.py silently require vLLM to even import cleanly.
  - This mirrors the project's existing lazy-import convention (see baseline_eval.py,
    finetune.py) taken one step further: instead of an optional import inside one file,
    the optional *dependency* gets its own file, so `pip install vllm` is only ever needed
    when you actually run this script.

vLLM is NOT in requirements.txt -- install it separately (`pip install vllm`) on the GPU
box you run this on. This script targets vLLM's standard offline LLM/SamplingParams API
as of vLLM's actively-documented usage; vLLM's API has moved fast historically, so if a
newer vLLM version renamed something here, check vLLM's docs for the current equivalent.

Run as (on a GPU box with vllm installed):
    python -m src.serve.serve_vllm
    python -m src.serve.serve_vllm --model_dir models/merged --port 8001
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI

from src.eval.prompts import build_messages
from src.serve.app import GenerateNoteRequest, GenerateNoteResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_MODEL_DIR = PROJECT_ROOT / "models" / "merged"
MAX_NEW_TOKENS = 256

logger = logging.getLogger("ambient_clinical_scribe.vllm")


def load_vllm_engine(model_dir: Path):
    """Lazy import -- vLLM is intentionally not a project dependency; see module docstring."""
    from vllm import LLM

    return LLM(model=str(model_dir))


def create_app(model_dir: Path = DEFAULT_MERGED_MODEL_DIR) -> FastAPI:
    """Build the vLLM-backed FastAPI app with the same /generate-note contract as app.py."""
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        from transformers import AutoTokenizer

        state["engine"] = load_vllm_engine(model_dir)
        state["tokenizer"] = AutoTokenizer.from_pretrained(str(model_dir))
        yield
        state.clear()

    app = FastAPI(title="Ambient Clinical Scribe (vLLM)", lifespan=lifespan)

    @app.post("/generate-note", response_model=GenerateNoteResponse)
    def generate_note(request: GenerateNoteRequest) -> GenerateNoteResponse:
        from vllm import SamplingParams

        start = time.time()

        tokenizer = state["tokenizer"]
        messages = build_messages(request.dialogue)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Greedy decoding (temperature=0), matching the eval harness's do_sample=False.
        sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
        outputs = state["engine"].generate([prompt], sampling_params)
        note = outputs[0].outputs[0].text.strip()

        elapsed = time.time() - start
        dialogue_char_count = len(request.dialogue)
        dialogue_token_count = len(tokenizer(request.dialogue)["input_ids"])
        # Same privacy-by-design logging choice as src/serve/app.py -- request metadata
        # only, never dialogue content or the generated note. See that file's comment.
        logger.info(
            "generate-note (vllm) request timestamp=%s dialogue_chars=%d dialogue_tokens=%d "
            "elapsed_s=%.2f",
            datetime.now(timezone.utc).isoformat(),
            dialogue_char_count,
            dialogue_token_count,
            elapsed,
        )

        return GenerateNoteResponse(note=note)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the merged model via vLLM.")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MERGED_MODEL_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(model_dir=args.model_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
