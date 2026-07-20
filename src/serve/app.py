"""FastAPI serving app for the Ambient Clinical Scribe -- default (CPU/GPU-agnostic) path.

Exposes a single POST /generate-note endpoint: give it a dialogue transcript, get back a
generated clinical note. Backed by the merged fine-tuned model (src/train/merge_adapter.py's
output) via a standard transformers `pipeline("text-generation", ...)`, using
src.eval.prompts.build_messages() UNCHANGED -- the same prompt structure used to train
(Phase 3) and evaluate (Phase 2/4) the model, so inference-time behavior matches what was
actually measured.

Model loading is deferred to app startup (FastAPI lifespan), not module import time, and is
injectable via `create_app(pipeline_factory=...)`. This means importing this module -- e.g.
to build the OpenAPI schema, or in a test with a mocked pipeline -- never touches
torch/transformers, matching the lazy-import pattern used throughout this repo (see
baseline_eval.py, finetune.py) so the app can be inspected/tested without CUDA installed.

Run as (on a box with the merged model at models/merged/):
    python -m src.serve.app
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

For an alternative vLLM-backed serving path (same endpoint contract, for real GPU
throughput numbers), see src/serve/serve_vllm.py -- that's a separate script, not a flag
here; see that file's docstring for why.
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.eval.prompts import build_messages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_MODEL_DIR = PROJECT_ROOT / "models" / "merged"
MAX_NEW_TOKENS = 256

logger = logging.getLogger("ambient_clinical_scribe")


class GenerateNoteRequest(BaseModel):
    dialogue: str = Field(..., min_length=1, description="Doctor-patient dialogue transcript.")


class GenerateNoteResponse(BaseModel):
    note: str = Field(..., description="Generated clinical note text.")


def load_pipeline(model_dir: Path):
    """Load the merged model as a transformers text-generation pipeline.

    Lazy import (only touches torch/transformers when actually called, i.e. at app startup
    or in a real serving run) -- never at module import time.
    """
    from transformers import pipeline

    return pipeline("text-generation", model=str(model_dir), device_map="auto")


def create_app(
    pipeline_factory: Callable[[Path], object] = load_pipeline,
    model_dir: Path = DEFAULT_MERGED_MODEL_DIR,
) -> FastAPI:
    """Build the FastAPI app, with the model pipeline injectable for testing.

    Args:
        pipeline_factory: Callable that loads and returns a text-generation pipeline given
            a model directory. Defaults to load_pipeline (transformers). Tests pass a fake
            factory that returns a mock pipeline, so no torch/transformers import ever
            happens under test.
        model_dir: Directory containing the merged model (see src/train/merge_adapter.py).

    Returns:
        A configured FastAPI app. The model is loaded once, at startup (lifespan), not per
        request and not at import time.
    """
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["pipeline"] = pipeline_factory(model_dir)
        yield
        state.clear()

    app = FastAPI(title="Ambient Clinical Scribe", lifespan=lifespan)

    @app.post("/generate-note", response_model=GenerateNoteResponse)
    def generate_note(request: GenerateNoteRequest) -> GenerateNoteResponse:
        start = time.time()

        text_pipeline = state["pipeline"]
        tokenizer = text_pipeline.tokenizer
        messages = build_messages(request.dialogue)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = text_pipeline(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_full_text=False,
        )
        note = outputs[0]["generated_text"].strip()

        elapsed = time.time() - start
        dialogue_char_count = len(request.dialogue)
        dialogue_token_count = len(tokenizer(request.dialogue)["input_ids"])
        # Deliberate privacy-by-design choice: log only request metadata (timestamp,
        # dialogue length, latency) -- NEVER the dialogue text or the generated note. This
        # matters for anything that touches clinical text, even synthetic data like
        # MTS-Dialog -- building the "don't log PHI-shaped content" discipline in from the
        # start means it's already there the day this code (or a fork of it) touches real
        # patient data, instead of being a retrofit under time pressure.
        logger.info(
            "generate-note request timestamp=%s dialogue_chars=%d dialogue_tokens=%d elapsed_s=%.2f",
            datetime.now(timezone.utc).isoformat(),
            dialogue_char_count,
            dialogue_token_count,
            elapsed,
        )

        return GenerateNoteResponse(note=note)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the merged model via a FastAPI app.")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MERGED_MODEL_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(model_dir=args.model_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
