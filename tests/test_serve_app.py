"""Tests for src/serve/app.py -- run entirely without CUDA, using a mocked model pipeline.

Same mocking approach as the smoke tests used to validate baseline_eval.py / llm_judge.py /
generate_report.py in earlier phases: a fake pipeline_factory stands in for the real
transformers pipeline, so these tests exercise the actual FastAPI request/response path
(routing, validation, the prompt-building call, the response schema) without ever importing
torch or transformers.

Run as:
    python -m pytest tests/test_serve_app.py -v
"""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from src.serve.app import create_app

SECRET_DIALOGUE_MARKER = "UNIQUE_SECRET_DIALOGUE_MARKER_12345"
SECRET_NOTE_MARKER = "UNIQUE_SECRET_GENERATED_NOTE_MARKER_67890"


class FakeTokenizer:
    """Stands in for a real HF tokenizer: chat template + a token count via whitespace split."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "FAKE_PROMPT"

    def __call__(self, text: str):
        return {"input_ids": text.split()}


class FakePipeline:
    """Stands in for a real transformers text-generation pipeline."""

    tokenizer = FakeTokenizer()

    def __call__(self, prompt, max_new_tokens=None, do_sample=None, return_full_text=None):
        return [{"generated_text": f"{SECRET_NOTE_MARKER} SOAP note text."}]


def fake_pipeline_factory(model_dir):
    return FakePipeline()


def test_generate_note_returns_expected_shape():
    app = create_app(pipeline_factory=fake_pipeline_factory)
    with TestClient(app) as client:
        response = client.post(
            "/generate-note", json={"dialogue": "Doctor: hello. Patient: hi, doc."}
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"note"}
    assert isinstance(body["note"], str)
    assert SECRET_NOTE_MARKER in body["note"]


def test_generate_note_rejects_empty_dialogue():
    app = create_app(pipeline_factory=fake_pipeline_factory)
    with TestClient(app) as client:
        response = client.post("/generate-note", json={"dialogue": ""})

    assert response.status_code == 422


def test_generate_note_rejects_missing_dialogue_field():
    app = create_app(pipeline_factory=fake_pipeline_factory)
    with TestClient(app) as client:
        response = client.post("/generate-note", json={})

    assert response.status_code == 422


def test_logging_never_contains_dialogue_or_note_content(caplog):
    """The privacy-by-design guarantee: only request metadata is logged, never content."""
    app = create_app(pipeline_factory=fake_pipeline_factory)
    dialogue_text = f"{SECRET_DIALOGUE_MARKER} patient reports chest pain and shortness of breath"

    with caplog.at_level(logging.INFO, logger="ambient_clinical_scribe"):
        with TestClient(app) as client:
            response = client.post("/generate-note", json={"dialogue": dialogue_text})

    assert response.status_code == 200

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert SECRET_DIALOGUE_MARKER not in log_text
    assert SECRET_NOTE_MARKER not in log_text
    # The metadata we DO want logged should be present.
    assert "dialogue_chars=" in log_text
    assert "dialogue_tokens=" in log_text
    assert "timestamp=" in log_text
