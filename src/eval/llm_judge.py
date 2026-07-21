"""LLM-as-judge scoring of generated clinical notes, using Google's Gemini API.

Scores each (dialogue, generated_note) pair from a baseline_eval.py / finetuned_eval.py
results JSON on four dimensions -- completeness, factual correctness, hallucination
presence, and structural adherence -- as structured JSON. Grading is anchored to the
dialogue transcript (the actual source of truth for what happened in the encounter), not
the reference note, since the reference is a single human-written example of an acceptable
note, not ground truth for hallucination detection.

Uses gemini-2.5-flash via the `google-genai` SDK (the current official SDK -- NOT the
deprecated `google-generativeai` package). This project originally used the Anthropic API
here; it was switched to Gemini specifically because Google AI Studio's free tier requires
no credit card, which matters for a portfolio project with no billing account. Within
Google's free-tier models, gemini-2.5-flash was picked for having the highest free daily
quota available, since this project judges up to 800 examples total (baseline + fine-tuned,
400 each) -- quota headroom matters more than raw model capability for a grading task like
this one.

Structured output is enforced via `response_mime_type="application/json"` +
`response_schema=JudgeScore` (a Pydantic model) on GenerateContentConfig -- Gemini's
equivalent of the structured-output approach the Anthropic version used
(`output_config.format`/json_schema). The rubric and JSON shape are otherwise unchanged
from that version.

Run this once against outputs/metrics/baseline_results.json and once against
outputs/metrics/finetuned_results.json -- src/eval/generate_report.py then compares the two.

Reads GEMINI_API_KEY from .env -- never hardcode it.

Run as:
    python -m src.eval.llm_judge --results_path outputs/metrics/baseline_results.json
    python -m src.eval.llm_judge --results_path outputs/metrics/finetuned_results.json
    python -m src.eval.llm_judge --results_path outputs/metrics/baseline_results.json --sample_size 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "baseline_results.json"

# gemini-2.5-flash: highest free-tier daily quota among Google AI Studio's models, chosen
# specifically for grading up to 800 examples per full baseline+finetuned comparison run.
JUDGE_MODEL = "gemini-2.5-flash"

JUDGE_SYSTEM_PROMPT = """You are an expert clinical documentation auditor. You will be shown a \
doctor-patient dialogue transcript and a clinical note section an AI system generated from it, \
along with a human-written reference note for context.

Grade the generated note against the DIALOGUE TRANSCRIPT, which is the actual source of truth \
for what happened in the encounter. The reference note is a single human-written example of an \
acceptable note for this dialogue, provided for context on expected content and style -- it is \
not ground truth, and the generated note should not be penalized for phrasing things differently \
than the reference as long as it is faithful to the dialogue.

Score on these dimensions:
- completeness (integer 1-5): does the note capture the clinically relevant information that is \
actually present in the dialogue for this note section? 1 = misses most relevant information, \
5 = captures everything clinically relevant and present in the dialogue.
- factual_correctness (integer 1-5): are the statements in the note consistent with what was \
actually said in the dialogue, with no contradictions? 1 = largely inconsistent with the \
dialogue, 5 = fully consistent.
- hallucination_present (boolean): true if the note states ANY clinical fact -- a symptom, \
medication, history item, finding, or measurement -- that is not stated or clearly implied in \
the dialogue, even if it sounds clinically plausible. This is about invented content, not phrasing.
- hallucinated_claims (array of strings): a short quote of each hallucinated claim from the \
note. Empty array if hallucination_present is false.
- structural_adherence (integer 1-5): does the note read as well-organized, standard clinical \
documentation prose appropriate for this note section? This is a single note section, not a \
complete multi-section SOAP note, so do not penalize it for lacking Subjective/Objective/\
Assessment/Plan headers -- judge whether it is structured the way a clinician would write this \
specific section (e.g. clear, appropriately terse, conventional clinical phrasing).
- rationale (string): one or two sentences justifying your scores.

Respond only with the requested JSON."""

USER_PROMPT_TEMPLATE = """Dialogue transcript:
{dialogue}

Reference note (human-written, for context only -- not ground truth):
{reference_note}

Generated note to evaluate:
{generated_note}"""


class JudgeScore(BaseModel):
    """Structured judge output -- passed to Gemini as `response_schema`."""

    completeness: int
    factual_correctness: int
    hallucination_present: bool
    hallucinated_claims: list[str]
    structural_adherence: int
    rationale: str


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect a 429 (rate limit / quota exceeded) error, defensively.

    Checks the google-genai SDK's documented `.code` attribute on APIError first, and
    falls back to substring-matching the exception's message. The fallback exists because
    this project could not verify the exact exception shape against a live 429 response
    (no GEMINI_API_KEY was available during development) -- string-matching is a hedge
    against that, not the primary signal.
    """
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in ("429", "resource_exhausted", "rate limit", "quota"))


def call_with_retry(fn, max_retries: int = 5, base_delay: float = 5.0, max_delay: float = 60.0):
    """Call fn(), retrying with exponential backoff on rate-limit errors only.

    The free tier's requests-per-minute cap means a burst of ~400 judge calls will likely
    hit a 429 at some point -- this retries those with backoff instead of aborting the run.
    Non-rate-limit errors are re-raised immediately (judge_one handles them per-example).

    Args:
        fn: Zero-arg callable making the API call.
        max_retries: Max retry attempts after the first try.
        base_delay: Initial backoff delay in seconds.
        max_delay: Cap on backoff delay.

    Returns:
        fn()'s return value.
    """
    from google.genai import errors

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except errors.APIError as exc:
            if not _is_rate_limit_error(exc) or attempt == max_retries:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            print(
                f"  [rate limited] retrying in {delay:.0f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)


def judge_one(
    client,
    dialogue: str,
    reference_note: str,
    generated_note: str,
    model: str = JUDGE_MODEL,
) -> dict | None:
    """Score a single generated note with Gemini, as structured JSON.

    Args:
        client: A google.genai.Client.
        dialogue: The source dialogue transcript.
        reference_note: The human-written reference note (context only).
        generated_note: The note to evaluate.
        model: Judge model id.

    Returns:
        The parsed judge score dict, or None if the call failed, was rate-limited past
        max_retries, was blocked, or returned unparseable output -- logged, not raised, so
        one bad example doesn't abort a run of hundreds.
    """
    from google.genai import types

    def _call():
        return client.models.generate_content(
            model=model,
            contents=USER_PROMPT_TEMPLATE.format(
                dialogue=dialogue,
                reference_note=reference_note,
                generated_note=generated_note,
            ),
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=JudgeScore,
                # Bulk grading over hundreds of examples doesn't need Gemini 2.5 Flash's
                # dynamic thinking -- disabling it keeps latency/cost down, mirroring this
                # project's earlier low-effort choice when the judge was Claude Sonnet.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    try:
        response = call_with_retry(_call)
    except Exception as exc:
        print(f"  [judge_one] API error, skipping example: {exc}")
        return None

    if not response.candidates:
        print("  [judge_one] no candidates in response (possibly blocked), skipping")
        return None

    finish_reason = getattr(response.candidates[0], "finish_reason", None)
    if finish_reason is not None and finish_reason != types.FinishReason.STOP:
        print(f"  [judge_one] non-STOP finish_reason={finish_reason}, skipping")
        return None

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed.model_dump() if isinstance(parsed, BaseModel) else dict(parsed)

    # Fall back to manual JSON parsing if `.parsed` wasn't populated -- a hedge against an
    # SDK version where auto-parsing behaves differently than verified here.
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        print("  [judge_one] response was not valid JSON, skipping")
        return None


def summarize_scores(scores: list[dict]) -> dict:
    """Aggregate per-example judge scores into summary statistics.

    Args:
        scores: List of successfully-parsed judge score dicts (see JudgeScore).

    Returns:
        Dict with mean completeness/factual_correctness/structural_adherence and the
        hallucination rate (fraction of examples with hallucination_present == True).
    """
    n = len(scores)
    return {
        "n_judged": n,
        "mean_completeness": round(sum(s["completeness"] for s in scores) / n, 3),
        "mean_factual_correctness": round(sum(s["factual_correctness"] for s in scores) / n, 3),
        "mean_structural_adherence": round(sum(s["structural_adherence"] for s in scores) / n, 3),
        "hallucination_rate": round(
            sum(1 for s in scores if s["hallucination_present"]) / n, 3
        ),
    }


def default_output_path(results_path: Path) -> Path:
    """Derive the judge-scores output path from a results path.

    e.g. outputs/metrics/baseline_results.json -> outputs/metrics/baseline_judge_scores.json
    """
    stem = results_path.stem.replace("_results", "")
    return results_path.parent / f"{stem}_judge_scores.json"


def run_judge(
    results_path: Path,
    output_path: Path | None = None,
    model: str = JUDGE_MODEL,
    sample_size: int | None = None,
    sleep_between_calls: float = 4.0,
) -> dict:
    """Judge every example in a baseline_eval.py / finetuned_eval.py results file.

    Args:
        results_path: Path to a *_results.json file (has an "examples" list with
            dialogue/reference_note/generated_note per example).
        output_path: Where to write judge scores. Defaults to a sibling
            "<source>_judge_scores.json" file (see default_output_path).
        model: Judge model id.
        sample_size: If set, judge only a random sample of this many examples.
        sleep_between_calls: Seconds to sleep between API calls. Defaults to a
            conservative 4s to proactively stay under the free tier's per-minute request
            cap -- call_with_retry is the real safety net if this still isn't enough, but
            spacing requests out means fewer 429s to retry in the first place.

    Returns:
        The results dict that was also written to output_path.
    """
    import random

    from google import genai

    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    results = json.loads(results_path.read_text())
    examples = results["examples"]
    if sample_size is not None and sample_size < len(examples):
        examples = random.Random(42).sample(examples, sample_size)

    per_example_scores: list[dict] = []
    for i, example in enumerate(examples):
        score = judge_one(
            client,
            dialogue=example["dialogue"],
            reference_note=example["reference_note"],
            generated_note=example["generated_note"],
            model=model,
        )
        if score is not None:
            per_example_scores.append({"id": example["id"], **score})
        print(f"  judged {i + 1}/{len(examples)} (kept {len(per_example_scores)})")
        if sleep_between_calls and i < len(examples) - 1:
            time.sleep(sleep_between_calls)

    if not per_example_scores:
        raise RuntimeError("No examples were successfully judged -- check API key / errors above.")

    output = {
        "source_results_path": str(results_path),
        "phase": results.get("phase", "unknown"),
        "judge_model": model,
        "n_examples_attempted": len(examples),
        "aggregate": summarize_scores(per_example_scores),
        "examples": per_example_scores,
    }

    output_path = output_path or default_output_path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved judge scores -> {output_path}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score generated clinical notes with an LLM judge (Gemini API)."
    )
    parser.add_argument("--results_path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Judge only a random sample of this many examples (for a quick/cheap test run).",
    )
    parser.add_argument(
        "--sleep_between_calls",
        type=float,
        default=4.0,
        help="Seconds to sleep between calls, to stay under the free tier's rate limit.",
    )
    args = parser.parse_args()

    run_judge(
        results_path=args.results_path,
        output_path=args.output_path,
        model=args.model,
        sample_size=args.sample_size,
        sleep_between_calls=args.sleep_between_calls,
    )


if __name__ == "__main__":
    main()
