"""LLM-as-judge scoring of generated clinical notes, using the Anthropic API.

Scores each (dialogue, generated_note) pair from a baseline_eval.py / finetuned_eval.py
results JSON on four dimensions -- completeness, factual correctness, hallucination
presence, and structural adherence -- as structured JSON, via Claude's `output_config.format`
(json_schema) so every response is guaranteed to parse. Grading is anchored to the dialogue
transcript (the actual source of truth for what happened in the encounter), not the reference
note, since the reference is a single human-written example of an acceptable note, not ground
truth for hallucination detection.

Run this once against outputs/metrics/baseline_results.json and once against
outputs/metrics/finetuned_results.json -- src/eval/generate_report.py then compares the two.

Reads ANTHROPIC_API_KEY from .env -- never hardcode it.

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "baseline_results.json"

# claude-sonnet-4-6 as specified for this project. It's a currently active model
# (see Anthropic's model catalog) -- not a placeholder or typo.
JUDGE_MODEL = "claude-sonnet-4-6"

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

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "completeness": {"type": "integer"},
        "factual_correctness": {"type": "integer"},
        "hallucination_present": {"type": "boolean"},
        "hallucinated_claims": {"type": "array", "items": {"type": "string"}},
        "structural_adherence": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": [
        "completeness",
        "factual_correctness",
        "hallucination_present",
        "hallucinated_claims",
        "structural_adherence",
        "rationale",
    ],
    "additionalProperties": False,
}


def judge_one(
    client,
    dialogue: str,
    reference_note: str,
    generated_note: str,
    model: str = JUDGE_MODEL,
    effort: str = "low",
) -> dict | None:
    """Score a single generated note with Claude, as structured JSON.

    Args:
        client: An anthropic.Anthropic client.
        dialogue: The source dialogue transcript.
        reference_note: The human-written reference note (context only).
        generated_note: The note to evaluate.
        model: Judge model id.
        effort: output_config effort level. Defaults to "low" -- this is a bulk grading task
            run over hundreds of examples, not a task that benefits much from deep reasoning,
            so we trade away Sonnet 4.6's "high" default for lower cost/latency at this scale.

    Returns:
        The parsed judge score dict, or None if the call failed or was refused (logged, not
        raised, so one bad example doesn't abort a run of hundreds).
    """
    import anthropic

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=JUDGE_SYSTEM_PROMPT,
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        dialogue=dialogue,
                        reference_note=reference_note,
                        generated_note=generated_note,
                    ),
                }
            ],
        )
    except anthropic.APIError as exc:
        print(f"  [judge_one] API error, skipping example: {exc}")
        return None

    if response.stop_reason == "refusal":
        print("  [judge_one] judge refused to score this example, skipping")
        return None

    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        print("  [judge_one] no text block in judge response, skipping")
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("  [judge_one] judge response was not valid JSON, skipping")
        return None


def summarize_scores(scores: list[dict]) -> dict:
    """Aggregate per-example judge scores into summary statistics.

    Args:
        scores: List of successfully-parsed judge score dicts (see JUDGE_SCHEMA).

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
    effort: str = "low",
    sample_size: int | None = None,
    sleep_between_calls: float = 0.0,
) -> dict:
    """Judge every example in a baseline_eval.py / finetuned_eval.py results file.

    Args:
        results_path: Path to a *_results.json file (has an "examples" list with
            dialogue/reference_note/generated_note per example).
        output_path: Where to write judge scores. Defaults to a sibling
            "<source>_judge_scores.json" file (see default_output_path).
        model: Judge model id.
        effort: output_config effort level passed to judge_one.
        sample_size: If set, judge only a random sample of this many examples.
        sleep_between_calls: Seconds to sleep between API calls, to stay under rate limits
            on large runs.

    Returns:
        The results dict that was also written to output_path.
    """
    import random

    import anthropic

    load_dotenv()
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
            effort=effort,
        )
        if score is not None:
            per_example_scores.append({"id": example["id"], **score})
        print(f"  judged {i + 1}/{len(examples)} (kept {len(per_example_scores)})")
        if sleep_between_calls:
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
        description="Score generated clinical notes with an LLM judge (Anthropic API)."
    )
    parser.add_argument("--results_path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Judge only a random sample of this many examples (for a quick/cheap test run).",
    )
    parser.add_argument("--sleep_between_calls", type=float, default=0.0)
    args = parser.parse_args()

    run_judge(
        results_path=args.results_path,
        output_path=args.output_path,
        model=args.model,
        effort=args.effort,
        sample_size=args.sample_size,
        sleep_between_calls=args.sleep_between_calls,
    )


if __name__ == "__main__":
    main()
