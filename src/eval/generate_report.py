"""Build the baseline-vs-fine-tuned comparison report: a markdown table + a bar chart.

Combines four JSON files produced by earlier phases:
  - outputs/metrics/baseline_results.json    (src/eval/baseline_eval.py)
  - outputs/metrics/finetuned_results.json   (src/eval/finetuned_eval.py)
  - outputs/metrics/baseline_judge_scores.json   (src/eval/llm_judge.py)
  - outputs/metrics/finetuned_judge_scores.json  (src/eval/llm_judge.py)

into outputs/metrics/comparison_report.md and outputs/figures/comparison_chart.png.

This script does not fabricate numbers: if any input file is missing, it raises a clear
error telling you which eval/judge script to run first, rather than guessing.

Explicitly flags the case that matters most for an honest report: ROUGE improving while the
LLM-judge hallucination rate goes UP. Better n-gram overlap with the reference does not imply
fewer invented clinical details -- if that happens, this script says so, prominently, instead
of only reporting the flattering numbers.

Run as:
    python -m src.eval.generate_report
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

DEFAULT_BASELINE_RESULTS = METRICS_DIR / "baseline_results.json"
DEFAULT_FINETUNED_RESULTS = METRICS_DIR / "finetuned_results.json"
DEFAULT_BASELINE_JUDGE = METRICS_DIR / "baseline_judge_scores.json"
DEFAULT_FINETUNED_JUDGE = METRICS_DIR / "finetuned_judge_scores.json"
DEFAULT_REPORT_PATH = METRICS_DIR / "comparison_report.md"
DEFAULT_CHART_PATH = FIGURES_DIR / "comparison_chart.png"

# Validated categorical pair (see dataviz skill palette + scripts/validate_palette.js):
# passes lightness/chroma/CVD-separation/normal-vision/contrast checks as an adjacent pair.
BASELINE_COLOR = "#2a78d6"  # blue
FINETUNED_COLOR = "#eb6834"  # orange


def load_json(path: Path, description: str) -> dict:
    """Load a JSON results file, or fail with a clear pointer to how to produce it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at {path}. Run the corresponding eval/judge script "
            f"first -- this report does not fabricate results."
        )
    return json.loads(path.read_text())


def build_metric_rows(
    baseline_rouge: dict,
    finetuned_rouge: dict,
    baseline_judge: dict,
    finetuned_judge: dict,
) -> list[dict]:
    """Build the comparison rows shown in the report table.

    Each row: name, baseline value, finetuned value, delta, whether higher is better, and a
    format spec, so the markdown renderer and the chart builder share one source of truth.
    """
    return [
        {
            "name": "ROUGE-1",
            "baseline": baseline_rouge["rouge1"],
            "finetuned": finetuned_rouge["rouge1"],
            "higher_is_better": True,
            "fmt": "{:.3f}",
        },
        {
            "name": "ROUGE-2",
            "baseline": baseline_rouge["rouge2"],
            "finetuned": finetuned_rouge["rouge2"],
            "higher_is_better": True,
            "fmt": "{:.3f}",
        },
        {
            "name": "ROUGE-L",
            "baseline": baseline_rouge["rougeL"],
            "finetuned": finetuned_rouge["rougeL"],
            "higher_is_better": True,
            "fmt": "{:.3f}",
        },
        {
            "name": "Completeness (1-5, LLM judge)",
            "baseline": baseline_judge["mean_completeness"],
            "finetuned": finetuned_judge["mean_completeness"],
            "higher_is_better": True,
            "fmt": "{:.2f}",
        },
        {
            "name": "Factual correctness (1-5, LLM judge)",
            "baseline": baseline_judge["mean_factual_correctness"],
            "finetuned": finetuned_judge["mean_factual_correctness"],
            "higher_is_better": True,
            "fmt": "{:.2f}",
        },
        {
            "name": "Structural adherence (1-5, LLM judge)",
            "baseline": baseline_judge["mean_structural_adherence"],
            "finetuned": finetuned_judge["mean_structural_adherence"],
            "higher_is_better": True,
            "fmt": "{:.2f}",
        },
        {
            "name": "Hallucination rate (LLM judge)",
            "baseline": baseline_judge["hallucination_rate"],
            "finetuned": finetuned_judge["hallucination_rate"],
            "higher_is_better": False,
            "fmt": "{:.1%}",
        },
    ]


def render_markdown_table(rows: list[dict]) -> str:
    """Render the metric rows as a markdown table, with an explicit (not color-only) arrow
    on each delta so direction reads correctly in plain-text markdown viewers."""
    lines = ["| Metric | Baseline | Fine-tuned | Δ |", "|---|---|---|---|"]
    for row in rows:
        fmt = row["fmt"]
        baseline_str = fmt.format(row["baseline"])
        finetuned_str = fmt.format(row["finetuned"])
        delta = row["finetuned"] - row["baseline"]
        improved = delta > 0 if row["higher_is_better"] else delta < 0
        arrow = "▲ better" if improved else ("▼ worse" if delta != 0 else "— no change")
        delta_str = fmt.format(abs(delta))
        lines.append(
            f"| {row['name']} | {baseline_str} | {finetuned_str} | "
            f"{'+' if delta > 0 else '-' if delta < 0 else ''}{delta_str} ({arrow}) |"
        )
    return "\n".join(lines)


def check_hallucination_regression(
    baseline_rouge: dict, finetuned_rouge: dict, baseline_judge: dict, finetuned_judge: dict
) -> tuple[bool, str]:
    """Check the case this report is required to surface honestly: ROUGE up, hallucinations up.

    Returns:
        (flagged, message). flagged is True only when ROUGE improved on at least one of
        rouge1/rouge2/rougeL AND the judge's hallucination rate also increased. The message
        is written either way -- it reports the hallucination-rate comparison honestly even
        when nothing is wrong.
    """
    rouge_improved = any(
        finetuned_rouge[k] > baseline_rouge[k] for k in ("rouge1", "rouge2", "rougeL")
    )
    baseline_rate = baseline_judge["hallucination_rate"]
    finetuned_rate = finetuned_judge["hallucination_rate"]
    hallucination_increased = finetuned_rate > baseline_rate

    if rouge_improved and hallucination_increased:
        return True, (
            f"Fine-tuning improved ROUGE but the LLM-judge hallucination rate went UP "
            f"({baseline_rate:.1%} -> {finetuned_rate:.1%}). Better n-gram overlap with the "
            f"reference notes did not translate into fewer invented clinical details here. "
            f"This is reported as-is, not hidden behind the ROUGE improvement -- treat the "
            f"fine-tuned model's factual reliability as a real regression to investigate "
            f"(e.g. more epochs overfitting to note *style* rather than note *content*, or a "
            f"training set too small to teach faithfulness), not as a shipped improvement."
        )
    if hallucination_increased:
        return False, (
            f"Hallucination rate increased ({baseline_rate:.1%} -> {finetuned_rate:.1%}), "
            f"though ROUGE did not improve alongside it."
        )
    if finetuned_rate < baseline_rate:
        return False, (
            f"Hallucination rate decreased ({baseline_rate:.1%} -> {finetuned_rate:.1%}) -- "
            f"fine-tuning did not trade factual reliability for style/overlap gains here."
        )
    return False, f"Hallucination rate unchanged ({baseline_rate:.1%})."


def make_chart(
    baseline_rouge: dict,
    finetuned_rouge: dict,
    baseline_judge: dict,
    finetuned_judge: dict,
    output_path: Path,
) -> Path:
    """Render the baseline-vs-fine-tuned comparison as a 3-panel grouped bar chart.

    Three panels (not one shared axis) because the metrics live on different scales --
    ROUGE is 0-1, LLM-judge quality scores are 1-5, and hallucination rate is 0-1 but means
    the opposite direction of "better." Mixing scales on one axis is the #1 chart mistake;
    this keeps every panel single-axis and lets the reader compare baseline vs fine-tuned
    within each panel instead of across incompatible units.
    """
    import matplotlib.pyplot as plt

    rouge_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    rouge_baseline = [baseline_rouge["rouge1"], baseline_rouge["rouge2"], baseline_rouge["rougeL"]]
    rouge_finetuned = [finetuned_rouge["rouge1"], finetuned_rouge["rouge2"], finetuned_rouge["rougeL"]]

    judge_labels = ["Completeness", "Factual\ncorrectness", "Structural\nadherence"]
    judge_baseline = [
        baseline_judge["mean_completeness"],
        baseline_judge["mean_factual_correctness"],
        baseline_judge["mean_structural_adherence"],
    ]
    judge_finetuned = [
        finetuned_judge["mean_completeness"],
        finetuned_judge["mean_factual_correctness"],
        finetuned_judge["mean_structural_adherence"],
    ]

    hallucination_labels = ["Hallucination rate"]
    hallucination_baseline = [baseline_judge["hallucination_rate"]]
    hallucination_finetuned = [finetuned_judge["hallucination_rate"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), width_ratios=[3, 3, 1.4])
    fig.suptitle("Baseline (zero-shot) vs. Fine-tuned (QLoRA)", fontsize=13, fontweight="bold")

    panel_specs = [
        (axes[0], rouge_labels, rouge_baseline, rouge_finetuned, "Score (0-1)", 1.0),
        (axes[1], judge_labels, judge_baseline, judge_finetuned, "Mean score (1-5)", 5.0),
        (
            axes[2],
            hallucination_labels,
            hallucination_baseline,
            hallucination_finetuned,
            "Rate (lower is better)",
            1.0,
        ),
    ]

    for ax, labels, baseline_vals, finetuned_vals, ylabel, ylim_top in panel_specs:
        x = range(len(labels))
        width = 0.35
        bars_baseline = ax.bar(
            [i - width / 2 for i in x], baseline_vals, width, label="Baseline",
            color=BASELINE_COLOR,
        )
        bars_finetuned = ax.bar(
            [i + width / 2 for i in x], finetuned_vals, width, label="Fine-tuned",
            color=FINETUNED_COLOR,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, ylim_top * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.bar_label(bars_baseline, fmt="%.2f", fontsize=7, padding=2)
        ax.bar_label(bars_finetuned, fmt="%.2f", fontsize=7, padding=2)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_report(
    baseline_results_path: Path = DEFAULT_BASELINE_RESULTS,
    finetuned_results_path: Path = DEFAULT_FINETUNED_RESULTS,
    baseline_judge_path: Path = DEFAULT_BASELINE_JUDGE,
    finetuned_judge_path: Path = DEFAULT_FINETUNED_JUDGE,
    report_path: Path = DEFAULT_REPORT_PATH,
    chart_path: Path = DEFAULT_CHART_PATH,
) -> str:
    """Load all four result files, build the chart, and write comparison_report.md.

    Returns:
        The full markdown report text (also written to report_path).
    """
    baseline_results = load_json(baseline_results_path, "Baseline results")
    finetuned_results = load_json(finetuned_results_path, "Fine-tuned results")
    baseline_judge = load_json(baseline_judge_path, "Baseline judge scores")
    finetuned_judge = load_json(finetuned_judge_path, "Fine-tuned judge scores")

    baseline_rouge = baseline_results["aggregate_rouge"]
    finetuned_rouge = finetuned_results["aggregate_rouge"]
    baseline_judge_agg = baseline_judge["aggregate"]
    finetuned_judge_agg = finetuned_judge["aggregate"]

    rows = build_metric_rows(baseline_rouge, finetuned_rouge, baseline_judge_agg, finetuned_judge_agg)
    table_md = render_markdown_table(rows)

    flagged, hallucination_message = check_hallucination_regression(
        baseline_rouge, finetuned_rouge, baseline_judge_agg, finetuned_judge_agg
    )

    chart_output = make_chart(
        baseline_rouge, finetuned_rouge, baseline_judge_agg, finetuned_judge_agg, chart_path
    )
    # Relative to the report's own directory, so the markdown image link resolves correctly
    # regardless of where report_path/chart_path happen to live relative to the project root.
    chart_rel_path = Path(os.path.relpath(chart_output, start=report_path.parent))

    warning_section = ""
    if flagged:
        warning_section = f"\n> ⚠️ **{hallucination_message}**\n"

    report = f"""# Baseline vs. Fine-Tuned Comparison Report

Generated: {datetime.now(timezone.utc).isoformat()}

- Baseline: `{baseline_results.get('model_name', 'unknown')}` ({baseline_results.get('n_examples', '?')} examples, {baseline_judge.get('n_examples_attempted', '?')} judged)
- Fine-tuned: `{finetuned_results.get('model_name', 'unknown')}` ({finetuned_results.get('n_examples', '?')} examples, {finetuned_judge.get('n_examples_attempted', '?')} judged)
- LLM judge: `{finetuned_judge.get('judge_model', 'unknown')}`
{warning_section}
## Metric comparison

{table_md}

## Hallucination check

{hallucination_message}

## Chart

![Baseline vs fine-tuned comparison]({chart_rel_path.as_posix()})

## Notes

- ROUGE is computed against the human-written reference note for each dialogue.
- LLM-judge scores are computed against the source dialogue transcript, not the reference
  note (see src/eval/llm_judge.py) -- the reference is context, not ground truth, so the
  judge does not penalize a note for being faithful to the dialogue but phrased differently.
- This report reflects whatever result files were present when it was generated. Re-run
  `python -m src.eval.generate_report` after any new eval or judge run to refresh it.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"Saved report -> {report_path}")
    print(f"Saved chart -> {chart_output}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the baseline-vs-fine-tuned comparison report and chart."
    )
    parser.add_argument("--baseline_results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--finetuned_results", type=Path, default=DEFAULT_FINETUNED_RESULTS)
    parser.add_argument("--baseline_judge", type=Path, default=DEFAULT_BASELINE_JUDGE)
    parser.add_argument("--finetuned_judge", type=Path, default=DEFAULT_FINETUNED_JUDGE)
    parser.add_argument("--report_output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--chart_output", type=Path, default=DEFAULT_CHART_PATH)
    args = parser.parse_args()

    generate_report(
        baseline_results_path=args.baseline_results,
        finetuned_results_path=args.finetuned_results,
        baseline_judge_path=args.baseline_judge,
        finetuned_judge_path=args.finetuned_judge,
        report_path=args.report_output,
        chart_path=args.chart_output,
    )


if __name__ == "__main__":
    main()
