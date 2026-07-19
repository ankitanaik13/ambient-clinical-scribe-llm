"""Load, clean, and split the MTS-Dialog dataset for the Ambient Clinical Scribe project.

Downloads the official CSVs from the MTS-Dialog GitHub repo (Ben Abacha et al., 2023,
CC BY 4.0), normalizes them to a common {id, dialogue, note, section_type} schema, runs
basic quality checks, and writes cleaned train/val/test splits to data/processed/.

The upstream repo ships two separate test-set CSVs (MEDIQA-Chat-2023 and MEDIQA-Sum-2023,
200 rows each); we concatenate them into a single 400-row test split to match the dataset's
commonly cited train/val/test sizes of 1201/100/400.

Run as:
    python -m src.data.load_data
    python -m src.data.load_data --force-download
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_RAW_BASE = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class SourceFile:
    """One MTS-Dialog CSV: which split it belongs to and a tag for globally-unique ids."""

    filename: str
    split: str  # "train", "val", or "test"
    tag: str  # short provenance tag, e.g. "test1" / "test2", used as an id prefix


# The two test CSVs (200 rows each) are combined into one 400-row "test" split.
SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile("MTS-Dialog-TrainingSet.csv", "train", "train"),
    SourceFile("MTS-Dialog-ValidationSet.csv", "val", "val"),
    SourceFile("MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv", "test", "test1"),
    SourceFile("MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv", "test", "test2"),
)

# Sanity-check only (logs a warning, doesn't fail the run) in case upstream data changes.
EXPECTED_SPLIT_SIZES = {"train": 1201, "val": 100, "test": 400}


def download_csv(filename: str, dest_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download one MTS-Dialog CSV into dest_dir if it isn't already cached there.

    Args:
        filename: Name of the CSV file within the MTS-Dialog Main-Dataset folder.
        dest_dir: Local directory to save the file into.
        force: Re-download even if the file already exists locally.

    Returns:
        Path to the local CSV file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if dest_path.exists() and not force:
        logger.info("Using cached %s", dest_path)
        return dest_path

    url = f"{REPO_RAW_BASE}/{filename}"
    logger.info("Downloading %s -> %s", url, dest_path)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    return dest_path


def load_source(
    source: SourceFile, raw_dir: Path = RAW_DIR, force_download: bool = False
) -> pd.DataFrame:
    """Download (if needed) and load one MTS-Dialog CSV, normalized to the common schema.

    Args:
        source: Which CSV to load, and which split/provenance tag it belongs to.
        raw_dir: Directory holding (or to hold) the raw CSV.
        force_download: Re-download the CSV even if cached locally.

    Returns:
        DataFrame with columns [id, dialogue, note, section_type, split].
    """
    path = download_csv(source.filename, raw_dir, force=force_download)
    df = pd.read_csv(path)

    missing = {"ID", "section_header", "section_text", "dialogue"} - set(df.columns)
    if missing:
        raise ValueError(f"{source.filename} is missing expected columns: {missing}")

    normalized = pd.DataFrame(
        {
            "id": [f"{source.tag}-{i}" for i in df["ID"]],
            "dialogue": df["dialogue"],
            "note": df["section_text"],
            "section_type": df["section_header"],
        }
    )
    normalized["split"] = source.split
    return normalized


def run_quality_checks(df: pd.DataFrame, split: str) -> dict:
    """Check for nulls/empty strings/duplicates and summarize dialogue & note length.

    This function only inspects df; `clean_split` is what actually drops rows.

    Args:
        df: Normalized split DataFrame with columns [id, dialogue, note, section_type].
        split: Name of the split, for logging.

    Returns:
        A JSON-serializable dict summarizing the checks.
    """
    n_null_dialogue = int(df["dialogue"].isna().sum())
    n_null_note = int(df["note"].isna().sum())
    n_empty_dialogue = int((df["dialogue"].astype(str).str.strip() == "").sum())
    n_empty_note = int((df["note"].astype(str).str.strip() == "").sum())
    n_dupes = int(df.duplicated(subset=["dialogue", "note"]).sum())

    dialogue_lengths = df["dialogue"].astype(str).str.split().str.len()
    note_lengths = df["note"].astype(str).str.split().str.len()

    summary = {
        "split": split,
        "n_rows": len(df),
        "n_null_dialogue": n_null_dialogue,
        "n_null_note": n_null_note,
        "n_empty_dialogue": n_empty_dialogue,
        "n_empty_note": n_empty_note,
        "n_duplicate_dialogue_note_pairs": n_dupes,
        "dialogue_word_count": {
            "mean": round(float(dialogue_lengths.mean()), 1),
            "median": float(dialogue_lengths.median()),
            "min": int(dialogue_lengths.min()),
            "max": int(dialogue_lengths.max()),
        },
        "note_word_count": {
            "mean": round(float(note_lengths.mean()), 1),
            "median": float(note_lengths.median()),
            "min": int(note_lengths.min()),
            "max": int(note_lengths.max()),
        },
    }

    logger.info(
        "[%s] rows=%d nulls(dialogue=%d, note=%d) empty(dialogue=%d, note=%d) dupes=%d "
        "dialogue_words(mean=%.1f, median=%.0f) note_words(mean=%.1f, median=%.0f)",
        split,
        summary["n_rows"],
        n_null_dialogue,
        n_null_note,
        n_empty_dialogue,
        n_empty_note,
        n_dupes,
        summary["dialogue_word_count"]["mean"],
        summary["dialogue_word_count"]["median"],
        summary["note_word_count"]["mean"],
        summary["note_word_count"]["median"],
    )
    return summary


def clean_split(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null/empty dialogue or note, and drop duplicate (dialogue, note) pairs."""
    cleaned = df.copy()
    cleaned["dialogue"] = cleaned["dialogue"].astype(str).str.strip()
    cleaned["note"] = cleaned["note"].astype(str).str.strip()
    cleaned = cleaned[
        (cleaned["dialogue"] != "")
        & (cleaned["note"] != "")
        & (cleaned["dialogue"].str.lower() != "nan")
        & (cleaned["note"].str.lower() != "nan")
    ]
    cleaned = cleaned.drop_duplicates(subset=["dialogue", "note"]).reset_index(drop=True)
    return cleaned


def build_splits(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download, normalize, quality-check, clean, and save the train/val/test splits.

    Args:
        raw_dir: Directory to cache raw CSVs in (gitignored).
        processed_dir: Directory to write cleaned split CSVs and the quality report to.
        force_download: Re-download raw CSVs even if already cached.

    Returns:
        Dict mapping split name -> cleaned DataFrame (also written to processed_dir as CSVs).
    """
    by_split: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for source in SOURCE_FILES:
        by_split[source.split].append(load_source(source, raw_dir, force_download))

    processed_dir.mkdir(parents=True, exist_ok=True)
    quality_report: dict[str, dict] = {}
    cleaned_splits: dict[str, pd.DataFrame] = {}

    for split, frames in by_split.items():
        raw_split_df = pd.concat(frames, ignore_index=True)
        quality_report[split] = run_quality_checks(raw_split_df, split)

        expected = EXPECTED_SPLIT_SIZES.get(split)
        if expected is not None and len(raw_split_df) != expected:
            logger.warning(
                "%s: expected %d rows from MTS-Dialog, found %d "
                "(upstream data may have changed)",
                split,
                expected,
                len(raw_split_df),
            )

        cleaned = clean_split(raw_split_df)
        n_dropped = len(raw_split_df) - len(cleaned)
        if n_dropped:
            logger.info(
                "%s: dropped %d row(s) during cleaning (nulls/empty/dupes)", split, n_dropped
            )

        out_path = processed_dir / f"{split}.csv"
        cleaned.drop(columns=["split"]).to_csv(out_path, index=False)
        logger.info("Saved %d cleaned rows -> %s", len(cleaned), out_path)
        cleaned_splits[split] = cleaned

    report_path = processed_dir / "data_quality_report.json"
    report_path.write_text(json.dumps(quality_report, indent=2))
    logger.info("Saved quality report -> %s", report_path)

    return cleaned_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and clean the MTS-Dialog dataset.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw CSVs even if already cached in data/raw/.",
    )
    args = parser.parse_args()
    build_splits(force_download=args.force_download)


if __name__ == "__main__":
    main()
