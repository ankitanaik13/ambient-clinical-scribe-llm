# Ambient Clinical Scribe

**Fine-tuning an open LLM for ambient clinical documentation** — turning doctor-patient
dialogue into structured clinical note text, and measuring, honestly, whether QLoRA
fine-tuning actually makes that better.

This project fine-tunes **Llama 3 8B Instruct** with **QLoRA** to generate clinical note
sections from conversation transcripts, benchmarks it against a zero-shot baseline on both
automatic metrics (ROUGE) and an LLM-judge rubric (completeness, factual correctness,
hallucination presence, structural adherence), and ships a minimal FastAPI serving layer.

> **This is a portfolio / research project, not a clinically validated tool.** See
> [Disclaimer](#disclaimer) before reading anything else here as a claim about real-world
> clinical accuracy.

## Current status

- [x] Phase 1 — Repo scaffold + data pipeline
- [x] Phase 2 — Baseline (zero-shot) evaluation harness
- [x] Phase 3 — QLoRA fine-tuning + adapter merging
- [x] Phase 4 — Fine-tuned evaluation + LLM-judge comparison
- [x] Phase 5 — Serving + this README
- [ ] An actual GPU training/eval run (see [Results](#results) — pending)

All code is written and locally validated (config/CLI parsing, prompt formatting, ROUGE
math, mocked-model test suites) on a CUDA-less machine. The QLoRA training run itself,
the fine-tuned evaluation, and the LLM-judge comparison still need to execute on a real
GPU (a free-tier Colab T4, per the training config, or a rented box) — see
[Setup](#setup--running-the-pipeline).

## Motivation

Clinical documentation burden is one of the most cited drivers of physician burnout, and
"ambient" scribes — tools that listen to (or read a transcript of) a visit and draft the
note — are one of the more concrete, well-scoped applications of LLMs in healthcare
workflows. This project is a small, honest end-to-end version of that pipeline: not a
production ambient-scribe product, but the full ML lifecycle a real one would need —
data, baseline, fine-tune, evaluate, compare, serve — built on public, synthetic data so
the whole thing is inspectable and reproducible.

## Architecture

```
                    MTS-Dialog (Ben Abacha et al., 2023, CC BY 4.0)
                                     |
                                     v
                     +-----------------------------------+
                     |   src/data/load_data.py            |
                     |   download, normalize, clean       |
                     +-----------------------------------+
                                     |
              +----------------------+-----------------------+
              v                      v                        v
   data/processed/train.csv  data/processed/val.csv  data/processed/test.csv
       (1,200 examples)         (100 examples)          (400 examples)
              |                                                |
              |                                                v
              |                          +--------------------------------------+
              |                          |  src/eval/baseline_eval.py            |
              |                          |  Meta-Llama-3-8B-Instruct, zero-shot  |
              |                          |  4-bit NF4 (bitsandbytes)             |
              |                          +--------------------------------------+
              |                                                |
              v                                                v
   +--------------------------------------+     outputs/metrics/baseline_results.json
   |  src/train/finetune.py                |                   |
   |  QLoRA SFT (peft + trl SFTTrainer)     |                   |
   |  LoRA r=16 / alpha=32, q/k/v/o_proj    |                   |
   |  sized for a free-tier T4 (16GB)       |                   |
   +--------------------------------------+                   |
              |                                                |
              v                                                |
       models/adapter/                                         |
              |                                                |
              v                                                |
   +--------------------------------------+                   |
   |  src/train/merge_adapter.py           |                   |
   +--------------------------------------+                   |
              |                                                |
              v                                                |
       models/merged/  -------------------------+              |
              |                                 |              |
              v                                 v              |
   +--------------------------------+  +----------------------------------+
   |  src/eval/finetuned_eval.py     |  |  src/serve/app.py (FastAPI)       |
   |  SAME harness as baseline_eval  |  |  src/serve/serve_vllm.py (vLLM)   |
   |  (prompt, decoding, ROUGE code) |  |  POST /generate-note              |
   +--------------------------------+  +----------------------------------+
              |
              v
   outputs/metrics/finetuned_results.json
              |
   +----------+----------+
   |                     |
   v                     v
baseline_results.json  finetuned_results.json
   |                     |
   +----------+----------+
              v
   +--------------------------------------+
   |  src/eval/llm_judge.py                |
   |  gemini-flash-latest as judge:         |
   |  completeness, factual correctness,    |
   |  hallucination, structural adherence   |
   +--------------------------------------+
              |
              v
   outputs/metrics/{baseline,finetuned}_judge_scores.json
              |
              v
   +--------------------------------------+
   |  src/eval/generate_report.py          |
   +--------------------------------------+
              |
              v
   outputs/metrics/comparison_report.md
   outputs/figures/comparison_chart.png
```

## Dataset

[**MTS-Dialog**](https://github.com/abachaa/MTS-Dialog) — a public dataset of short
doctor-patient conversations paired with the clinical note section a clinician wrote from
each one.

> Asma Ben Abacha, Wen-wai Yim, Yadan Fan, and Thomas Lin. **"An Empirical Study of
> Clinical Note Generation from Doctor-Patient Encounters."** *Proceedings of the 17th
> Conference of the European Chapter of the Association for Computational Linguistics
> (EACL 2023).*

Licensed **CC BY 4.0**. `src/data/load_data.py` downloads the official
`train` (1,201) / `validation` (100) / `test` (two 200-example files, combined to 400)
CSVs directly from the MTS-Dialog GitHub repo, normalizes them to a common
`{id, dialogue, note, section_type}` schema, and writes cleaned splits to
`data/processed/`.

**Important scope note:** each MTS-Dialog example is a single note *section* (e.g.
`GENHX`, `ROS`, `CC`, `ASSESSMENT`) generated from a dialogue snippet, not a complete
multi-section SOAP note per encounter. Everywhere this project says "SOAP-style," read it
as "clinical-note-section style" — the model is trained and judged on producing one
well-formed section at a time, not assembling a full Subjective/Objective/Assessment/Plan
note.

## Tech stack

| Layer | Tools |
|---|---|
| Data | pandas, requests |
| Baseline + fine-tuned inference | Hugging Face `transformers`, `bitsandbytes` (4-bit NF4 QLoRA quantization), `accelerate` |
| Fine-tuning | `peft` (LoRA), `trl` (`SFTTrainer`) |
| Experiment tracking | Weights & Biases |
| Automatic metrics | `rouge-score` |
| LLM-judge + report | Gemini API via `google-genai` (`gemini-flash-latest`), `matplotlib` |
| Serving | FastAPI, `uvicorn`, `vllm` (optional, GPU-only path) |
| Testing | `pytest`, FastAPI `TestClient` |

## Setup & running the pipeline

> **Base model note:** this project defaults to `NousResearch/Meta-Llama-3-8B-Instruct`
> (see `BASE_MODEL_NAME` in `src/model_utils.py`) rather than Meta's own
> `meta-llama/Meta-Llama-3-8B-Instruct` repo. NousResearch's is a community-hosted mirror
> of the exact same weights and architecture, published without Meta's access-request
> gate — used here because that gated-access approval was still pending at the time. This
> doesn't change anything downstream (tokenizer, chat template, generation behavior are
> all identical); if Meta's approval clears later, switching `BASE_MODEL_NAME` back is a
> one-line change.

### 1. Environment

```bash
conda create -n ambient-scribe python=3.10 -y
conda activate ambient-scribe
pip install -r requirements.txt
pip uninstall -y liger-kernel   # see note below -- skip if the uninstall says "not installed"
```

`requirements.txt` is organized by phase. Phase 1 (data) and the CPU-safe parts of
Phase 4/5 (report generation, LLM judge, the FastAPI app's tests) run anywhere. Phases 2
and 3, and any real model loading in Phase 4/5, need a **CUDA GPU** — `bitsandbytes`
4-bit quantization is CUDA-only and does not run on CPU or Apple Silicon/MPS. This
project was developed against that constraint: code and CLIs were written and validated
on a CUDA-less machine with lazy imports and mocked models, and are meant to actually
execute on a free-tier Colab T4 (see the sizing comments in `src/train/finetune.py`) or a
rented GPU box.

> **On Colab specifically, run the `pip uninstall -y liger-kernel` line above.** Colab's
> base runtime image ships its own `liger-kernel`, which is not a dependency of anything
> in `requirements.txt` and so is never touched by `pip install -r requirements.txt` —
> but `trl`'s `SFTTrainer` (used by `src/train/finetune.py`) opportunistically imports it
> if present, and Colab's pre-installed version hard-requires `transformers>=4.52`, which
> conflicts with this project's pinned `transformers==4.45.2` (see the comment in
> `requirements.txt` for the full history). This project doesn't use liger-kernel's fused
> kernels — trl only reaches for it if it happens to be importable — so removing it is a
> deliberate, no-downside fix, not a workaround.

`vllm` (used only by `src/serve/serve_vllm.py`) is deliberately **not** in
`requirements.txt` — install it separately on whatever GPU box you serve from; see that
file's docstring for why it's kept out.

### 2. Secrets

```bash
cp .env.example .env
```

Fill in:

| Key | Used by |
|---|---|
| `HF_TOKEN` | `baseline_eval.py`, `finetune.py` — not required for the ungated NousResearch mirror this project defaults to, but still recommended: HF throttles anonymous downloads more aggressively than authenticated ones, and it's required if `BASE_MODEL_NAME` is ever pointed back at Meta's gated repo |
| `WANDB_API_KEY` | `baseline_eval.py`, `finetune.py`, `finetuned_eval.py` — experiment logging |
| `GEMINI_API_KEY` | `llm_judge.py` — LLM-judge scoring (Google AI Studio, free tier, no credit card required) |

`.env` is gitignored; never commit it.

### 3. Run each phase in order

```bash
# Phase 1 — data
python -m src.data.load_data

# Phase 2 — zero-shot baseline (needs a CUDA GPU)
python -m src.eval.baseline_eval
# quick smoke test first:
python -m src.eval.baseline_eval --sample_size 20

# Phase 3 — QLoRA fine-tune + merge (needs a CUDA GPU)
python -m src.train.finetune
python -m src.train.merge_adapter

# Phase 4 — fine-tuned eval + comparison
python -m src.eval.finetuned_eval
python -m src.eval.llm_judge --results_path outputs/metrics/baseline_results.json
python -m src.eval.llm_judge --results_path outputs/metrics/finetuned_results.json
python -m src.eval.generate_report

# Phase 5 — serve
python -m src.serve.app                     # transformers pipeline, default path
python -m src.serve.serve_vllm              # vLLM path, needs `pip install vllm` + GPU

curl -X POST http://localhost:8000/generate-note \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Doctor: What brings you in today? Patient: I have had a cough for a week..."}'
```

### 4. Tests (no GPU needed)

```bash
python -m pytest tests/ -v
```

## Results

<!--
Pulled from outputs/metrics/comparison_report.md once a real training/eval run has
produced it. Do not fill this in by hand with estimated numbers -- regenerate it with
`python -m src.eval.generate_report` and copy the table over.
-->

**[Results pending — training run in progress]**

`outputs/metrics/comparison_report.md` and `outputs/figures/comparison_chart.png` do not
exist yet in this repo — no GPU training/eval run has been executed. Once one has, this
section will be replaced with the actual ROUGE-1/2/L, LLM-judge (completeness / factual
correctness / structural adherence / hallucination rate) comparison table and chart from
that file, including an explicit call-out if hallucination rate went up despite better
ROUGE (see `src/eval/generate_report.py` — it's built to flag that automatically, not
hide it).

## Known limitations

- **Small training set.** ~1,200 training examples after cleaning — enough to adapt
  style/structure with LoRA, not enough to expect the fine-tune to generalize the way a
  larger clinical corpus would.
- **Synthetic dialogues.** MTS-Dialog's conversations were constructed for the dataset,
  not captured from real clinical encounters — tone, structure, and information density
  may not match real visits.
- **Single note section per example, not a full SOAP note.** See the Dataset section
  above — this project generates one clinical-note section at a time, not a complete
  multi-part note.
- **Single base-model comparison.** Only one base model (Llama 3 8B Instruct) and one
  fine-tuning configuration are evaluated — this is a before/after comparison for one
  model, not a survey across model families or LoRA configurations.
- **Portfolio scope.** No human clinical review of outputs, no production error handling
  for adversarial/out-of-distribution inputs, no HIPAA-grade infrastructure — see
  Disclaimer below.

## Disclaimer

This is a **portfolio and research project**, built to demonstrate an end-to-end
fine-tuning + evaluation + serving pipeline. It is **not a clinically validated tool**,
has not been reviewed by clinicians, and is **not intended for use on real patient data**
or in any real clinical workflow. All development and testing here uses only the public,
synthetic MTS-Dialog dataset.
