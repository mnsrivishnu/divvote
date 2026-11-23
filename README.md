# DivVote — Semantically-Diverse Prompt Ensembling for LLM-Based Causal Discovery

DivVote replaces the base paper's seven fixed causal verbs with **k
semantically-decorrelated prompt medoids**, selected by clustering an
LLM-generated prompt pool in embedding space, aggregated by **learned
reliability weighting** with **per-edge confidence output**, and evaluated
head-to-head against the base paper on five standard medical Bayesian-network
benchmarks.

## What this is

- **Research contribution type:** Algorithmic modification to an existing IEEE
  TKDE method (Shaposhnyk & Yanushkevich, 2026), grounded in classical
  ensemble-diversity theory (Krogh & Vedelsby, 1995). Not an application; not a
  product.
- **Expected headline result:** ≥20% reduction in Structural Hamming Distance
  (SHD) on all five benchmarks (Asia, Cancer, Medical Diagnostic, Covid-19,
  Endorisk), with ~30% fewer LLM API calls, and a new per-edge confidence
  output calibrated to an Expected Calibration Error (ECE) below 0.1.

## Project info

| | |
|---|---|
| **Author** | SRI VISHNU M N (727722EUCI048) |
| **Supervisor** | Mr. Vimit Varghese |
| **Target venue** | IEEE TKDE (primary), IEEE TAI (backup) |
| **Course** | M.Tech CSE — Project Phase II, Sri Krishna College of Engineering and Technology |

See [PLAN.md](PLAN.md) for the full project plan, algorithm, evaluation plan,
and timeline.

## Repository layout

```
divvote/
├── src/
│   ├── prompts/     # Pool generation + templates
│   ├── embedding/   # Sentence encoder wrapper
│   ├── clustering/  # k-means / medoid selection
│   ├── voting/      # Aggregation + weighting schemes
│   ├── metrics/     # SHD, TERE, TENE, ECE
│   └── llm/         # LLM interface (OpenAI, local vLLM)
├── data/
│   ├── benchmarks/  # 5 ground-truth DAGs (.bif + .npy)
│   └── prompts/     # Generated prompt pools (.json)
├── experiments/
│   ├── baseline/    # Reproduce base paper
│   └── divvote/     # Our contribution
├── results/         # CSV tables, plots
├── notebooks/       # Jupyter for exploration
└── thesis_draft/    # Writing-as-you-go
```

## Setup

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Unix:     source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```
