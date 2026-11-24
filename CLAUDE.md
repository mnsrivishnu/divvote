# CLAUDE.md — DivVote Project Context

This file gives Claude Code persistent context on every session. Read it before doing any work in this repo.

## What this project is

**DivVote — Semantically-Diverse Prompt Ensembling for LLM-Based Causal Discovery.**

Algorithmic research project (M.Tech thesis, Phase II). NOT an application or product. The full plan is in `PLAN.md` — always consult it for details, algorithms, and the timeline.

- **Author:** Sri Vishnu M N (727722EUCI048)
- **Supervisor:** Mr. Vimit Varghese
- **Target venue:** IEEE TKDE (primary), IEEE TAI (backup)
- **Timeline:** Nov 2026 – Apr 2027

## The core idea (one paragraph)

The base paper (Shaposhnyk & Yanushkevich, IEEE TKDE 2026) does LLM-based causal discovery by querying an LLM with **7 fixed, hand-picked causal verbs** per variable pair and taking a majority vote. DivVote replaces those 7 correlated verbs with **k=5 semantically-decorrelated prompt medoids**, selected by clustering an LLM-generated prompt pool in sentence-embedding space, then aggregates votes with **learned reliability weighting** and outputs a **per-edge confidence** score. Grounded in ensemble-diversity theory (Krogh & Vedelsby 1995).

## Benchmarks

Five medical Bayesian networks. Node/edge counts from the base paper's Table I (authoritative): **Asia** (8/8), **Cancer** (5/4), **Medical Diagnostic** (10/9), **Covid-19** (11/20), **Endorisk** (18/37). Asia is the smallest — always test on Asia first.

Get the network files from the base paper's own repo: https://github.com/ExcellentDarkTea/LLM-Causal-Discovery — this guarantees the head-to-head comparison uses the exact same graphs. (Asia/Cancer/Medical Diagnostic also in the bnlearn repository; Covid-19 from Mascaro et al. 2023; Endorisk from Reijnen et al. 2020.)

Base paper reproduction setup to match: GPT-4o-mini, temperature 0, LangChain, majority-vote threshold k >= 4 of 7.

## Metrics

- **SHD** (Structural Hamming Distance) — total differing cells; reversed edge counts twice. Primary metric, lower is better.
- **TENE** (True Edge to Negative Edge) — missed true edges (true=1, pred=0).
- **TERE** (True Edge to Reverse Edge) — reversed edges (true[i,j]=1, pred[j,i]=1).
- **ECE** (Expected Calibration Error) — new; confidence calibration; target < 0.1.
- Definitions follow base paper reference/src/metrics.py exactly. A reversed edge counts in BOTH TENE and TERE — preserve this for comparable numbers.

## Directory layout

See section 8.1 of `PLAN.md`. Key folders: `src/{prompts,embedding,clustering,voting,metrics,llm}`, `data/{benchmarks,prompts}`, `experiments/{baseline,divvote}`, `results/`, `notebooks/`, `thesis_draft/`.

## Conventions and rules

- **Python 3.10+**, dependencies in `requirements.txt`. Always run inside the `venv` virtual environment.
- **Never commit secrets.** The OpenAI key lives in `.env` (gitignored). Use `.env.example` as the template.
- **Determinism:** LLM calls use `temperature=0`. Clustering uses `random_state=42`. This matters for reproducibility.
- **Cost control:** Cache LLM responses where possible. Test on Asia (smallest) before running the big benchmarks. Total API budget is ~$30.
- **Reproduce before you innovate:** Do not move on from the baseline until the base paper's numbers are matched within ±10%.
- **Write tests** for the metrics (SHD/TERE/TENE) — they are the foundation of every result.
- Keep results as CSV in `results/`; keep exploratory work in `notebooks/`.

## Current status

Week 1 (Nov 23–30): Foundation — repo setup and benchmark loading. Update this line as the project progresses.

## Where to go next

Follow the week-by-week plan in sections 9 and 10 of `PLAN.md`. When asked to "do the next step," check the current-status line above against section 10 and proceed from there.