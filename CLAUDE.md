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

Week 2 baseline complete and running live; Asia Raw reproduced (SHD 14 vs paper 11, gap attributed to gpt-4o-mini version drift); pruning added. Week 3 Day 3-4: uniform DivVote (k=5) wins 5/5 vs 7-verb baseline (Raw) under reference framing — Asia -4, Cancer -1, Medical -3, Covid -4, Endorisk -26 SHD. Reference style is now canonical for all five (Covid-19/Endorisk NODE_STATES added: Endorisk authoritative, Covid inferred binary). Medical and Covid are still below the 20% relative-improvement target. Week 3 Day 5-6: learned reliability weighting implemented (weights learned on Asia); at the pre-specified gamma=1 it is a no-op vs uniform (near-flat weights never flip a thresholded edge). A gamma-sweep (experiments/gamma_sweep.py, DESCRIPTIVE ONLY — no gamma selected, to avoid leakage) shows at most -2 SHD on the two large graphs (Covid 27->25, Endorisk 46->44, saturating at gamma>=2) and no change on Cancer/Medical. Weighting alone cannot clear 20% on Medical (17%) or Covid (19%); gap-closing work moves to the 0.5 decision threshold, k, and confidence-based filtering. Contribution C3 started: per-edge confidence + ECE in src/metrics/calibration.py (binned ECE over predicted edges; run via experiments/ece_eval.py, cached). Raw (uncalibrated) confidence FAILS the <0.1 target on all five (ECE 0.14-0.40) and is overconfident: edges where all 5 medoids agree (confidence 1.0) are correct <50% of the time on most benchmarks, because confidence measures medoid agreement, not probability of correctness. Post-hoc calibration (temperature scaling, T fit on Asia only) is the next step; report pre- and post-calibration ECE. Section 7.4 cluster-count ablation (experiments/cluster_count_ablation.py, medoids_k{3,7,10}.json, results/cluster_count_ablation.csv; same bge-m3 pool + random_state=42, uniform weighting): k=5 is the sweet spot (wins 5/5 vs baseline) — k>=7 degrades because finer clusters split within-category (direct and mechanistic finally separate at k>=6, but risk/counterfactual/temporal fragment into near-duplicate medoids, reintroducing correlated votes). No k clears the 20% target on Medical (best 17%); k=3 is the only setting that clears it on Covid (23%), but k=3 loses Cancer and weakens Endorisk vs k=5. DESCRIPTIVE ONLY — k not selected from these benchmarks (leakage); k=5 stays the pre-specified choice. Section 4.4 weighting-scheme ablation (experiments/weighting_schemes.py, results/weighting_schemes.csv; all four schemes re-aggregate the cached k=5 votes, zero new API calls): cluster-size weighting (w_j ∝ |C_j|) is the ONLY scheme that beats uniform — validation-consistent, wins/ties held-out 4/5 (Asia 10→9, Medical 15→14, Covid 27→25, Endorisk 46→39, Cancer tie). Learned-reliability (gamma=1) AND inverse-variance (w_j ∝ 1/sigma_j^2) are both no-ops vs uniform (near-flat weights 0.14-0.25 never flip a thresholded edge); only cluster-size has a concentrated weight (the merged direct+mechanistic cluster, |C|=8, gets 0.348). NOTABLE NARRATIVE FINDING: the simple size prior beats the learned-reliability "publishable centrepiece" (C2) — flag for thesis framing, PENDING multi-run confirmation (single-run so far). Section 7.4 clustering-algorithm ablation (experiments/clustering_algorithm.py, medoids_hierarchical.json, medoids_hdbscan.json, results/clustering_algorithm.csv; same bge-m3 pool, uniform weighting): k-means@k=5 is best or tied on all five; hierarchical (k=5) ties except Covid (27→30 worse, merges direct+temporal instead of direct+mechanistic); HDBSCAN (min_cluster_size=2 → k=8 + 3 noise) is worst on all five, fragmenting risk/direct/mechanistic into size-2 near-duplicate clusters. Same lesson as the k-ablation: fewer, broader, well-separated clusters help; fragmentation reintroduces correlated votes and hurts. Section 7.4 encoder ablation (experiments/encoder_ablation.py, medoids_enc_e5-large-v2.json, medoids_enc_minilm-l6.json, results/encoder_ablation.csv; same pool/k=5/random_state=42, uniform weighting): encoder matters a LOT — the three encoders select near-disjoint medoid sets (e5 shares 1/5, MiniLM 2/5 with bge-m3), so medoid selection is NOT robust to the encoder (a documented limitation). bge-m3 stays canonical: clears baseline 5/5. all-MiniLM-L6-v2 (384-dim) fails baseline 5/5 — fragments risk into size-2 clusters, two merged 7-8 clusters, same fragmentation pathology. e5-large-v2 is a credible challenger (wins/ties 4/5 vs bge-m3: Asia 10→9, Medical 15→13, Endorisk 46→39, Cancer tie, loses Covid 27→29), BUT was run WITHOUT its trained query:/passage: prefixes (embedded plain, like bge-m3, for an apples-to-apples selection comparison), so SUGGESTIVE ONLY — needs a prefixed, multi-run follow-up before any switch. All single-run.

## Where to go next

Follow the week-by-week plan in sections 9 and 10 of `PLAN.md`. When asked to "do the next step," check the current-status line above against section 10 and proceed from there.