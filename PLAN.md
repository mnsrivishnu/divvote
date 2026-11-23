# DivVote — Semantically-Diverse Prompt Ensembling for LLM-Based Causal Discovery

**Full Project Plan & Handoff Document**

| | |
|---|---|
| **Author** | SRI VISHNU M N (727722EUCI048) |
| **Supervisor** | Mr. Vimit Varghese |
| **Batch** | 48 (2022–2027) |
| **Project Type** | Research-oriented — algorithmic modification |
| **Base Paper** | Shaposhnyk & Yanushkevich, "Augmenting Expert Elicitation With LLMs for Building Causal Models to Aid Decision Making" |
| **Domain** | LLM-Based Causal Discovery / NLP / Ensemble Learning |
| **Target Venue** | IEEE TKDE (primary), IEEE TAI (backup) |
| **Course** | M.Tech CSE — Project Phase II — Zeroth Review, Sri Krishna College of Engineering and Technology |

---

## 1. Executive Summary

**Project name:** DivVote — Semantically-Diverse Prompt Ensembling for LLM-Based Causal Discovery.

**One-sentence description:** DivVote replaces the base paper's seven fixed causal verbs with k semantically-decorrelated prompt medoids selected by clustering an LLM-generated prompt pool in embedding space, aggregated by learned reliability weighting with per-edge confidence output, and evaluated head-to-head against the base paper on five standard medical Bayesian-network benchmarks.

**Research contribution type:** Algorithmic modification to an existing IEEE TKDE method, grounded in classical ensemble-diversity theory (Krogh & Vedelsby, 1995). Not an application; not a product.

**Expected headline result:** ≥20% reduction in Structural Hamming Distance (SHD) on all five benchmarks (Asia, Cancer, Medical Diagnostic, Covid-19, Endorisk), with ~30% fewer LLM API calls, and a new per-edge confidence output calibrated to Expected Calibration Error (ECE) below 0.1.

---

## 2. Base Paper — What It Does

**Full citation:** O. Shaposhnyk and S. Yanushkevich, "Augmenting Expert Elicitation With LLMs for Building Causal Models to Aid Decision Making," *IEEE Transactions on Knowledge and Data Engineering*, Vol. 38, No. 12, Dec. 2026. DOI: 10.1109/TKDE.2026.3731098

### 2.1 Core idea
LLMs are sensitive to the exact phrasing of causal questions — asking "Does X cause Y?" vs. "Does X lead to Y?" gives different answers. The base paper fixes this by querying the LLM with 7 different causal verbs for every variable pair and taking a majority vote (threshold k ≥ 4 out of 7).

### 2.2 The 7 fixed verbs
- cause
- lead to
- affect
- influence
- raise the risk of
- increase the chance of
- result in

Prompt template: **"Does {X} {verb} {Y}? Answer Yes or No."**

### 2.3 Algorithm (verbatim from base paper)
```
INPUT:
    V     = set of variables
    LLM   = GPT-4o-mini
    verbs = [cause, lead to, affect, influence, raise the risk of,
             increase the chance of, result in]
    k     = 4    # majority threshold

For each ordered pair (X, Y) in V x V:
    agree = 0
    For each verb v in verbs:
        prompt = f"Does {X} {v} {Y}? Answer Yes or No."
        if LLM(prompt) == "Yes": agree += 1

    if agree >= k:
        add edge X -> Y

Return the resulting directed graph G
```

### 2.4 Benchmarks
- **Asia** (8 variables, 8 edges) — smallest, classic Lauritzen & Spiegelhalter 1988 lung-disease network.
- **Cancer** (5 variables, 4 edges) — simplest sanity-check network.
- **Medical Diagnostic** (~18 variables, ~25 edges) — general hospital diagnostic reasoning.
- **Covid-19** (~12 variables, ~18 edges) — modern (built 2020-2022), not in LLM pre-training as a textbook example.
- **Endorisk** (~20 variables, ~25 edges) — endometrial-cancer risk clinical network; largest and hardest.

All five are publicly available in standard `.bif` format from bnlearn.com/bnrepository/.

### 2.5 Metrics
- **SHD (Structural Hamming Distance)** = (missing edges) + (extra edges) + (reversed edges). Master metric. Lower = better.
- **TERE (True Edge Recall Error)** = number of true edges missed by the algorithm. Measures recall failure.
- **TENE (Total Edge Not Expected)** = number of false edges added by the algorithm. Measures hallucination.

### 2.6 Key findings (from the base paper's own experiments)
- Voting with k ≥ 4 beats any single-verb baseline on all 5 benchmarks.
- Optimal verb varies by domain: *result in* wins on Covid-19, *raise the risk of* wins on Medical Diagnostic. No universal best.
- SHD remains above zero on every benchmark — there is room to improve.
- Computational complexity: 7 × n(n-1) LLM calls per graph; quadratic in n.

---

## 3. Research Gap

The base paper has four interconnected weaknesses, all rooted in one cause: **the prompt ensemble is not designed for diversity**.

| # | Gap | Why it matters |
|---|---|---|
| 1 | 7 verbs are semantically correlated (high cosine similarity in sentence embedding) | Correlated votes are not independent → over-counts agreement → violates ensemble theory |
| 2 | Verbs chosen by author intuition; no selection rule | Unreproducible; cannot adapt to new domains; arbitrary count (why 7?) |
| 3 | Equal-weight majority voting | Base paper itself shows reliability varies by domain, but uniform rule throws this information away |
| 4 | Binary output; no per-edge confidence | Downstream systems cannot filter, abstain, or calibrate |

**Gap in one sentence:** "The base paper's majority vote uses 7 hand-picked, semantically-correlated causal verbs with equal weights — ignoring ensemble-diversity theory, giving no confidence output, and leaving measurable SHD on every benchmark."

---

## 4. DivVote — Our Algorithm

### 4.1 Core idea
Replace the 7 fixed correlated verbs with **k semantically-decorrelated prompt medoids**, selected by clustering an LLM-generated prompt pool in sentence-embedding space, then aggregate votes using **reliability-weighted voting** with per-edge confidence output.

### 4.2 Full algorithm
```
INPUT:
    V       = variables
    LLM_v   = voting LLM (GPT-4o-mini or local Llama/Mistral)
    LLM_g   = generator LLM (same or different)
    encoder = sentence encoder (bge-m3 or e5-large-v2)
    N       = 25    # candidate pool size
    k       = 5     # cluster count (hyperparameter, tuned)
    gamma   = 1     # reliability sharpness

STEP 1 - Prompt pool generation (one-time, offline)
    Seed with 6 causal-framing categories:
        direct, risk, counterfactual, interventional, mechanistic, temporal
    Ask LLM_g to produce ~4 variants per category -> pool P = {p_1..p_N}
    Deduplicate: drop any pair with cosine > 0.95

STEP 2 - Semantic embedding
    e_i = encoder(p_i) for each prompt

STEP 3 - Diversity clustering
    kmeans on {e_i} with k clusters
    Select medoid m_j of each cluster -> final prompts {m_1..m_k}

STEP 4 - Reliability learning (on held-out validation benchmark)
    For each cluster j:
        r_j = accuracy of medoid m_j on validation edges
        w_j = r_j^gamma / sum_i r_i^gamma     # normalized reliability weights

STEP 5 - Voting across medoids (per variable pair)
    For each ordered pair (X, Y) in V x V:
        For j in 1..k:
            answer_j = LLM_v(format(m_j, X, Y))
        edge_score = sum_j (w_j * answer_j) / sum_j w_j
        edge(X, Y)  = 1 if edge_score >= 0.5 else 0
        confidence  = |2 * edge_score - 1|

Return directed graph G with per-edge confidence
```

### 4.3 Prompt pool — 6 seed categories
| Category | Example template |
|---|---|
| Direct causation | Does {X} cause {Y}? |
| Risk / probability | Does {X} increase the probability of {Y}? |
| Counterfactual | Would {Y} occur without {X}? |
| Interventional | If we intervene on {X}, does {Y} change? |
| Mechanistic | Is {Y} mechanistically downstream of {X}? |
| Temporal | Does {X} temporally precede {Y}? |

### 4.4 Four weighting schemes (ablated in thesis)
| Scheme | Formula | Rationale |
|---|---|---|
| Uniform | w_j = 1/k | Baseline; each cluster equal |
| Cluster size | w_j = \|C_j\| / sum \|C_i\| | Dominant framings weigh more |
| Inverse variance | w_j prop 1/sigma_j^2 | Tight clusters more trustworthy |
| Learned reliability | w_j = r_j^gamma / sum r_i^gamma | Validation-learned per-cluster accuracy; PUBLISHABLE CENTREPIECE |

---

## 5. Head-to-Head Comparison with Base Paper

| Aspect | Base Paper (2026) | DivVote (Ours) |
|---|---|---|
| Prompt source | 7 fixed hand-picked verbs | ~25 LLM-generated, 6-category-seeded |
| Selection rule | Author intuition | Semantic clustering -> k medoids |
| Diversity guarantee | None | Enforced by k-means in embedding space |
| Weighting | Uniform (majority) | Learned reliability (also uniform/size/variance ablations) |
| Output | Binary edge decision | Edge + confidence in [0, 1] |
| LLM calls per pair | 7 | k (typically 5) |
| Theoretical basis | Majority voting intuition | Ensemble diversity theory (Krogh & Vedelsby 1995) |
| Adapts to domain? | No | Yes (reliability learned per domain) |
| Confidence calibration | N/A | ECE measured and reported |

---

## 6. Research Contributions

- **C1 — Novel method:** Semantic cluster voting for LLM-based causal discovery. First application of ensemble-diversity theory to LLM prompt ensembles.
- **C2 — Novel method:** Learned reliability weighting that adapts per-cluster weights from validation data — something the base paper's uniform majority rule cannot do.
- **C3 — Novel output:** Per-edge confidence score, with Expected Calibration Error (ECE) as the reliability metric. Base paper has no confidence output.
- **C4 — Empirical findings:** Head-to-head measurements against the base paper on 5 benchmarks, plus ablations across k (3/5/7/10) and 4 weighting schemes.
- **C5 — Cross-LLM generalization:** Validation that the technique works across multiple open-weight models (GPT-4o-mini, Llama 3.1-8B, Mistral-7B, Phi-3).
- **C6 — Reproducibility artifact:** Public code release + generated prompt pool as reusable scientific artifact.

---

## 7. Evaluation Plan

### 7.1 Datasets (same as base paper)
- Asia (8 variables)
- Cancer (5 variables)
- Medical Diagnostic (~18 variables)
- Covid-19 (~12 variables)
- Endorisk (~20 variables)

All available from bnlearn.com/bnrepository/ as .bif files, or via the Python `bnlearn` package.

### 7.2 Metrics
- **SHD** — primary metric; direct comparison with base paper.
- **TERE** — missed true edges.
- **TENE** — invented false edges.
- **ECE (Expected Calibration Error)** — new metric for confidence calibration; target < 0.1.
- **LLM calls per graph** — cost/efficiency metric.
- **Statistical significance:** paired t-test or Wilcoxon over 5 runs per benchmark.

### 7.3 Baselines to compare against
- Single-verb baselines (one per verb from base paper)
- Base paper's 7-verb uniform majority vote (THE baseline)
- DivVote with uniform weighting (isolates clustering contribution)
- DivVote with learned reliability (full method)
- Random prompt subsets of size k (sanity check)

### 7.4 Ablation axes
- Cluster count k = 3, 5, 7, 10
- Clustering algorithm: k-means vs hierarchical vs HDBSCAN
- Weighting scheme: uniform / size / variance / learned
- Sentence encoder: bge-m3 vs e5-large-v2 vs all-MiniLM-L6-v2
- Variant generator: GPT-4o-mini vs Claude vs Llama
- Voting LLM: GPT-4o-mini vs Llama 3.1-8B vs Mistral-7B vs Phi-3

---

## 8. Code Structure and Tech Stack

### 8.1 Directory layout
```
divvote/
|-- src/
|   |-- prompts/        # Pool generation + templates
|   |-- embedding/      # Sentence encoder wrapper
|   |-- clustering/     # k-means / medoid selection
|   |-- voting/         # Aggregation + weighting schemes
|   |-- metrics/        # SHD, TERE, TENE, ECE
|   |-- llm/            # LLM interface (OpenAI, local vLLM)
|-- data/
|   |-- benchmarks/     # 5 ground-truth DAGs (.bif + .npy)
|   |-- prompts/        # Generated prompt pools (.json)
|-- experiments/
|   |-- baseline/       # Reproduce base paper
|   |-- divvote/        # Our contribution
|-- results/            # CSV tables, plots
|-- notebooks/          # Jupyter for exploration
|-- thesis_draft/       # Writing-as-you-go
|-- README.md
|-- requirements.txt
`-- .gitignore
```

### 8.2 Dependencies (requirements.txt)
```
torch
transformers
sentence-transformers
scikit-learn
numpy
pandas
matplotlib
seaborn
openai
bnlearn
pgmpy
causal-learn
vllm            # optional, needs GPU
jupyter
tqdm
```

### 8.3 LLM interface skeleton
```python
from openai import OpenAI
client = OpenAI()

def llm_vote(prompt: str, model="gpt-4o-mini") -> int:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": prompt + " Answer only Yes or No."}],
        temperature=0, max_tokens=5,
    )
    return 1 if "yes" in resp.choices[0].message.content.lower() else 0
```

### 8.4 Metrics skeleton
```python
import numpy as np

def shd(pred, true):
    missing  = ((true == 1) & (pred == 0)).sum()
    extra    = ((true == 0) & (pred == 1)).sum()
    reversed_= ((true == 1) & (pred.T == 1) & (pred == 0)).sum()
    return int(missing + extra + reversed_)

def tere(pred, true):
    return int(((true == 1) & (pred == 0)).sum())

def tene(pred, true):
    return int(((true == 0) & (pred == 1)).sum())
```

### 8.5 Medoid selection skeleton
```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

encoder = SentenceTransformer("BAAI/bge-m3")

def select_medoids(prompts, k=5):
    E = encoder.encode(prompts)
    km = KMeans(n_clusters=k, random_state=42).fit(E)
    medoids = []
    for c in range(k):
        idxs = np.where(km.labels_ == c)[0]
        centroid = km.cluster_centers_[c]
        best = min(idxs, key=lambda i: np.linalg.norm(E[i] - centroid))
        medoids.append(prompts[best])
    return medoids
```

---

## 9. 6-Month Timeline (Nov 2026 – Apr 2027)

| Month | Milestones | Deliverables |
|---|---|---|
| Nov 2026 | Environment setup; read base paper 3x; download 5 benchmarks; implement metrics (SHD/TERE/TENE) | Clean Git repo; base-paper notes; 5 benchmarks loaded |
| Dec 2026 | Reproduce base paper on all 5 benchmarks; generate prompt pool; implement embedding + clustering | baseline_reproduction.csv matching the paper; first DivVote run |
| Jan 2027 | DivVote on all 5 benchmarks; implement 4 weighting schemes; ablation across k = 3/5/7/10 | DivVote results across all weight schemes; ablation table v1 |
| Feb 2027 | Learned reliability weighting; per-edge confidence; ECE calibration; cross-LLM study (3-4 models) | Reliability weights; ECE numbers |
| Mar 2027 | Full experiments; statistical significance; human-interpretability study; IEEE TKDE draft; arXiv | Complete tables, figures; paper draft |
| Apr 2027 | Submit paper; finalize thesis; demo; viva prep | Submitted paper; thesis PDF; GitHub release; viva presentation |

---

## 10. First 3 Weeks — Concrete Actions

### Week 1 (Nov 23–30): Foundation
- Day 1-2: Create GitHub repo; set up Python env; install dependencies.
- Day 3-5: Read base paper 3 times. Write base_paper_notes.md.
- Day 6-7: Download 5 benchmarks from bnlearn repository; parse to .npy.
- End-of-week deliverable: Clean repo + base-paper notes + 5 benchmarks loaded.

### Week 2 (Dec 1–7): Reproduce base paper
- Day 1-2: Implement metrics (SHD, TERE, TENE) with unit tests.
- Day 3-4: Implement LLM interface (OpenAI GPT-4o-mini wrapper).
- Day 5-6: Implement base paper's 7-verb majority voting algorithm.
- Day 7: Run on all 5 benchmarks; compare to paper's numbers (within +-10%).
- End-of-week deliverable: baseline_reproduction.csv matching the paper.

### Week 3 (Dec 8–14): First DivVote run
- Day 1: Generate prompt pool (~25 variants) using LLM + seed categories.
- Day 2: Embed pool with bge-m3; cluster with k-means; select k=5 medoids.
- Day 3-4: Run DivVote with uniform weighting on Asia (smallest benchmark).
- Day 5-6: Implement learned reliability weighting using Asia as validation; run on Cancer, Medical Diagnostic, Covid-19, Endorisk.
- Day 7: First head-to-head comparison table; commit everything to Git.
- End-of-week deliverable: First DivVote vs base paper comparison table.

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cannot reproduce base paper numbers | Low | High | Debug prompts; check LLM temperature=0; re-read paper. Don't move on until baseline matches |
| LLM API cost overruns | Low | Medium | Cache responses; small-scale tests on Asia first; estimated total cost < $30 |
| DivVote fails to improve on base paper | Medium | High | Try different k values; different clustering; different weighting. Even if marginal, the method is publishable |
| Local LLM GPU access limited | Medium | Low | Use OpenAI API primarily; local LLM is bonus cross-validation |
| ECE calibration too high (>0.1) | Medium | Low | Apply post-hoc temperature scaling; report both pre- and post-calibration |
| Someone publishes similar work first | Low | High | Post arXiv preprint early (March); forensic case-study section (if needed) |

---

## 12. Success Criteria

**Minimum (must have):**
- Base paper reproduced on all 5 benchmarks.
- DivVote implemented end-to-end with at least uniform weighting.
- Head-to-head comparison table on at least 3 benchmarks.
- Thesis document (80-120 pages) defended.

**Target (expected):**
- All above + all 4 weighting schemes ablated.
- ECE calibration reported.
- Cross-LLM study on 3+ models.
- arXiv preprint posted.
- IEEE TKDE submission.

**Stretch (if time permits):**
- Open-source release with reproducibility GitHub repo (100+ stars).
- Secondary paper on cross-domain application.
- Conference presentation at an Indian AI venue.

---

## 13. Key References
- **[BASE PAPER]** O. Shaposhnyk and S. Yanushkevich, "Augmenting Expert Elicitation With LLMs for Building Causal Models to Aid Decision Making," *IEEE Transactions on Knowledge and Data Engineering*, Vol. 38, No. 12, Dec. 2026. DOI: 10.1109/TKDE.2026.3731098
- A. Krogh and J. Vedelsby, "Neural Network Ensembles, Cross Validation, and Active Learning," *NeurIPS 1995*. [Ensemble-diversity theory]
- S. Pan et al., "Unifying Large Language Models and Knowledge Graphs: A Roadmap," *IEEE TKDE*, Vol. 36, No. 7, 2024.
- W. Shang and X. Huang, "A Survey of LLMs on Generative Graph Analytics," *IEEE TKDE*, Vol. 37, No. 12, Dec. 2025.
- C. N. Hang, P.-D. Yu, C. W. Tan, "TrumorGPT: Graph-based RAG LLM for Fact-Checking," *IEEE TAI*, Vol. 6, No. 11, 2025.
- E. Kiciman et al., "Causal Reasoning and LLMs: Opening a New Frontier for Causality," arXiv, 2023.
- T. Ban et al., "Causal Structure Learning Supervised by Large Language Model," arXiv:2311.11689, 2023.
- A. Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," *ICLR 2024*.
- L. Kuncheva, "Combining Pattern Classifiers: Methods and Algorithms," Wiley, 2004. [Classical ensemble weighting reference]
- bnlearn Bayesian network repository: bnlearn.com/bnrepository/ — source for Asia, Cancer, Medical Diagnostic, Covid-19, Endorisk .bif files.

---

## 14. Key Facts for Continuation
- Project is **algorithmic research**, not application.
- Base paper uses **7 fixed causal verbs** with majority voting; we replace with **k=5 semantically-clustered medoids**.
- Same 5 medical benchmarks: Asia, Cancer, Medical Diagnostic, Covid-19, Endorisk.
- Same primary metrics: SHD, TERE, TENE. We add ECE for calibration.
- Weighting schemes: uniform, size, inverse-variance, learned-reliability.
- Target venue: **IEEE TKDE** (same as base paper).
- Timeline: Nov 2026 – Apr 2027.
