"""Base paper's 7-verb majority-vote causal discovery (the reproduction baseline).

Purpose
-------
Reproduce the base paper (Shaposhnyk & Yanushkevich, IEEE TKDE 2026): for every
ordered variable pair (X, Y), query the LLM once per causal verb, and add a
directed edge X -> Y when at least k of the 7 verbs vote "yes" (majority,
k = 4). This is THE baseline every DivVote result is compared against
(PLAN.md sections 2.3 / 7.3).

Two prompt styles (see CLAUDE.md decision):
  - "reference" (default): the base paper's exact prompt from its repo
    (reference/LLM-for-Causal-Discovery.ipynb, "Table II") -- a system + human
    message pair that includes each variable's discrete states and asks for a
    'True'/'False' answer. Required for numbers comparable to the base paper.
  - "simple": the literal template from PLAN.md section 2.3
    ("Does {X} {verb} {Y}? Answer Yes or No."). Needs no state labels; useful
    as a prompt ablation, but NOT directly comparable to the published numbers.

The reference style needs per-variable state labels; these are transcribed
verbatim from the base paper's notebook for Asia, Cancer, and Medical
Diagnostic (the three networks that repo ships). Covid-19 and Endorisk have no
such labels in the reference, so the reference style raises for them until
states are provided; the simple style works for all five.

Maps to: Week 2, Days 5-6 (PLAN.md section 10) — "Implement base paper's 7-verb
majority voting algorithm" and run on the benchmarks.

Run from the repo root, e.g.:
    python -m src.voting.baseline asia
    python -m src.voting.baseline asia --prompt-style simple
"""

import argparse
from pathlib import Path

import numpy as np

from src.llm import interface
from src.metrics import metrics
from src.voting import postprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "data" / "benchmarks"

# The 7 fixed causal verbs (PLAN.md section 2.2). Majority vote is
# order-independent; the same 7 verbs are used by the base paper.
CAUSAL_VERBS = [
    "cause",
    "lead to",
    "affect",
    "influence",
    "raise the risk of",
    "increase the chance of",
    "result in",
]

MAJORITY_K = 4  # add edge when yes-votes >= k (of 7)
DEFAULT_MODEL = interface.DEFAULT_MODEL

# --- Reference prompt (verbatim from the base paper's notebook, "Table II") ---
SYSTEM_PROMPT = (
    "You are a reasoning assistant tasked with identifying whether a *causal* "
    "relationship exists between two medical-related variables. Each variable "
    "has discrete states, listed in parentheses. Assume that variables "
    "represent conditions, risk factors, symptoms, or diagnostic test results "
    "related to a single patient. Causality should reflect real-world "
    "biomedical reasoning: for example, a disease might cause a symptom, or a "
    "risk factor might cause a disease. Only return 'True' if changing the "
    "first variable is likely to lead to a change in the second. Return only "
    "'True' or 'False'. Do not confuse correlation with causation."
)
HUMAN_PROMPT = (
    "Consider the following two variables:\n"
    "  - {cause}: ({cause_states})\n"
    "  - {effect}: ({effect_states})\n\n"
    "Is it plausible that changes in '{cause}' can {causal_verb} '{effect}'?"
    "Answer only with 'True' or 'False'."
)

# --- Simple prompt (PLAN.md section 2.3) ---
SIMPLE_PROMPT = "Does {cause} {causal_verb} {effect}?"

# Per-variable (display name, states) transcribed from the base paper's
# notebook. Keyed by benchmark, then by the node name as stored in the
# <name>_nodes.txt file (so lookups align with the ground-truth node order).
NODE_STATES = {
    "asia": {
        "Visited_Asia": ("Visited Asia", "yes/no"),
        "Smoker": ("Smoker", "yes/no"),
        "Tuberculosis": ("Tuberculosis", "yes/no"),
        "Lung_Cancer": ("Lung Cancer", "yes/no"),
        "Bronchitis": ("Bronchitis", "yes/no"),
        "individual_has_either_Tuberculosis_or_Lung_Cancer": (
            "individual has either Tuberculosis or Lung Cancer", "yes/no"),
        "X-ray_Result": ("X-ray Result", "positive/negative"),
        "Dyspnea": ("Dyspnea", "yes/no"),
    },
    "cancer": {
        "Exposure_to_pollutants": ("Exposure to pollutants", "low/high"),
        "Smoker": ("Smoker", "yes/no"),
        "individual_has_Cancer": ("individual has Cancer", "yes/no"),
        "Dyspnea": ("Dyspnea", "yes/no"),
        "X-ray_Result": ("X-ray Result", "positive/negative"),
    },
    "medical_diagnostic": {
        "Eat Fatty Food": (
            "Eat Fatty Food", "Consumes fatty foods) / Does not consume fatty foods"),
        "Arteriosclerosis": ("Arteriosclerosis", "Present/Absent"),
        "Right Heart Syndrome": ("Right Heart Syndrome", "Present/Absent"),
        "Left Heart Syndrome": ("Left Heart Syndrome", "Present/Absent"),
        "Lungs Sound Funny": ("Lungs Sound Funny", "Abnormal sounds/Normal sounds"),
        "Difficulty Breathing": (
            "Difficulty Breathing", "Experiences difficulty/No difficulty"),
        "Smoking": ("Smoking", "Smoker/Non-smoker"),
        "Radon Exposure": ("Radon Exposure", "Exposed/Not exposed"),
        "Lung Cancer": ("Lung Cancer", "Present/Absent"),
        "Cough Up Blood": ("Cough Up Blood", "Yes/No"),
    },
}


def load_ground_truth(benchmark_name):
    """Return (adjacency int matrix, node list) for a benchmark from disk."""
    adj_path = BENCH_DIR / f"{benchmark_name}.npy"
    nodes_path = BENCH_DIR / f"{benchmark_name}_nodes.txt"
    if not adj_path.exists() or not nodes_path.exists():
        raise FileNotFoundError(
            f"Missing {adj_path.name} / {nodes_path.name}. Run "
            "`python -m src.data.load_benchmarks` first."
        )
    adj = np.load(adj_path)
    nodes = nodes_path.read_text(encoding="utf-8").splitlines()
    return adj, nodes


def _vote_pair(cause, effect, verb, prompt_style, states, model, use_cache):
    """Return the LLM's binary vote (1/0) for a single (cause, effect, verb)."""
    if prompt_style == "reference":
        c_disp, c_states = states[cause]
        e_disp, e_states = states[effect]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": HUMAN_PROMPT.format(
                cause=c_disp, cause_states=c_states,
                effect=e_disp, effect_states=e_states, causal_verb=verb)},
        ]
        return interface.llm_vote_messages(
            messages, positive_token="true", model=model, use_cache=use_cache)
    # simple style
    prompt = SIMPLE_PROMPT.format(
        cause=cause.replace("_", " "), effect=effect.replace("_", " "),
        causal_verb=verb)
    return interface.llm_vote(prompt, model=model, use_cache=use_cache)


def run_voting(nodes, prompt_style="reference", k=MAJORITY_K,
               model=DEFAULT_MODEL, use_cache=True, states=None, vote_fn=None,
               return_scores=False):
    """Run 7-verb majority voting over all ordered pairs; return adjacency.

    adj[i, j] == 1 iff at least k of the 7 verbs voted yes for node_i -> node_j.
    The node order of `adj` matches `nodes` exactly. Pass `vote_fn(cause,
    effect, verb) -> 0/1` to inject a stand-in LLM (used in tests). With
    return_scores=True, also returns the yes-vote-count matrix.
    """
    n = len(nodes)
    scores = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if vote_fn is not None:
                yes = sum(vote_fn(nodes[i], nodes[j], v) for v in CAUSAL_VERBS)
            else:
                yes = sum(
                    _vote_pair(nodes[i], nodes[j], v, prompt_style,
                               states, model, use_cache)
                    for v in CAUSAL_VERBS)
            scores[i, j] = yes
    adj = (scores >= k).astype(int)
    np.fill_diagonal(adj, 0)
    if return_scores:
        return adj, scores
    return adj


def _variant_label(cycle_removal, mediator_removal):
    if not cycle_removal and not mediator_removal:
        return "Raw"
    if cycle_removal and mediator_removal:
        return "Pruned"
    return "Pruned (partial)"


def run_baseline(benchmark_name, prompt_style="reference", k=MAJORITY_K,
                 model=DEFAULT_MODEL, use_cache=True, vote_fn=None,
                 cycle_removal=False, mediator_removal=False, verbose=True):
    """Load ground truth, run the baseline, score it, print and return results.

    Post-processing (both default off = "Raw"; both on = "Pruned"):
      - cycle_removal: resolve bidirectional 2-cycle conflicts
      - mediator_removal: transitive reduction (drop mediated direct edges)
    """
    true_adj, nodes = load_ground_truth(benchmark_name)

    states = None
    if prompt_style == "reference" and vote_fn is None:
        states = NODE_STATES.get(benchmark_name)
        if states is None:
            raise ValueError(
                f"No variable-state labels for '{benchmark_name}' (reference "
                "style). Add them to NODE_STATES, or use prompt_style='simple'."
            )
        missing = [nd for nd in nodes if nd not in states]
        if missing:
            raise ValueError(
                f"Missing state labels for nodes in '{benchmark_name}': {missing}"
            )

    variant = _variant_label(cycle_removal, mediator_removal)
    if verbose:
        n_calls = len(CAUSAL_VERBS) * len(nodes) * (len(nodes) - 1)
        print(f"Running base-paper baseline on '{benchmark_name}' "
              f"({len(nodes)} nodes, prompt_style={prompt_style}, k={k}, "
              f"variant={variant})")
        print(f"LLM calls (max, pre-cache): {n_calls}\n")

    pred_adj, scores = run_voting(
        nodes, prompt_style=prompt_style, k=k, model=model,
        use_cache=use_cache, states=states, vote_fn=vote_fn, return_scores=True)

    if cycle_removal or mediator_removal:
        pred_adj = postprocess.prune(
            pred_adj, scores=scores,
            cycle_removal=cycle_removal, mediator=mediator_removal)

    result = metrics.evaluate(pred_adj, true_adj)
    result["benchmark"] = benchmark_name
    result["variant"] = variant
    result["pred_edges"] = int(pred_adj.sum())
    result["true_edges"] = int(true_adj.sum())
    result["pred_adj"] = pred_adj

    if verbose:
        print(f"{'metric':<20}{'value':>10}")
        print("-" * 30)
        print(f"{'true edges':<20}{result['true_edges']:>10}")
        print(f"{'predicted edges':<20}{result['pred_edges']:>10}")
        print(f"{'SHD':<20}{result['SHD']:>10}")
        print(f"{'SHD (norm)':<20}{result['SHD_normalized']:>10.4f}")
        print(f"{'TENE':<20}{result['TENE']:>10}")
        print(f"{'TENE (norm)':<20}{result['TENE_normalized']:>10.4f}")
        print(f"{'TERE':<20}{result['TERE']:>10}")
        print(f"{'TERE (norm)':<20}{result['TERE_normalized']:>10.4f}")

    return result


def compare_raw_pruned(benchmark_name, prompt_style="reference", k=MAJORITY_K,
                       model=DEFAULT_MODEL, use_cache=True, vote_fn=None):
    """Run the benchmark Raw and Pruned, print a side-by-side SHD/TENE/TERE table."""
    raw = run_baseline(benchmark_name, prompt_style=prompt_style, k=k,
                       model=model, use_cache=use_cache, vote_fn=vote_fn,
                       cycle_removal=False, mediator_removal=False, verbose=False)
    pruned = run_baseline(benchmark_name, prompt_style=prompt_style, k=k,
                          model=model, use_cache=use_cache, vote_fn=vote_fn,
                          cycle_removal=True, mediator_removal=True, verbose=False)

    print(f"\n{benchmark_name} - base-paper baseline (prompt_style={prompt_style}, "
          f"k={k})")
    print(f"{'metric':<14}{'Raw':>8}{'Pruned':>8}")
    print("-" * 30)
    print(f"{'edges':<14}{raw['pred_edges']:>8}{pruned['pred_edges']:>8}")
    print(f"{'SHD':<14}{raw['SHD']:>8}{pruned['SHD']:>8}")
    print(f"{'TENE':<14}{raw['TENE']:>8}{pruned['TENE']:>8}")
    print(f"{'TERE':<14}{raw['TERE']:>8}{pruned['TERE']:>8}")
    return {"raw": raw, "pruned": pruned}


def main():
    parser = argparse.ArgumentParser(description="Base-paper 7-verb baseline.")
    parser.add_argument("benchmark", help="e.g. asia, cancer, medical_diagnostic")
    parser.add_argument("--prompt-style", choices=["reference", "simple"],
                        default="reference")
    parser.add_argument("--k", type=int, default=MAJORITY_K)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cycle-removal", action="store_true",
                        help="remove bidirectional 2-cycle conflicts")
    parser.add_argument("--mediator-removal", action="store_true",
                        help="transitive reduction (drop mediated direct edges)")
    parser.add_argument("--compare", action="store_true",
                        help="run Raw vs Pruned and print a side-by-side table")
    args = parser.parse_args()
    if args.compare:
        compare_raw_pruned(args.benchmark, prompt_style=args.prompt_style,
                           k=args.k, model=args.model,
                           use_cache=not args.no_cache)
    else:
        run_baseline(args.benchmark, prompt_style=args.prompt_style, k=args.k,
                     model=args.model, use_cache=not args.no_cache,
                     cycle_removal=args.cycle_removal,
                     mediator_removal=args.mediator_removal)


if __name__ == "__main__":
    main()
