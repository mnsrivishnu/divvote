"""DivVote voting: k medoid prompts, uniform weighting, per-edge confidence.

Purpose
-------
Implement DivVote's voting stage (PLAN.md section 4.2, STEP 5). For each ordered
variable pair (X, Y), query the LLM once per medoid prompt (the k=5
semantically-decorrelated medoids selected in data/prompts/medoids.json), then
aggregate:

    edge_score = sum_j (w_j * answer_j) / sum_j w_j
    edge       = 1 if edge_score >= 0.5 else 0
    confidence = |2 * edge_score - 1|

This module implements the uniform weighting scheme (w_j = 1/k), so edge_score
is the mean vote. The other weighting schemes (size / inverse-variance / learned
reliability, section 4.4) build on this in later weeks.

To keep the comparison against the 7-verb baseline fair, the voting LLM sees the
same conditions as the baseline (same system prompt, the same per-variable
discrete states, and a True/False answer) -- only the question phrasing changes,
substituting a medoid template for the base paper's verb question. prompt_style
can be "reference" (default; needs state labels) or "simple".

Maps to: Week 3, Days 3-4 (PLAN.md section 10) — "Run DivVote with uniform
weighting on Asia."

Run from the repo root:
    python -m src.voting.divvote asia
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.llm import interface
from src.metrics import metrics
from src.voting import baseline

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDOIDS_PATH = REPO_ROOT / "data" / "prompts" / "medoids.json"
DEFAULT_MODEL = interface.DEFAULT_MODEL

# Reference-style human prompt: same variable/states framing as the baseline,
# with the medoid question substituted in place of the verb question.
DIVVOTE_HUMAN = (
    "Consider the following two variables:\n"
    "  - {cause}: ({cause_states})\n"
    "  - {effect}: ({effect_states})\n\n"
    "{question}\n"
    "Answer only with 'True' or 'False'."
)


def load_medoids(path=MEDOIDS_PATH):
    """Return the list of medoid prompt templates from medoids.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    medoids = data.get("medoids", [])
    if not medoids:
        raise ValueError(
            f"No medoids in {path}. Run `python -m src.clustering.medoids` first."
        )
    return medoids


def _fill(template, cause_disp, effect_disp):
    """Substitute {X}/{Y} in a medoid template with the variable display names."""
    return template.replace("{X}", cause_disp).replace("{Y}", effect_disp)


def _vote_pair(cause, effect, medoid, prompt_style, states, model, use_cache):
    """Return the LLM's binary vote (1/0) for one (cause, effect, medoid)."""
    if prompt_style == "reference":
        c_disp, c_states = states[cause]
        e_disp, e_states = states[effect]
        question = _fill(medoid, c_disp, e_disp)
        messages = [
            {"role": "system", "content": baseline.SYSTEM_PROMPT},
            {"role": "user", "content": DIVVOTE_HUMAN.format(
                cause=c_disp, cause_states=c_states,
                effect=e_disp, effect_states=e_states, question=question)},
        ]
        return interface.llm_vote_messages(
            messages, positive_token="true", model=model, use_cache=use_cache)
    # simple style
    prompt = _fill(medoid, cause.replace("_", " "), effect.replace("_", " "))
    return interface.llm_vote(prompt, model=model, use_cache=use_cache)


def collect_votes(nodes, medoids, prompt_style="reference", model=DEFAULT_MODEL,
                  use_cache=True, states=None, vote_fn=None):
    """Query every (ordered pair, medoid) once; return a (k, n, n) vote tensor.

    V[j, i, l] is medoid j's binary vote (0/1) for the edge node_i -> node_l
    (diagonal left 0). This is the single LLM-calling pass; different weighting
    schemes then aggregate the SAME tensor, so no scheme costs extra API calls.
    Pass vote_fn(cause, effect, medoid) -> 0/1 to inject a stand-in LLM (tests).
    """
    n = len(nodes)
    k = len(medoids)
    V = np.zeros((k, n, n), dtype=float)
    for i in range(n):
        for l in range(n):
            if i == l:
                continue
            for jm, m in enumerate(medoids):
                if vote_fn is not None:
                    V[jm, i, l] = vote_fn(nodes[i], nodes[l], m)
                else:
                    V[jm, i, l] = _vote_pair(nodes[i], nodes[l], m, prompt_style,
                                             states, model, use_cache)
    return V


def aggregate_votes(V, weights=None):
    """Aggregate a (k, n, n) vote tensor into (adj, confidence, edge_score).

    edge_score[i, l] = sum_j (w_j * V[j, i, l]) / sum_j w_j  (PLAN.md STEP 5).
    weights=None uses uniform w_j = 1/k. adj = (edge_score >= 0.5); confidence =
    |2 * edge_score - 1|.
    """
    k = V.shape[0]
    if weights is None:
        weights = np.ones(k, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:                       # degenerate: fall back to uniform
        weights = np.ones(k, dtype=float)
        total = float(k)
    edge_score = np.tensordot(weights / total, V, axes=(0, 0))
    adj = (edge_score >= 0.5).astype(int)
    np.fill_diagonal(adj, 0)
    confidence = np.abs(2 * edge_score - 1)
    return adj, confidence, edge_score


def run_divvote_voting(nodes, medoids, prompt_style="reference",
                       model=DEFAULT_MODEL, use_cache=True, states=None,
                       vote_fn=None, weights=None):
    """Vote over all ordered pairs, then aggregate (uniform unless `weights`).

    Returns (adj, confidence, edge_score) matrices, all in `nodes` order. Thin
    wrapper over collect_votes + aggregate_votes. Pass vote_fn(cause, effect,
    medoid) -> 0/1 to inject a stand-in LLM (tests).
    """
    V = collect_votes(nodes, medoids, prompt_style=prompt_style, model=model,
                      use_cache=use_cache, states=states, vote_fn=vote_fn)
    return aggregate_votes(V, weights=weights)


def _resolve_states(benchmark_name, prompt_style, nodes, vote_fn):
    """Return the NODE_STATES dict for reference style (None otherwise).

    Raises if reference style is requested for a benchmark with no (or
    incomplete) state labels. vote_fn (tests) bypasses the requirement.
    """
    if prompt_style != "reference" or vote_fn is not None:
        return None
    states = baseline.NODE_STATES.get(benchmark_name)
    if states is None:
        raise ValueError(
            f"No variable-state labels for '{benchmark_name}' (reference "
            "style). Use prompt_style='simple' or add them to NODE_STATES."
        )
    missing = [nd for nd in nodes if nd not in states]
    if missing:
        raise ValueError(
            f"Missing state labels for nodes in '{benchmark_name}': {missing}"
        )
    return states


def medoid_reliability(validation="asia", prompt_style="reference",
                       model=DEFAULT_MODEL, use_cache=True,
                       medoids_path=MEDOIDS_PATH, vote_fn=None):
    """PLAN.md STEP 4: per-medoid accuracy r_j on the validation benchmark.

    For each medoid, r_j = fraction of off-diagonal ordered pairs where that
    medoid's lone vote matches the validation ground-truth adjacency. Returns
    (r, medoids) with r a length-k float array in [0, 1].
    """
    true_adj, nodes = baseline.load_ground_truth(validation)
    medoids = load_medoids(medoids_path)
    states = _resolve_states(validation, prompt_style, nodes, vote_fn)
    V = collect_votes(nodes, medoids, prompt_style=prompt_style, model=model,
                      use_cache=use_cache, states=states, vote_fn=vote_fn)
    n = len(nodes)
    mask = ~np.eye(n, dtype=bool)                 # score off-diagonal cells only
    truth = true_adj[mask]
    r = np.array([(V[j][mask] == truth).mean() for j in range(len(medoids))],
                 dtype=float)
    return r, medoids


def reliability_weights(validation="asia", gamma=1.0, prompt_style="reference",
                        model=DEFAULT_MODEL, use_cache=True,
                        medoids_path=MEDOIDS_PATH, vote_fn=None):
    """PLAN.md STEP 4: learned weights w_j = r_j^gamma / sum_i r_i^gamma.

    Returns (weights, r). gamma sharpens (>1) or flattens (<1) the reliability
    spread; gamma=1 is linear. Falls back to uniform if every r_j is 0.
    """
    r, _ = medoid_reliability(validation=validation, prompt_style=prompt_style,
                              model=model, use_cache=use_cache,
                              medoids_path=medoids_path, vote_fn=vote_fn)
    raw = r ** gamma
    total = raw.sum()
    if total <= 0:
        return np.ones_like(r) / len(r), r
    return raw / total, r


def run_divvote(benchmark_name, prompt_style="reference", model=DEFAULT_MODEL,
                use_cache=True, medoids_path=MEDOIDS_PATH, vote_fn=None,
                weights=None, verbose=True):
    """Load ground truth + medoids, run DivVote, score it, print and return.

    weights=None -> uniform; pass a length-k array (e.g. from
    reliability_weights) to aggregate with a learned weighting scheme.
    """
    true_adj, nodes = baseline.load_ground_truth(benchmark_name)
    medoids = load_medoids(medoids_path)

    states = _resolve_states(benchmark_name, prompt_style, nodes, vote_fn)

    if verbose:
        n_calls = len(medoids) * len(nodes) * (len(nodes) - 1)
        print(f"Running DivVote (uniform) on '{benchmark_name}' "
              f"({len(nodes)} nodes, k={len(medoids)} medoids, "
              f"prompt_style={prompt_style})")
        print(f"LLM calls (max, pre-cache): {n_calls}\n")

    adj, confidence, edge_score = run_divvote_voting(
        nodes, medoids, prompt_style=prompt_style, model=model,
        use_cache=use_cache, states=states, vote_fn=vote_fn, weights=weights)

    result = metrics.evaluate(adj, true_adj)
    result["benchmark"] = benchmark_name
    result["weighting"] = "uniform" if weights is None else "reliability"
    result["k"] = len(medoids)
    result["pred_edges"] = int(adj.sum())
    result["true_edges"] = int(true_adj.sum())
    result["pred_adj"] = adj
    result["confidence"] = confidence
    result["edge_score"] = edge_score

    if verbose:
        print(f"{'metric':<20}{'value':>10}")
        print("-" * 30)
        print(f"{'true edges':<20}{result['true_edges']:>10}")
        print(f"{'predicted edges':<20}{result['pred_edges']:>10}")
        print(f"{'SHD':<20}{result['SHD']:>10}")
        print(f"{'TENE':<20}{result['TENE']:>10}")
        print(f"{'TERE':<20}{result['TERE']:>10}")
        # Confidence of the predicted edges.
        conf_edges = confidence[adj == 1]
        if conf_edges.size:
            print(f"\nmean confidence on predicted edges: "
                  f"{conf_edges.mean():.3f} "
                  f"(min {conf_edges.min():.3f}, max {conf_edges.max():.3f})")

    return result


def compare_to_baseline(benchmark_name, prompt_style="reference",
                        model=DEFAULT_MODEL, use_cache=True, vote_fn=None):
    """Run DivVote and the 7-verb baseline (Raw); print a side-by-side table."""
    dv = run_divvote(benchmark_name, prompt_style=prompt_style, model=model,
                     use_cache=use_cache, vote_fn=vote_fn, verbose=False)
    base = baseline.run_baseline(benchmark_name, prompt_style=prompt_style,
                                 model=model, use_cache=use_cache,
                                 vote_fn=vote_fn, verbose=False)

    print(f"\n{benchmark_name}: DivVote (uniform, k={dv['k']}) vs 7-verb "
          f"baseline (Raw)")
    print(f"{'metric':<12}{'Baseline':>10}{'DivVote':>10}{'delta':>8}")
    print("-" * 40)
    for key in ["pred_edges", "SHD", "TENE", "TERE"]:
        label = "edges" if key == "pred_edges" else key
        d = dv[key] - base[key]
        print(f"{label:<12}{base[key]:>10}{dv[key]:>10}{d:>+8}")
    print(f"\nHEADLINE  deltaSHD = {dv['SHD'] - base['SHD']:+d} "
          f"(baseline {base['SHD']} -> DivVote {dv['SHD']}; "
          f"negative = improvement)")
    return {"divvote": dv, "baseline": base}


def compare_benchmarks(specs, model=DEFAULT_MODEL, use_cache=True):
    """Run DivVote and baseline-Raw per (benchmark, prompt_style); print a table.

    `specs` is a list of (benchmark_name, prompt_style). Single-run,
    uniform-weighting, Raw-vs-Raw.
    """
    rows = []
    for name, style in specs:
        dv = run_divvote(name, prompt_style=style, model=model,
                         use_cache=use_cache, verbose=False)
        base = baseline.run_baseline(name, prompt_style=style, model=model,
                                     use_cache=use_cache, verbose=False)
        rows.append((name, style, base["SHD"], dv["SHD"],
                     dv["SHD"] - base["SHD"]))

    print("\nDivVote (uniform, k=5) vs 7-verb baseline (Raw)  "
          "[single-run, Raw-vs-Raw]")
    print(f"{'benchmark':<20}{'style':<11}{'Base SHD':>9}{'DivVote':>9}"
          f"{'delta':>8}")
    print("-" * 57)
    for name, style, b_shd, d_shd, delta in rows:
        print(f"{name:<20}{style:<11}{b_shd:>9}{d_shd:>9}{delta:>+8}")
    return rows


def compare_weighting(specs, validation="asia", gamma=1.0, model=DEFAULT_MODEL,
                      use_cache=True, medoids_path=MEDOIDS_PATH):
    """Uniform vs learned-reliability DivVote on each spec, printed as a table.

    Learns weights on `validation` (PLAN.md STEP 4), then for each (benchmark,
    style) collects the medoid votes once and aggregates them both uniformly and
    with the learned weights, against the 7-verb Raw baseline. Columns: Base SHD,
    uniform SHD (+deltaSHD, +rel% vs base), reliability SHD (+deltaSHD, +rel%).
    """
    weights, r = reliability_weights(validation=validation, gamma=gamma,
                                     model=model, use_cache=use_cache,
                                     medoids_path=medoids_path)
    medoids = load_medoids(medoids_path)

    print(f"Learned reliability on '{validation}' (gamma={gamma})")
    print(f"{'medoid':<7}{'r_j':>8}{'w_j':>8}   prompt")
    print("-" * 70)
    for j, (rj, wj, m) in enumerate(zip(r, weights, medoids)):
        print(f"m{j:<6}{rj:>8.3f}{wj:>8.3f}   {m[:46]}")

    rows = []
    for name, style in specs:
        true_adj, nodes = baseline.load_ground_truth(name)
        states = _resolve_states(name, style, nodes, None)
        V = collect_votes(nodes, medoids, prompt_style=style, model=model,
                          use_cache=use_cache, states=states)
        adj_u, _, _ = aggregate_votes(V, weights=None)
        adj_w, _, _ = aggregate_votes(V, weights=weights)
        base = baseline.run_baseline(name, prompt_style=style, model=model,
                                     use_cache=use_cache, verbose=False)
        b = base["SHD"]
        su = metrics.shd(adj_u, true_adj)
        sw = metrics.shd(adj_w, true_adj)
        rel_u = (b - su) / b if b else 0.0
        rel_w = (b - sw) / b if b else 0.0
        rows.append((name, b, su, su - b, rel_u, sw, sw - b, rel_w))

    print("\nUniform vs reliability-weighted DivVote (k=5) vs 7-verb baseline "
          "(Raw)  [single-run]")
    print(f"{'benchmark':<20}{'Base':>6}"
          f"{'Unif':>6}{'dSHD':>6}{'rel%':>7}"
          f"{'Rel':>6}{'dSHD':>6}{'rel%':>7}")
    print("-" * 64)
    for name, b, su, du, relu, sw, dw, relw in rows:
        print(f"{name:<20}{b:>6}"
              f"{su:>6}{du:>+6}{relu*100:>6.0f}%"
              f"{sw:>6}{dw:>+6}{relw*100:>6.0f}%")
    print("\n(rel% = SHD reduction vs baseline; target >= 20%. Validation "
          f"benchmark '{validation}' excluded from the table.)")
    return {"weights": weights, "r": r, "rows": rows}


# Benchmarks scored for the weighting comparison (Asia is the validation set,
# so it is held out of this table).
WEIGHTING_SPECS = [
    ("cancer", "reference"),
    ("medical_diagnostic", "reference"),
    ("covid19", "reference"),
    ("endorisk", "reference"),
]


# Canonical config: reference style for all five benchmarks. Covid-19 and
# Endorisk now have NODE_STATES (Endorisk authoritative from Reijnen et al.
# 2020 Fig 2A; Covid-19 inferred binary present/absent), so the whole set runs
# under the same base-paper framing -- apples-to-apples with the baseline.
# (simple style remains available per-benchmark via --prompt-style for ablation.)
ALL_SPECS = [
    ("asia", "reference"),
    ("cancer", "reference"),
    ("medical_diagnostic", "reference"),
    ("covid19", "reference"),
    ("endorisk", "reference"),
]


def main():
    parser = argparse.ArgumentParser(description="DivVote voting (uniform).")
    parser.add_argument("benchmark", nargs="?",
                        help="e.g. asia, cancer, medical_diagnostic (omit with --all)")
    parser.add_argument("--prompt-style", choices=["reference", "simple"],
                        default="reference")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-compare", action="store_true",
                        help="only run DivVote, skip the baseline comparison")
    parser.add_argument("--all", action="store_true",
                        help="run all 5 benchmarks and print one SHD table")
    parser.add_argument("--compare-weighting", action="store_true",
                        help="uniform vs learned-reliability DivVote "
                             "(Asia validation) on the other 4 benchmarks")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="reliability sharpness for learned weights")
    args = parser.parse_args()
    if args.compare_weighting:
        compare_weighting(WEIGHTING_SPECS, validation="asia", gamma=args.gamma,
                          model=args.model, use_cache=not args.no_cache)
    elif args.all:
        compare_benchmarks(ALL_SPECS, model=args.model,
                           use_cache=not args.no_cache)
    elif args.no_compare:
        run_divvote(args.benchmark, prompt_style=args.prompt_style,
                    model=args.model, use_cache=not args.no_cache)
    else:
        compare_to_baseline(args.benchmark, prompt_style=args.prompt_style,
                            model=args.model, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
