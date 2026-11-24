"""Load the benchmark ground-truth DAGs from the base-paper reference repo.

Purpose
-------
Read each benchmark's ground-truth directed adjacency matrix from the base
paper's own repository (cloned into reference/, from
https://github.com/ExcellentDarkTea/LLM-Causal-Discovery) and save it as a
.npy file in data/benchmarks/. Prints node and edge counts per network so they
can be checked against the known ground-truth sizes.

Adjacency convention: adj[i, j] == 1 means a directed edge node_i -> node_j
(same orientation as the source CSVs: rows = cause, columns = effect).
A companion <name>_nodes.txt records the node order for each matrix.

Maps to: Week 1, Days 6-7 (PLAN.md section 10) — "Download 5 benchmarks ...;
parse to .npy."

Availability note
-----------------
The base-paper repo ships ground-truth adjacency CSVs for only THREE of the
five benchmarks: Asia, Cancer, and Medical Diagnostic (data/adj_matrix_med.csv).
It contains no Covid-19 or Endorisk network, so those two are left with
source=None. Drop a CSV (same format: an index column of node names, then a
0/1 matrix) into reference/data/ or elsewhere and set its `source` to load it.

Run from the repo root:
    python -m src.data.load_benchmarks
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Repo-root-relative paths: .../divvote
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "benchmarks"
REF_DATA = REPO_ROOT / "reference" / "data"


# For each benchmark: `source` is a path to a ground-truth adjacency CSV
# (first column = node-name index, remaining columns = 0/1 adjacency), or None
# if the network is not available. `expected` is (nodes, edges) for a check.
BENCHMARKS = {
    "asia": {
        "source": REF_DATA / "adj_matrix_asia.csv",
        "expected_nodes": 8,
        "expected_edges": 8,
    },
    "cancer": {
        "source": REF_DATA / "adj_matrix_cancer.csv",
        "expected_nodes": 5,
        "expected_edges": 4,
    },
    "medical_diagnostic": {
        "source": REF_DATA / "adj_matrix_med.csv",
        "expected_nodes": 10,
        "expected_edges": 9,
    },
    "covid19": {
        # Supplied separately (not in the base-paper repo); from Mascaro
        # et al. 2023 per CLAUDE.md.
        "source": OUT_DIR / "adj_matrix_covid.csv",
        "expected_nodes": 11,
        "expected_edges": 20,
    },
    "endorisk": {
        # Supplied separately (not in the base-paper repo); from Reijnen
        # et al. 2020 per CLAUDE.md.
        "source": OUT_DIR / "adj_matrix_endorisk.csv",
        "expected_nodes": 18,
        "expected_edges": 37,
    },
}


def load_dag(source):
    """Read an adjacency CSV and return (adjacency int matrix, node list).

    adj[i, j] == 1 means a directed edge node_i -> node_j.
    """
    df = pd.read_csv(source, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"adjacency matrix is not square: {df.shape[0]}x{df.shape[1]}"
        )
    nodes = [str(c) for c in df.columns]
    adj = df.to_numpy().astype(int)
    return adj, nodes


def save_benchmark(name, adj, nodes):
    """Save the adjacency matrix (.npy) and node order (.txt)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{name}.npy", adj)
    (OUT_DIR / f"{name}_nodes.txt").write_text(
        "\n".join(nodes), encoding="utf-8"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving ground-truth adjacency matrices to: {OUT_DIR}\n")
    print(f"{'benchmark':<20}{'nodes':>8}{'edges':>8}   expected   status")
    print("-" * 64)

    for name, cfg in BENCHMARKS.items():
        source = cfg["source"]
        exp_n, exp_e = cfg["expected_nodes"], cfg["expected_edges"]
        exp = f"{exp_n}/{exp_e}"

        if source is None:
            print(f"{name:<20}{'--':>8}{'--':>8}   {exp:<10} SKIPPED (no source)")
            continue

        try:
            adj, nodes = load_dag(source)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{name:<20}{'--':>8}{'--':>8}   {exp:<10} FAILED: {exc}")
            continue

        n_nodes, n_edges = len(nodes), int(adj.sum())
        save_benchmark(name, adj, nodes)
        match = "ok" if (n_nodes, n_edges) == (exp_n, exp_e) else "MISMATCH"
        print(f"{name:<20}{n_nodes:>8}{n_edges:>8}   {exp:<10} {match}")

    print(
        "\nNote: Covid-19 and Endorisk are not in the base-paper repo; their "
        "adjacency\nCSVs were supplied separately in data/benchmarks/."
    )


if __name__ == "__main__":
    main()
