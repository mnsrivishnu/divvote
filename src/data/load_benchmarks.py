"""Download the 5 benchmark Bayesian networks and save their ground-truth DAGs.

Purpose
-------
Fetch the five medical Bayesian-network benchmarks used in the base paper
(PLAN.md sections 2.4 / 7.1) via the `bnlearn` Python package, extract each
network's ground-truth directed adjacency matrix, and save it as a .npy file
in data/benchmarks/. Prints node and edge counts per network so they can be
checked against PLAN.md section 2.4.

Adjacency convention: adj[i, j] == 1 means a directed edge node_i -> node_j.
A companion <name>_nodes.txt records the node order for each matrix.

Maps to: Week 1, Days 6-7 (PLAN.md section 10) — "Download 5 benchmarks from
bnlearn repository; parse to .npy."

Availability note
-----------------
Only Asia and Cancer are actually hosted on bnlearn.com/bnrepository. The
other three networks named in the plan (Medical Diagnostic, Covid-19,
Endorisk) are NOT in that repository under any standard slug (all return 404),
so they cannot be downloaded automatically here. Each is left with
source=None; set its `source` to a .bif URL or a local .bif path once the file
is obtained, and the loader will handle it the same way.

Run from the repo root:
    python -m src.data.load_benchmarks
"""

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np

try:
    import bnlearn as bn
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "The 'bnlearn' package is required. Install with: pip install bnlearn"
    ) from exc


# Repo-root-relative paths: .../divvote/data/benchmarks
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "benchmarks"
BIF_CACHE = OUT_DIR / "_bif_cache"


# For each benchmark: a `source` and the expected (nodes, edges) from PLAN.md
# section 2.4 (values prefixed "~" in the plan are approximate).
# `source` may be:
#   - a .bif or .bif.gz URL (downloaded + decompressed + parsed),
#   - a local .bif path,
#   - a bnlearn built-in short name (e.g. "asia"),
#   - or None if the network is not yet available.
BENCHMARKS = {
    "asia": {
        "source": "https://www.bnlearn.com/bnrepository/asia/asia.bif.gz",
        "expected_nodes": 8,
        "expected_edges": 8,
    },
    "cancer": {
        "source": "https://www.bnlearn.com/bnrepository/cancer/cancer.bif.gz",
        "expected_nodes": 5,
        "expected_edges": 4,
    },
    "medical_diagnostic": {
        # Not on bnrepository (404). Set to a .bif URL or local path to enable.
        "source": None,
        "expected_nodes": 18,  # ~18
        "expected_edges": 25,  # ~25
    },
    "covid19": {
        "source": None,  # not on bnrepository (404) -- see availability note
        "expected_nodes": 12,  # ~12
        "expected_edges": 18,  # ~18
    },
    "endorisk": {
        "source": None,  # not on bnrepository (404) -- see availability note
        "expected_nodes": 20,  # ~20
        "expected_edges": 25,  # ~25
    },
}


def _resolve_bif(name, source):
    """Return a bnlearn-loadable reference for `source`.

    URLs are downloaded into the .bif cache (and gunzipped if needed) and the
    local path is returned. Local paths and built-in short names pass through.
    """
    if source.startswith(("http://", "https://")):
        BIF_CACHE.mkdir(parents=True, exist_ok=True)
        if source.endswith(".gz"):
            gz_path = BIF_CACHE / f"{name}.bif.gz"
            bif_path = BIF_CACHE / f"{name}.bif"
            urllib.request.urlretrieve(source, gz_path)
            with gzip.open(gz_path, "rb") as fin, open(bif_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            return str(bif_path)
        bif_path = BIF_CACHE / f"{name}.bif"
        urllib.request.urlretrieve(source, bif_path)
        return str(bif_path)
    # Local .bif path or bnlearn built-in short name.
    return source


def load_dag(name, source):
    """Import a DAG via bnlearn and return (adjacency int matrix, node list).

    adj[i, j] == 1 means a directed edge node_i -> node_j.
    """
    ref = _resolve_bif(name, source)
    model = bn.import_DAG(ref, CPD=False, checkmodel=False)
    adjmat = model.get("adjmat")
    if adjmat is None:
        raise ValueError(
            f"bnlearn returned no adjacency matrix for source {source!r} "
            "(unknown built-in name or unparseable file)"
        )
    nodes = list(adjmat.columns)  # rows=source, cols=target
    adj = adjmat.to_numpy().astype(int)
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
        exp = f"~{cfg['expected_nodes']}/{cfg['expected_edges']}"

        if source is None:
            print(f"{name:<20}{'--':>8}{'--':>8}   {exp:<10} SKIPPED (no source)")
            continue

        try:
            adj, nodes = load_dag(name, source)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{name:<20}{'--':>8}{'--':>8}   {exp:<10} FAILED: {exc}")
            continue

        n_nodes, n_edges = len(nodes), int(adj.sum())
        save_benchmark(name, adj, nodes)
        print(f"{name:<20}{n_nodes:>8}{n_edges:>8}   {exp:<10} ok")

    print(
        "\nNote: medical_diagnostic, covid19 and endorisk are not on "
        "bnlearn.com/bnrepository.\nObtain each .bif and set its 'source' in "
        "BENCHMARKS (a URL or local path) to load it."
    )


if __name__ == "__main__":
    main()
