"""Prompt pool generation + templates.

Purpose
-------
Generate the candidate prompt pool P = {p_1 .. p_N} (PLAN.md section 4.2,
STEP 1): seed an LLM generator with the 6 causal-framing categories (direct,
risk, counterfactual, interventional, mechanistic, temporal; section 4.3), ask
for ~4 variant templates per category (~24-25 total), then deduplicate by
dropping any prompt whose cosine similarity to an already-kept prompt exceeds
0.95 (computed in sentence-embedding space, section 4.2).

Each template is a yes/no causal question containing the placeholders {X} and
{Y}, filled per variable pair at voting time.

Maps to: Week 3, Day 1 (PLAN.md section 10) — "Generate prompt pool (~25
variants) using LLM + seed categories."

Run from the repo root, e.g.:
    python -m src.prompts.pool
    python -m src.prompts.pool --encoder sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
from datetime import date
from pathlib import Path

from src.embedding.encoder import Encoder, cosine_similarity_matrix
from src.llm import interface

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "prompts"
DEFAULT_POOL_PATH = OUT_DIR / "pool.json"

# 6 seed categories with one example template each (PLAN.md section 4.3).
SEED_CATEGORIES = {
    "direct": "Does {X} cause {Y}?",
    "risk": "Does {X} increase the probability of {Y}?",
    "counterfactual": "Would {Y} occur without {X}?",
    "interventional": "If we intervene on {X}, does {Y} change?",
    "mechanistic": "Is {Y} mechanistically downstream of {X}?",
    "temporal": "Does {X} temporally precede {Y}?",
}

VARIANTS_PER_CATEGORY = 4
DEDUP_THRESHOLD = 0.95

_GEN_SYSTEM = (
    "You generate diverse causal-question prompt templates for probing whether "
    "one variable causes another. Every template MUST contain the placeholders "
    "{X} and {Y}, each exactly once, and must read as a yes/no question. Return "
    "ONLY the templates, one per line, with no numbering, bullets, or quotes."
)


def _gen_user(category, example, n):
    return (
        f"Causal-framing category: '{category}'.\n"
        f"Example in this framing: \"{example}\"\n\n"
        f"Generate exactly {n} distinct paraphrased templates that keep the "
        f"'{category}' framing but vary the wording. Use the placeholders "
        f"{{X}} and {{Y}}."
    )


def _parse_templates(text):
    """Extract valid {X}/{Y} templates from a newline-separated completion."""
    out = []
    for line in text.splitlines():
        s = line.strip().lstrip("-*0123456789.) ").strip().strip('"').strip()
        if "{X}" in s and "{Y}" in s:
            out.append(s)
    return out


def generate_variants(category, example, n=VARIANTS_PER_CATEGORY,
                      model=interface.DEFAULT_MODEL, use_cache=True):
    """Ask the generator LLM for `n` template variants in one framing."""
    text = interface.llm_generate(
        _gen_user(category, example, n), system=_GEN_SYSTEM, model=model,
        use_cache=use_cache)
    return _parse_templates(text)[:n]


def generate_pool(n_per_category=VARIANTS_PER_CATEGORY,
                  model=interface.DEFAULT_MODEL, use_cache=True,
                  include_seeds=False):
    """Generate the full candidate pool as a list of {category, text} dicts.

    Per PLAN.md section 4.2 the pool is the ~4 generated variants per category
    (~24-25 total). Set include_seeds=True to also add each category's canonical
    seed exemplar to the pool.
    """
    items = []
    seen = set()
    for category, example in SEED_CATEGORIES.items():
        texts = []
        if include_seeds:
            texts.append(example)
        texts.extend(generate_variants(category, example, n_per_category,
                                       model=model, use_cache=use_cache))
        for t in texts:
            if t not in seen:          # exact-duplicate guard before embedding
                seen.add(t)
                items.append({"category": category, "text": t})
    return items


def deduplicate(items, encoder, threshold=DEDUP_THRESHOLD):
    """Greedy dedup: drop any item whose cosine to a kept item exceeds threshold.

    Returns (kept_items, dropped) where each dropped entry records the item, the
    kept item it collided with, and their cosine similarity.
    """
    texts = [it["text"] for it in items]
    sim = cosine_similarity_matrix(encoder.encode(texts))

    kept_idx, kept_items, dropped = [], [], []
    for i in range(len(items)):
        collision = None
        for j in kept_idx:
            if sim[i, j] > threshold:
                collision = (j, float(sim[i, j]))
                break
        if collision is None:
            kept_idx.append(i)
            kept_items.append(items[i])
        else:
            j, score = collision
            dropped.append({
                "text": items[i]["text"], "category": items[i]["category"],
                "duplicate_of": items[j]["text"], "cosine": round(score, 4)})
    return kept_items, dropped


def save_pool(kept, dropped, meta, path=DEFAULT_POOL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": meta,
               "prompts": [it["text"] for it in kept],
               "prompts_with_category": kept,
               "dropped": dropped}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_pool(n_per_category=VARIANTS_PER_CATEGORY,
               model=interface.DEFAULT_MODEL, encoder_name=Encoder().model_name,
               threshold=DEDUP_THRESHOLD, use_cache=True, path=DEFAULT_POOL_PATH,
               verbose=True):
    """Generate, deduplicate, and save the prompt pool; return the result dict."""
    items = generate_pool(n_per_category, model=model, use_cache=use_cache)
    encoder = Encoder(encoder_name)
    kept, dropped = deduplicate(items, encoder, threshold=threshold)

    meta = {"date": str(date.today()), "generator_model": model,
            "encoder": encoder_name, "dedup_threshold": threshold,
            "n_per_category": n_per_category,
            "n_generated": len(items), "n_kept": len(kept),
            "n_dropped": len(dropped)}
    saved = save_pool(kept, dropped, meta, path=path)

    if verbose:
        print(f"Prompt pool ({meta['n_generated']} generated -> "
              f"{meta['n_kept']} kept, {meta['n_dropped']} dropped at "
              f"cosine > {threshold})")
        print(f"generator={model}  encoder={encoder_name}\n")
        by_cat = {}
        for it in kept:
            by_cat.setdefault(it["category"], []).append(it["text"])
        for cat in SEED_CATEGORIES:
            print(f"[{cat}]")
            for t in by_cat.get(cat, []):
                print(f"  - {t}")
        if dropped:
            print("\nDropped (near-duplicate):")
            for d in dropped:
                print(f"  - ({d['cosine']}) {d['text']}  ~=  {d['duplicate_of']}")
        print(f"\nSaved to {saved}")
    return {"kept": kept, "dropped": dropped, "metadata": meta, "path": str(saved)}


def main():
    parser = argparse.ArgumentParser(description="Generate the DivVote prompt pool.")
    parser.add_argument("--n-per-category", type=int, default=VARIANTS_PER_CATEGORY)
    parser.add_argument("--model", default=interface.DEFAULT_MODEL)
    parser.add_argument("--encoder", default=Encoder().model_name,
                        help="sentence-transformers model for dedup")
    parser.add_argument("--threshold", type=float, default=DEDUP_THRESHOLD)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    build_pool(n_per_category=args.n_per_category, model=args.model,
               encoder_name=args.encoder, threshold=args.threshold,
               use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
