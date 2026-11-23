"""Prompt pool generation + templates.

Purpose
-------
Generate the candidate prompt pool P = {p_1 .. p_N} (PLAN.md section 4.2,
STEP 1) by seeding an LLM generator with the 6 causal-framing categories
(direct, risk, counterfactual, interventional, mechanistic, temporal; see
section 4.3), asking for ~4 variants per category, and deduplicating any pair
with cosine similarity > 0.95.

Maps to: Week 3, Day 1 (PLAN.md section 10) — "Generate prompt pool (~25
variants) using LLM + seed categories."

This is a stub: structure only, no full generation logic yet.
"""

# 6 seed categories with one example template each (PLAN.md section 4.3).
SEED_CATEGORIES = {
    "direct": "Does {X} cause {Y}?",
    "risk": "Does {X} increase the probability of {Y}?",
    "counterfactual": "Would {Y} occur without {X}?",
    "interventional": "If we intervene on {X}, does {Y} change?",
    "mechanistic": "Is {Y} mechanistically downstream of {X}?",
    "temporal": "Does {X} temporally precede {Y}?",
}


def generate_pool(n=25, variants_per_category=4, seed_categories=None):
    """Generate the candidate prompt pool via the generator LLM. Stub."""
    raise NotImplementedError


def deduplicate(prompts, encoder, threshold=0.95):
    """Drop any prompt pair with cosine similarity above `threshold`. Stub."""
    raise NotImplementedError


def format_prompt(template, x, y):
    """Fill a prompt template's {X}/{Y} placeholders for a variable pair. Stub."""
    raise NotImplementedError
