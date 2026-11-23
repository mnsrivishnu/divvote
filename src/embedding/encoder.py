"""Sentence encoder wrapper.

Purpose
-------
Wrap a sentence encoder (default bge-m3; e5-large-v2 and all-MiniLM-L6-v2 are
ablation alternatives, PLAN.md section 7.4) to embed each prompt p_i into a
vector e_i (section 4.2, STEP 2). These embeddings feed the diversity
clustering step.

Maps to: Week 3, Day 2 (PLAN.md section 10) — "Embed pool with bge-m3."

This is a stub: structure only, no model loading or inference logic yet.
"""

DEFAULT_ENCODER = "BAAI/bge-m3"


class Encoder:
    """Thin wrapper over a SentenceTransformer model. Stub."""

    def __init__(self, model_name=DEFAULT_ENCODER):
        self.model_name = model_name
        self._model = None

    def encode(self, prompts):
        """Return the embedding matrix E for a list of prompts. Stub."""
        raise NotImplementedError
