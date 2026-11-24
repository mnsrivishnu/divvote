"""Sentence encoder wrapper.

Purpose
-------
Wrap a sentence encoder (default bge-m3; e5-large-v2 and all-MiniLM-L6-v2 are
ablation alternatives, PLAN.md section 7.4) to embed each prompt p_i into a
vector e_i (section 4.2, STEP 2). Embeddings are L2-normalized so a dot product
equals cosine similarity, which the prompt-pool deduplication (STEP 1) and the
diversity clustering (STEP 3) both rely on.

Maps to: Week 3, Day 2 (PLAN.md section 10) — "Embed pool with bge-m3." Also
used by Week 3, Day 1 deduplication.
"""

import numpy as np

DEFAULT_ENCODER = "BAAI/bge-m3"


class Encoder:
    """Thin wrapper over a SentenceTransformer model (loaded lazily)."""

    def __init__(self, model_name=DEFAULT_ENCODER):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, prompts):
        """Return an L2-normalized embedding matrix for a list of prompts."""
        model = self._load()
        return model.encode(
            list(prompts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


def cosine_similarity_matrix(embeddings):
    """Pairwise cosine similarity for L2-normalized embeddings (dot product)."""
    e = np.asarray(embeddings)
    return e @ e.T
