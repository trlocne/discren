"""
Sentence encoder that turns LLM-generated profiles into fixed-size embeddings.

Uses ``sentence-transformers``. The default model is
``sentence-transformers/stsb-roberta-large``, matching
``llm_augment/config.yaml`` and the ``--encoder`` defaults of both build
scripts. It outputs 1024-dim vectors — the SAME dimensionality as the existing
``text_feat.npy`` (23033, 1024), so item and user semantic spaces are
comparable. ``BAAI/bge-large-en-v1.5`` is also 1024-dim and can be swapped in;
``all-MiniLM-L6-v2`` (384-dim) is lighter but changes the feature width, which
the model reads from the array at load time.
"""

from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_ENCODER = "sentence-transformers/stsb-roberta-large"


class SentenceEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        batch_size: int = 64,
        normalize: bool = True,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return emb.astype(np.float32)


class HashingEmbedder:
    """Dependency-free fallback encoder (bag-of-hashed-words, L2-normalised).

    NOT for production — only lets you exercise the full pipeline (alignment,
    file shapes, saving) without downloading a sentence-transformer.

    Bucketing uses BLAKE2b rather than the built-in :func:`hash`. Python salts
    string hashing with a per-process random seed (unless ``PYTHONHASHSEED`` is
    pinned before interpreter start), so the previous implementation mapped the
    same word to a different bucket on every run: identical input text produced
    different embeddings, and nothing about the artifact revealed it. A stable
    digest makes the fallback genuinely reproducible.
    """

    def __init__(self, dim: int = 1024, **_ignored):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, self._bucket(tok)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


def build_embedder(model_name: str, **kwargs):
    """Return a real sentence encoder, or the hashing fallback if
    ``model_name == 'hashing'``."""
    if model_name == "hashing":
        return HashingEmbedder(**kwargs)
    return SentenceEmbedder(model_name=model_name, **kwargs)
