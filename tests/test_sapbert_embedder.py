"""
Basic sanity checks for SapBERTEmbedder.

These tests do not assert on specific embedding values (since embeddings can
vary across model versions) but verify shape, dtype, and that similar terms
produce comparable embeddings.
"""

import numpy as np

from clinical_kg.umls.sapbert_embedder import SapBERTEmbedder


def test_encode_shapes_and_dtype():
    terms = ["aspirin", "lisinopril"]
    embedder = SapBERTEmbedder()
    embeddings = embedder.encode(terms, batch_size=2)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(terms)
    assert embeddings.ndim == 2
    assert embeddings.dtype == np.float32


def test_similar_terms_more_similar_than_unrelated():
    terms = ["aspirin", "aspirin 81 mg", "basketball"]
    embedder = SapBERTEmbedder()
    embeddings = embedder.encode(terms, batch_size=3)

    asp = embeddings[0]
    asp81 = embeddings[1]
    ball = embeddings[2]

    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_close = cos_sim(asp, asp81)
    sim_far = cos_sim(asp, ball)

    assert sim_close > sim_far


if __name__ == "__main__":
    test_encode_shapes_and_dtype()
    test_similar_terms_more_similar_than_unrelated()
    print("SapBERTEmbedder tests passed.")
