# umls_faiss_lookup.py

from typing import List, Dict, Any
import json
import numpy as np
import faiss

from sapbert_embedder import SapBERTEmbedder


class UmlsFaissSearcher:
    def __init__(
        self,
        index_path: str = "umls_sapbert.faiss",
        mapping_path: str = "umls_sapbert_mapping.json",
    ):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load mapping
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.cuis = mapping["cuis"]
        self.terms = mapping["terms"]

        # Sanity check
        assert len(self.cuis) == self.index.ntotal
        assert len(self.terms) == self.index.ntotal

        # Embedder
        self.embedder = SapBERTEmbedder()

    def search(
        self,
        mention: str,
        top_k: int = 10,
        min_score: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Search for the nearest UMLS terms to a mention string.

        Returns a list of dicts:
        [
          {
            "cui": ...,
            "term": ...,
            "score": float,  # cosine similarity
          },
          ...
        ]
        """
        emb = self.embedder.encode([mention])  # shape (1, d)
        emb = emb.astype(np.float32)

        scores, idxs = self.index.search(emb, top_k)
        scores = scores[0]
        idxs = idxs[0]

        results = []
        for score, idx in zip(scores, idxs):
            if score < min_score:
                continue
            results.append(
                {
                    "cui": self.cuis[int(idx)],
                    "term": self.terms[int(idx)],
                    "score": float(score),
                }
            )
        return results


if __name__ == "__main__":
    searcher = UmlsFaissSearcher(
        index_path="umls_sapbert.faiss",
        mapping_path="umls_sapbert_mapping.json",
    )

    queries = [
        "drinks socially",
        "appetite change",
        "shortness of breath",
        "low mood",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        hits = searcher.search(q, top_k=5, min_score=0.4)
        for h in hits:
            print(f"  {h['score']:.3f}  {h['cui']}  {h['term']}")
