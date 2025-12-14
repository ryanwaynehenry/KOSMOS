# umls_faiss_lookup.py

from typing import List, Dict, Any
from time import perf_counter
import json
import numpy as np
import faiss

from clinical_kg.umls.sapbert_embedder import SapBERTEmbedder


class UmlsFaissSearcher:
    def __init__(
        self,
        index_path: str = "UMLS_sapbert.faiss",
        mapping_path: str = "UMLS_sapbert_mapping.json",
    ):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load mapping
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.cuis = mapping["cuis"]
        self.terms = mapping["terms"]
        self.sources = mapping["sources"]

        # Sanity check
        assert len(self.cuis) == self.index.ntotal
        assert len(self.terms) == self.index.ntotal
        assert len(self.sources) == self.index.ntotal

        # Embedder
        self.embedder = SapBERTEmbedder()

    def search(
        self,
        mention: str,
        source: str = None,
        top_k: int = 10,
        min_score: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Search for the nearest UMLS terms to a mention string, optionally
        restricted to a specific source vocabulary (SAB).

        Returns a list of dicts:
        [
          {
            "cui": ...,
            "term": ...,
            "source": ...,  # SAB/source vocabulary name (e.g., SNOMEDCT_US)
            "score": float,  # cosine similarity
          },
          ...
        ]
        """
        emb = self.embedder.encode([mention])  # shape (1, d)
        emb = emb.astype(np.float32)

        # If a source filter is supplied, search a bit deeper to allow filtering
        # without losing all results.
        search_k = top_k if source is None else max(top_k * 5, top_k)

        scores, idxs = self.index.search(emb, search_k)
        scores = scores[0]
        idxs = idxs[0]

        results = []
        for score, idx in zip(scores, idxs):
            if score < min_score:
                continue
            if source is not None and self.sources[int(idx)] != source:
                continue
            results.append(
                {
                    "cui": self.cuis[int(idx)],
                    "term": self.terms[int(idx)],
                    "source": self.sources[int(idx)],
                    "score": float(score),
                }
            )
            if len(results) >= top_k:
                break
        return results


if __name__ == "__main__":
    t0 = perf_counter()
    searcher = UmlsFaissSearcher(
        index_path="UMLS_sapbert.faiss",
        mapping_path="UMLS_sapbert_mapping.json",
    )
    init_time = perf_counter() - t0

    queries = [
        {"mention": "peanut allergy", "source": "SNOMEDCT_US"},
        {"mention": "drinks socially", "source": "SNOMEDCT_US"},
        {"mention": "appetite change", "source": "SNOMEDCT_US"},
        {"mention": "shortness of breath", "source": "SNOMEDCT_US"},
        {"mention": "low mood", "source": "SNOMEDCT_US"},
        # LOINC and RXNORM examples (imperfect phrasing)
        {"mention": "oxygen saturation level", "source": "LNC"},
        {"mention": "tylenol extra strength pills", "source": "RXNORM"},
    ]

    print(f"Initialized searcher in {init_time:.3f}s")
    for q in queries:
        print(f"\nQuery: {q['mention']} [{q['source']}]")
        t_query = perf_counter()
        hits = searcher.search(
            q["mention"], source=q["source"], top_k=5, min_score=0.4
        )
        elapsed = perf_counter() - t_query
        for h in hits:
            print(
                f"  {h['score']:.3f}  {h['cui']}  [{h['source']}]  {h['term']}"
            )
        print(f"Search time: {elapsed:.3f}s")
