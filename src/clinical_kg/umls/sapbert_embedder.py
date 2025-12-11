# sapbert_embedder.py

from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class SapBERTEmbedder:
    def __init__(self, device: str = None):
        """
        Simple wrapper around SapBERT to get embeddings for short entity strings.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of short phrases into L2-normalized embeddings.
        Returns a NumPy array of shape (len(texts), hidden_dim).
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=25,  # SapBERT is trained on short entity names
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            # Use CLS token from last hidden state (SapBERT paper’s approach)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)

            # Move to CPU, convert to NumPy
            cls_embeddings = cls_embeddings.cpu().numpy()

            # L2 normalize for cosine similarity via inner product
            norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True) + 1e-12
            cls_embeddings = cls_embeddings / norms

            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)
