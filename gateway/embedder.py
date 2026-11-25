"""Text embeddings. An embedding turns text into a vector so that texts with similar
meaning point in similar directions. Vectors are L2-normalized (length 1), so cosine
similarity is just the dot product."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        # CPU keeps results the same on every machine; one short text takes ~10 ms.
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-length float32 vector for the text.
        No BGE query-instruction prefix: we compare a question with other questions
        (symmetric), not a short query with long documents."""
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)
