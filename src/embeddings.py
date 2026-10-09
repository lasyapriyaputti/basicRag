import numpy as np
from typing import List
from sentence_transformers import (
    SentenceTransformer
)


class EmbeddingManager:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model_name = model_name
        self.model = None

        self._load_model()

    def _load_model(self):

        print(
            f"\nLoading embedding model: "
            f"{self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        print(
            "Model loaded successfully."
        )

        print(
            "Embedding dimension:",
            self.model.get_sentence_embedding_dimension()
        )

    def generate_embeddings(
        self,
        texts: List[str]
    ) -> np.ndarray:

        if self.model is None:

            raise ValueError(
                "Model not loaded"
            )

        print(
            f"\nGenerating embeddings "
            f"for {len(texts)} texts..."
        )

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        print(
            "Embedding shape:",
            embeddings.shape
        )

        return embeddings

    def get_embedding_dimension(self):

        return (
            self.model
            .get_sentence_embedding_dimension()
        )