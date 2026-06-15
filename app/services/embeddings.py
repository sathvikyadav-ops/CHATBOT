# =========================
# IMPORTS
# =========================
from sentence_transformers import SentenceTransformer
from app.config import Config


# =========================
# EMBEDDING MODEL CLASS
# =========================
class EmbeddingModel:
    """
    Handles embedding generation using Sentence Transformers.
    """

    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Store embedding dimension (useful for vector DB like Qdrant)
        self.dimension = self.model.get_sentence_embedding_dimension()

    # =========================
    # SINGLE TEXT EMBEDDING
    # =========================
    def encode(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding.tolist()

    # =========================
    # BATCH EMBEDDING
    # =========================
    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False
        )

        return embeddings.tolist()

    # =========================
    # COSINE SIMILARITY
    # =========================
    def similarity(self, text1: str, text2: str) -> float:
        emb1 = self.model.encode(
            text1,
            normalize_embeddings=True
        )

        emb2 = self.model.encode(
            text2,
            normalize_embeddings=True
        )

        return float(emb1 @ emb2)