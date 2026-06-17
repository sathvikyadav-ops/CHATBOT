# =========================
# IMPORTS
# =========================
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from app.config import Config


# =========================
# EMBEDDING MODEL CLASS
# =========================
class EmbeddingModel:
    """
    Handles dense + sparse embeddings for hybrid search.
    """

    def __init__(self):
        # Dense embedding model
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Sparse BM25 model (keyword search)
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")

        # Dimension for dense vectors
        self.dimension = self.model.get_sentence_embedding_dimension()

    # =========================
    # DENSE EMBEDDING
    # =========================
    def encode(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()

    # =========================
    # BATCH DENSE EMBEDDING
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
    # SPARSE EMBEDDING (BM25)
    # =========================
    def encode_sparse(self, texts: list[str]):
        """
        Returns sparse vectors for Qdrant.
        """
        return list(self.sparse_model.embed(texts))

    def encode_sparse_single(self, text: str):
        return list(self.sparse_model.embed([text]))[0]