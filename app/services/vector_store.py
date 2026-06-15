# ==========================================================
# IMPORTS
# ==========================================================
import uuid
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue
)

from app.config import Config
from app.services.embeddings import EmbeddingModel


# ==========================================================
# VECTOR STORE
# ==========================================================
class VectorStore:

    def __init__(self):

        print("\n========== QDRANT INIT ==========")

        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=Config.QDRANT_TIMEOUT
        )

        self.collection_name = Config.QDRANT_COLLECTION
        self.embedding_model = EmbeddingModel()

        print(f"Collection: {self.collection_name}")
        print(f"Embedding Dimension: {self.embedding_model.dimension}")

        self._create_collection()

    # ======================================================
    # CREATE COLLECTION + INDEXES
    # ======================================================
    def _create_collection(self):

        collections = self.client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection_name not in existing:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_model.dimension,
                    distance=Distance.COSINE
                )
            )

            print(f"Created collection: {self.collection_name}")

        # ==================================================
        # CREATE PAYLOAD INDEXES (IMPORTANT FIX)
        # ==================================================
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_name",
                field_schema=PayloadSchemaType.KEYWORD
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )

            print("Payload indexes created (file_name, file_hash)")

        except Exception as e:
            print(f"[INDEX WARNING] {e}")

    # ======================================================
    # FILE HASH UTILITY
    # ======================================================
    def get_file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    # ======================================================
    # DUPLICATE CHECK
    # ======================================================
    def file_already_indexed(self, file_hash: str) -> bool:

        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_hash",
                            match=MatchValue(value=file_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )

            points = result[0]
            return len(points) > 0

        except Exception as e:
            print(f"[DUPLICATE CHECK ERROR] {e}")
            return False

    # ======================================================
    # ADD DOCUMENTS
    # ======================================================
    def add_documents(self, chunks: list[dict]):

        if not chunks:
            return

        print(f"Preparing {len(chunks)} chunks...")

        texts = [c["text"] for c in chunks]

        print("Generating embeddings...")

        embeddings = self.embedding_model.encode_batch(texts)

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=chunk
                )
            )

        batch_size = Config.VECTOR_BATCH_SIZE
        total_batches = (len(points) + batch_size - 1) // batch_size

        print(f"Uploading {len(points)} vectors in {total_batches} batches")

        for i in range(0, len(points), batch_size):

            batch = points[i:i + batch_size]
            batch_number = (i // batch_size) + 1

            print(f"Uploading batch {batch_number}/{total_batches}")

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )

        print(f"Inserted {len(points)} chunks")

    # ======================================================
    # SEARCH
    # ======================================================
    def search(self, query: str, top_k: int = None):

        if top_k is None:
            top_k = Config.TOP_K

        query_embedding = self.embedding_model.encode(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )

        return response.points

    # ======================================================
    # COLLECTION INFO
    # ======================================================
    def collection_info(self):

        return self.client.get_collection(self.collection_name)

    # ======================================================
    # DELETE COLLECTION
    # ======================================================
    def delete_collection(self):

        self.client.delete_collection(
            collection_name=self.collection_name
        )

        print(f"Deleted collection: {self.collection_name}")

    # ======================================================
    # COUNT
    # ======================================================
    def count(self):

        info = self.client.get_collection(self.collection_name)
        return info.points_count