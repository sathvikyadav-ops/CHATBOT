# ==========================================================
# IMPORTS
# ==========================================================
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    Fusion,
    FusionQuery
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

        self.collection_name = (
            Config.QDRANT_COLLECTION
        )

        self.embedding_model = (
            EmbeddingModel()
        )

        print(
            f"Collection: "
            f"{self.collection_name}"
        )

        print(
            f"Embedding Dimension: "
            f"{self.embedding_model.dimension}"
        )

        self._create_collection()

    # ======================================================
    # VALIDATE HYBRID SCHEMA
    # ======================================================
    def _is_hybrid_schema_valid(
        self
    ) -> bool:

        try:

            info = (
                self.client.get_collection(
                    self.collection_name
                )
            )

            print(
                "\n========== COLLECTION INFO =========="
            )

            print(info)

            vectors = (
                info.config.params.vectors
            )

            sparse_vectors = (
                info.config.params.sparse_vectors
            )

            dense_ok = (
                isinstance(
                    vectors,
                    dict
                )
                and
                "dense" in vectors
            )

            sparse_ok = (
                sparse_vectors is not None
                and
                "bm25" in sparse_vectors
            )

            print(
                f"Dense Vector Exists: "
                f"{dense_ok}"
            )

            print(
                f"Sparse Vector Exists: "
                f"{sparse_ok}"
            )

            return (
                dense_ok
                and
                sparse_ok
            )

        except Exception as e:

            print(
                f"[SCHEMA CHECK ERROR] {e}"
            )

            return False

    # ======================================================
    # CREATE COLLECTION
    # ======================================================
    def _create_collection(self):

        collections = (
            self.client.get_collections()
        )

        existing = [
            c.name
            for c in collections.collections
        ]

        print(
            f"Existing Collections: "
            f"{existing}"
        )

        # --------------------------------------------------
        # DELETE OLD SCHEMA
        # --------------------------------------------------
        if self.collection_name in existing:

            print(
                f"{self.collection_name} "
                f"already exists."
            )

            if not self._is_hybrid_schema_valid():

                print(
                    "Old schema detected."
                )

                print(
                    "Deleting collection..."
                )

                self.client.delete_collection(
                    self.collection_name
                )

                existing.remove(
                    self.collection_name
                )

                print(
                    "Collection deleted."
                )

        # --------------------------------------------------
        # CREATE NEW COLLECTION
        # --------------------------------------------------
        if self.collection_name not in existing:

            print(
                "Creating Hybrid Collection..."
            )

            self.client.create_collection(
                collection_name=self.collection_name,

                vectors_config={
                    "dense":
                    VectorParams(
                        size=self.embedding_model.dimension,
                        distance=Distance.COSINE
                    )
                },

                sparse_vectors_config={
                    "bm25":
                    SparseVectorParams()
                }
            )

            print(
                f"Created Collection: "
                f"{self.collection_name}"
            )

        # --------------------------------------------------
        # PAYLOAD INDEXES
        # --------------------------------------------------
        try:

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_name",
                field_schema=PayloadSchemaType.KEYWORD
            )

            print(
                "Payload indexes created."
            )

        except Exception as e:

            print(
                f"[INDEX WARNING] {e}"
            )

    # ======================================================
    # DUPLICATE DETECTION
    # ======================================================
    def file_already_indexed(
        self,
        file_hash: str
    ) -> bool:

        try:

            points, _ = (
                self.client.scroll(
                    collection_name=self.collection_name,

                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_hash",
                                match=MatchValue(
                                    value=file_hash
                                )
                            )
                        ]
                    ),

                    limit=1
                )
            )

            found = (
                len(points) > 0
            )

            print(
                f"Duplicate Check: "
                f"{found}"
            )

            return found

        except Exception as e:

            print(
                f"[DUPLICATE CHECK ERROR] {e}"
            )

            return False

    # ======================================================
    # ADD DOCUMENTS
    # ======================================================
    def add_documents(
        self,
        chunks: list[dict]
    ):

        if not chunks:

            print(
                "No chunks to insert."
            )

            return

        print(
            "\n========== EMBEDDINGS =========="
        )

        print(
            f"Chunks To Index: "
            f"{len(chunks)}"
        )

        texts = [
            c["text"]
            for c in chunks
        ]

        dense_vectors = (
            self.embedding_model
            .encode_batch(texts)
        )

        sparse_vectors = (
            self.embedding_model
            .encode_sparse(texts)
        )

        print(
            f"Dense Vectors: "
            f"{len(dense_vectors)}"
        )

        print(
            f"Sparse Vectors: "
            f"{len(sparse_vectors)}"
        )

        if sparse_vectors:

            print(
                f"Example Sparse Terms: "
                f"{len(sparse_vectors[0].indices)}"
            )

        points = []

        for (
            chunk,
            dense,
            sparse
        ) in zip(
            chunks,
            dense_vectors,
            sparse_vectors
        ):

            point = PointStruct(
                id=str(uuid.uuid4()),

                vector={
                    "dense":
                    dense,

                    "bm25":
                    SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist()
                    )
                },

                payload=chunk
            )

            points.append(
                point
            )

        print(
            "\n========== QDRANT INSERT =========="
        )

        print(
            f"Points: "
            f"{len(points)}"
        )

        print(
            f"Batch Size: "
            f"{Config.VECTOR_BATCH_SIZE}"
        )

        batch_size = (
            Config.VECTOR_BATCH_SIZE
        )

        inserted = 0

        for i in range(
            0,
            len(points),
            batch_size
        ):

            batch = (
                points[
                    i:
                    i + batch_size
                ]
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )

            inserted += len(batch)

            print(
                f"Inserted "
                f"{inserted}/"
                f"{len(points)}"
            )

        print(
            f"Successfully inserted "
            f"{len(points)} chunks."
        )

    # ======================================================
    # HYBRID SEARCH
    # ======================================================
    def hybrid_search(
        self,
        query: str,
        top_k: int
    ):

        print(
            "\n========== HYBRID SEARCH =========="
        )

        print(
            f"Query: {query}"
        )

        # --------------------------------------------------
        # Dense Embedding
        # --------------------------------------------------
        dense_vector = (
            self.embedding_model
            .encode(query)
        )

        print(
            f"Dense Dimension: "
            f"{len(dense_vector)}"
        )

        # --------------------------------------------------
        # Sparse Embedding
        # --------------------------------------------------
        sparse = (
            self.embedding_model
            .encode_sparse_single(
                query
            )
        )

        sparse_vector = (
            SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist()
            )
        )

        print(
            f"Sparse Terms: "
            f"{len(sparse.indices)}"
        )

        # --------------------------------------------------
        # Native Hybrid Search
        # --------------------------------------------------
        response = (
            self.client.query_points(
                collection_name=self.collection_name,

                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k * 3
                    ),

                    Prefetch(
                        query=sparse_vector,
                        using="bm25",
                        limit=top_k * 3
                    )
                ],

                query=FusionQuery(
                    fusion=Fusion.RRF
                ),

                limit=top_k,

                with_payload=True,
                with_vectors=False
            )
        )

        print(
            f"Retrieved: "
            f"{len(response.points)} docs"
        )

        # --------------------------------------------------
        # LOG RESULTS
        # --------------------------------------------------
        for i, point in enumerate(
            response.points,
            start=1
        ):

            payload = (
                point.payload
                or {}
            )

            print(
                f"{i}. "
                f"{payload.get('file_name')} "
                f"| Page="
                f"{payload.get('page_number')} "
                f"| Chunk="
                f"{payload.get('chunk_index')} "
                f"| Score="
                f"{round(point.score, 4)}"
            )

        return response.points