# ==========================================================
# IMPORTS
# ==========================================================
from app.config import Config


# ==========================================================
# RETRIEVER
# ==========================================================
class Retriever:

    def __init__(
        self,
        vector_store
    ):
        self.vector_store = vector_store

    # ======================================================
    # RETRIEVE DOCUMENTS
    # ======================================================
    def retrieve(
        self,
        query: str
    ):

        print("\n========== RETRIEVAL ==========")
        print(f"Query: {query}")

        results = self.vector_store.search(
            query=query,
            top_k=Config.TOP_K
        )

        documents = []

        for result in results:

            print(
                f"Score: {result.score:.4f}"
            )

            documents.append(
                {
                    "text":
                    result.payload.get(
                        "text",
                        ""
                    ),

                    "score":
                    result.score,

                    "file_name":
                    result.payload.get(
                        "file_name",
                        "Unknown"
                    ),

                    "page":
                    result.payload.get(
                        "page_number",
                        0
                    ),

                    "chunk_index":
                    result.payload.get(
                        "chunk_index",
                        0
                    )
                }
            )

        print(
            f"Retrieved: {len(documents)}"
        )

        filtered = [
            doc
            for doc in documents
            if doc["score"]
            >= Config.SIMILARITY_THRESHOLD
        ]

        print(
            f"After threshold: {len(filtered)}"
        )

        # ==================================================
        # FALLBACK
        # ==================================================
        if not filtered and documents:

            best_match = max(
                documents,
                key=lambda x: x["score"]
            )

            print(
                f"Fallback score: "
                f"{best_match['score']:.4f}"
            )

            return [best_match]

        return filtered