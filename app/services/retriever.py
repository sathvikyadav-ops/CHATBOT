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
        self.vector_store = (
            vector_store
        )

    # ======================================================
    # RETRIEVE DOCUMENTS
    # ======================================================
    def retrieve(
        self,
        query: str
    ):

        print(
            f"\nSearch: {query}"
        )

        try:

            # --------------------------------------------------
            # Hybrid Search
            # --------------------------------------------------
            results = (
                self.vector_store
                .hybrid_search(
                    query=query,
                    top_k=Config.TOP_K
                )
            )

            if not results:

                print(
                    "No documents retrieved."
                )

                return []

            documents = []

            for result in results:

                payload = (
                    result.payload or {}
                )

                documents.append(
                    {
                        "text":
                        payload.get(
                            "text",
                            ""
                        ),

                        "score":
                        round(
                            result.score,
                            4
                        ),

                        "file_name":
                        payload.get(
                            "file_name",
                            "Unknown"
                        ),

                        "page":
                        payload.get(
                            "page_number",
                            0
                        ),

                        "chunk_index":
                        payload.get(
                            "chunk_index",
                            0
                        ),

                        "source":
                        payload.get(
                            "source",
                            ""
                        )
                    }
                )

            print(
                f"Retrieved {len(documents)} documents."
            )

            return documents

        except Exception as e:

            print(
                f"[RETRIEVAL ERROR] {e}"
            )

            return []