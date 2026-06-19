# ==========================================================
# IMPORTS
# ==========================================================
from app.config import Config


# ==========================================================
# RETRIEVER
# ==========================================================
class Retriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    # ======================================================
    # RETRIEVE DOCUMENTS
    # ======================================================
    def retrieve(
        self,
        query: str,
        file_name: str = None,
        is_summary: bool = False
    ):

        print("\n========== RETRIEVER ==========")
        print(f"Query: {query}")
        print(f"Requested File: {file_name}")
        print(f"Is Summary: {is_summary}")

        try:

            # ----------------------------------------------
            # APPLY FILE FILTER ONLY FOR SUMMARIZATION
            # ----------------------------------------------
            search_file = None

            if is_summary:

                if not file_name:
                    raise ValueError(
                        "file_name is required when is_summary=True"
                    )

                search_file = file_name

            print(f"Search File Filter: {search_file}")

            # ----------------------------------------------
            # HYBRID SEARCH
            # ----------------------------------------------
            results = self.vector_store.hybrid_search(
                query=query,
                top_k=Config.TOP_K,
                file_name=search_file
            )

            if not results:
                print("No documents retrieved.")
                return []

            documents = []

            print("\n========== RETRIEVED FILES ==========")

            for result in results:

                payload = result.payload or {}

                retrieved_file = payload.get(
                    "file_name",
                    "Unknown"
                )

                print(
                    f"Retrieved: "
                    f"{retrieved_file}"
                )

                # ------------------------------------------
                # EXTRA SAFETY CHECK
                # ------------------------------------------
                if (
                    is_summary
                    and search_file
                    and retrieved_file != search_file
                ):
                    print(
                        f"[SKIPPED] "
                        f"Expected={search_file} "
                        f"Got={retrieved_file}"
                    )
                    continue

                documents.append(
                    {
                        "text":
                        payload.get("text", ""),

                        "score":
                        round(result.score, 4),

                        "file_name":
                        retrieved_file,

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
                f"Retrieved "
                f"{len(documents)} "
                f"documents."
            )

            return documents

        except Exception as e:

            print(
                f"[RETRIEVAL ERROR] {e}"
            )

            return []