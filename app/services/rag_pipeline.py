# ==========================================================
# IMPORTS
# ==========================================================
from pathlib import Path
import shutil
import hashlib
import traceback

from app.services.loader import PDFLoader
from app.services.chunker import TextChunker
from app.utils.metadata import MetadataBuilder


# ==========================================================
# RAG PIPELINE
# ==========================================================
class RAGPipeline:

    def __init__(
        self,
        vector_store=None
    ):

        self.loader = PDFLoader()

        self.chunker = (
            TextChunker()
        )

        self.metadata_builder = (
            MetadataBuilder()
        )

        self.vector_store = (
            vector_store
        )

    # ======================================================
    # INGEST FILES
    # ======================================================
    def ingest_files(
        self,
        files,
        upload_dir: Path,
        supported_ext: list
    ):

        upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        chunks_to_store = []

        print(
            "\n========== FILE INGESTION STARTED =========="
        )

        for file in files:

            try:

                extension = (
                    Path(
                        file.filename
                    ).suffix.lower()
                )

                # --------------------------------------------------
                # Unsupported File
                # --------------------------------------------------
                if extension not in supported_ext:

                    print(
                        f"Skipping: "
                        f"{file.filename}"
                    )

                    continue

                print(
                    f"\nProcessing: "
                    f"{file.filename}"
                )

                # --------------------------------------------------
                # Save File
                # --------------------------------------------------
                file_path = (
                    upload_dir /
                    file.filename
                )

                with open(
                    file_path,
                    "wb"
                ) as buffer:

                    shutil.copyfileobj(
                        file.file,
                        buffer
                    )

                # --------------------------------------------------
                # File Hash
                # --------------------------------------------------
                with open(
                    file_path,
                    "rb"
                ) as f:

                    file_hash = (
                        hashlib.sha256(
                            f.read()
                        ).hexdigest()
                    )

                # --------------------------------------------------
                # Duplicate Detection
                # --------------------------------------------------
                if (
                    self.vector_store
                    and
                    self.vector_store
                    .file_already_indexed(
                        file_hash
                    )
                ):

                    print(
                        f"{file.filename} "
                        f"already indexed."
                    )

                    continue

                # ==================================================
                # PDF
                # ==================================================
                if extension == ".pdf":

                    pages = (
                        self.loader
                        .extract_text_from_pdf(
                            str(file_path)
                        )
                    )

                    print(
                        f"Extracted "
                        f"{len(pages)} pages"
                    )

                    total_chunks = 0

                    for page_data in pages:

                        page_number = (
                            page_data[
                                "page_number"
                            ]
                        )

                        page_text = (
                            page_data[
                                "text"
                            ]
                        )

                        if not page_text.strip():
                            continue

                        chunks = (
                            self.chunker
                            .chunk_text(
                                page_text
                            )
                        )

                        total_chunks += (
                            len(chunks)
                        )

                        # --------------------------------------------------
                        # Create Metadata
                        # --------------------------------------------------
                        for (
                            chunk_index,
                            chunk
                        ) in enumerate(
                            chunks
                        ):

                            if not chunk.strip():
                                continue

                            metadata = (
                                self
                                .metadata_builder
                                .create_metadata(
                                    chunk_text=chunk,
                                    file_name=file.filename,
                                    file_type="pdf",
                                    page_number=page_number,
                                    chunk_index=chunk_index,
                                    file_hash=file_hash
                                )
                            )

                            chunks_to_store.append(
                                metadata
                            )

                    print(
                        f"Chunks Created: "
                        f"{total_chunks}"
                    )

                # ==================================================
                # TXT
                # ==================================================
                elif extension == ".txt":

                    text = (
                        file.file
                        .read()
                        .decode(
                            "utf-8",
                            errors="ignore"
                        )
                    )

                    chunks = (
                        self.chunker
                        .chunk_text(
                            text
                        )
                    )

                    for (
                        chunk_index,
                        chunk
                    ) in enumerate(
                        chunks
                    ):

                        metadata = (
                            self
                            .metadata_builder
                            .create_metadata(
                                chunk_text=chunk,
                                file_name=file.filename,
                                file_type="txt",
                                page_number=1,
                                chunk_index=chunk_index,
                                file_hash=file_hash
                            )
                        )

                        chunks_to_store.append(
                            metadata
                        )

                    print(
                        f"Chunks Created: "
                        f"{len(chunks)}"
                    )

                # ==================================================
                # DOCX / IMAGES
                # ==================================================
                else:

                    print(
                        f"{extension} "
                        f"processing not implemented."
                    )

            except Exception as e:

                print(
                    f"[INGEST ERROR] "
                    f"{file.filename}"
                )

                print(str(e))

                traceback.print_exc()

        # --------------------------------------------------
        # Final Summary
        # --------------------------------------------------
        print(
            "\n========== INGESTION COMPLETE =========="
        )

        print(
            f"Total Chunks To Store: "
            f"{len(chunks_to_store)}"
        )

        return chunks_to_store