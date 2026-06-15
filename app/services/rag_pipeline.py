# ==========================================================
# IMPORTS
# ==========================================================
from pathlib import Path
import shutil
import hashlib

from app.services.loader import PDFLoader
from app.services.chunker import TextChunker
from app.utils.metadata import MetadataBuilder


# ==========================================================
# RAG PIPELINE
# ==========================================================
class RAGPipeline:

    def __init__(self):

        self.loader = PDFLoader()

        self.chunker = TextChunker()

        self.metadata_builder = (
            MetadataBuilder()
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

        for file in files:

            extension = (
                Path(
                    file.filename
                ).suffix.lower()
            )

            if extension not in supported_ext:
                continue

            # ==================================================
            # SAVE FILE
            # ==================================================
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

            # ==================================================
            # GENERATE FILE HASH
            # ==================================================
            with open(
                file_path,
                "rb"
            ) as f:

                file_hash = hashlib.sha256(
                    f.read()
                ).hexdigest()

            # ==================================================
            # PDF PROCESSING
            # ==================================================
            if extension == ".pdf":

                pages = (
                    self.loader
                    .extract_text_from_pdf(
                        str(file_path)
                    )
                )

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

                    for (
                        chunk_index,
                        chunk
                    ) in enumerate(
                        chunks
                    ):

                        if not chunk.strip():
                            continue

                        metadata = (
                            self.metadata_builder
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

        return chunks_to_store