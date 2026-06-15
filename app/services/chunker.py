# ==========================================================
# IMPORTS
# ==========================================================
import re

from app.config import Config
from app.utils.text_utils import TextCleaner


# ==========================================================
# TEXT CHUNKER
# ==========================================================
class TextChunker:

    def __init__(self):

        self.chunk_size = (
            Config.CHUNK_SIZE
        )

        self.chunk_overlap = (
            Config.CHUNK_OVERLAP
        )

        self.cleaner = (
            TextCleaner()
        )

    # ======================================================
    # SPLIT INTO SENTENCES
    # ======================================================
    def split_sentences(
        self,
        text: str
    ) -> list[str]:

        if not text:
            return []

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # ======================================================
    # SPLIT VERY LARGE SENTENCE
    # ======================================================
    def split_large_sentence(
        self,
        sentence: str
    ) -> list[str]:

        if len(sentence) <= self.chunk_size:
            return [sentence]

        parts = []

        start = 0

        while start < len(sentence):

            end = (
                start +
                self.chunk_size
            )

            parts.append(
                sentence[start:end]
            )

            start = end

        return parts

    # ======================================================
    # CREATE OVERLAP
    # ======================================================
    def create_overlap(
        self,
        text: str
    ) -> str:

        if len(text) <= self.chunk_overlap:
            return text

        return text[
            -self.chunk_overlap:
        ]

    # ======================================================
    # MAIN CHUNKING
    # ======================================================
    def chunk_text(
        self,
        text: str
    ) -> list[str]:

        # ------------------------------------------
        # CLEAN TEXT
        # ------------------------------------------
        text = (
            self.cleaner.clean_text(
                text
            )
        )

        if not text:
            return []

        sentences = (
            self.split_sentences(
                text
            )
        )

        chunks = []

        current_chunk = ""

        for sentence in sentences:

            # --------------------------------------
            # HANDLE HUGE SENTENCES
            # --------------------------------------
            if len(sentence) > self.chunk_size:

                large_parts = (
                    self.split_large_sentence(
                        sentence
                    )
                )

                for part in large_parts:

                    if current_chunk.strip():

                        chunks.append(
                            current_chunk.strip()
                        )

                        current_chunk = ""

                    chunks.append(
                        part.strip()
                    )

                continue

            # --------------------------------------
            # APPEND IF FITS
            # --------------------------------------
            proposed_length = (
                len(current_chunk)
                + len(sentence)
                + 1
            )

            if proposed_length <= self.chunk_size:

                current_chunk += (
                    " " + sentence
                )

            else:

                # Save current chunk
                if current_chunk.strip():

                    chunks.append(
                        current_chunk.strip()
                    )

                overlap = (
                    self.create_overlap(
                        current_chunk
                    )
                )

                current_chunk = (
                    overlap +
                    " " +
                    sentence
                )

        # ------------------------------------------
        # FINAL CHUNK
        # ------------------------------------------
        if current_chunk.strip():

            chunks.append(
                current_chunk.strip()
            )

        # ------------------------------------------
        # REMOVE EMPTY CHUNKS
        # ------------------------------------------
        chunks = [
            chunk
            for chunk in chunks
            if chunk.strip()
        ]

        return chunks

    # ======================================================
    # CHUNK PAGE
    # ======================================================
    def chunk_page(
        self,
        page_text: str,
        page_number: int
    ) -> list[dict]:

        chunks = self.chunk_text(
            page_text
        )

        results = []

        for index, chunk in enumerate(
            chunks
        ):

            results.append(
                {
                    "page_number":
                    page_number,

                    "chunk_index":
                    index,

                    "text":
                    chunk
                }
            )

        return results

    # ======================================================
    # STATS
    # ======================================================
    def get_chunk_stats(
        self,
        chunks: list[str]
    ) -> dict:

        if not chunks:

            return {
                "total_chunks": 0,
                "avg_chunk_size": 0
            }

        sizes = [
            len(chunk)
            for chunk in chunks
        ]

        return {
            "total_chunks":
            len(chunks),

            "avg_chunk_size":
            round(
                sum(sizes)
                / len(sizes),
                2
            ),

            "largest_chunk":
            max(sizes),

            "smallest_chunk":
            min(sizes)
        }