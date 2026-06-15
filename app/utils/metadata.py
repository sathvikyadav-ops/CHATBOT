# ==========================================================
# IMPORTS
# ==========================================================
from datetime import datetime


# ==========================================================
# METADATA BUILDER
# ==========================================================
class MetadataBuilder:

    def create_metadata(
        self,
        chunk_text: str,
        file_name: str,
        file_type: str,
        page_number: int,
        chunk_index: int,
        file_hash: str
    ):

        return {
            "text": chunk_text,
            "file_name": file_name,
            "file_type": file_type,
            "file_hash": file_hash,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "source": f"{file_name} | Page {page_number}"
        }