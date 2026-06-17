from typing import List
from pydantic import BaseModel, Field


# ==========================================================
# CHAT REQUEST (NO session_id)
# ==========================================================
class ChatRequest(BaseModel):
    query: str = Field(
        example="What is Retrieval Augmented Generation?"
    )



# ==========================================================
# SOURCE MODEL
# ==========================================================
class Source(BaseModel):
    file_name: str
    page: int
    score: float
    chunk_text: str


# ==========================================================
# CHAT RESPONSE
# ==========================================================
class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)


# ==========================================================
# UPLOAD RESPONSE
# ==========================================================
class UploadResponse(BaseModel):
    status: str
    message: str
    files_uploaded: int = 0
    duplicates_skipped: List[str] = Field(default_factory=list)
    chunks_indexed: int = 0