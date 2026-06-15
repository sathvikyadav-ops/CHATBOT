from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    message: str
    files_processed: int