# ==========================================================
# IMPORTS
# ==========================================================
from pathlib import Path
import traceback

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import (
    CORSMiddleware
)
from fastapi.openapi.utils import get_openapi

from pydantic import BaseModel

from app.config import Config

from app.services.rag_pipeline import (
    RAGPipeline
)

from app.services.vector_store import (
    VectorStore
)

from app.services.retriever import (
    Retriever
)

from app.services.llm import (
    GroqLLM
)


# ==========================================================
# VALIDATE CONFIG
# ==========================================================
Config.validate()


# ==========================================================
# APP
# ==========================================================
app = FastAPI(
    title=Config.APP_NAME,
    version=Config.APP_VERSION
)


# ---------------------------------------------------
# OPENAPI FIX
# ---------------------------------------------------

def _openapi_file_property(prop: dict):
    if prop.get("type") == "string" and "contentMediaType" in prop:
        prop["format"] = "binary"
        prop.pop("contentMediaType", None)

    if prop.get("type") == "array" and "items" in prop:
        _openapi_file_property(prop["items"])


def _openapi_fix_schema(obj):
    if isinstance(obj, dict):
        for prop in obj.get("properties", {}).values():
            _openapi_file_property(prop)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                _openapi_fix_schema(value)
    elif isinstance(obj, list):
        for item in obj:
            _openapi_fix_schema(item)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )

    for body in schema.get("components", {}).get("schemas", {}).values():
        for prop in body.get("properties", {}).values():
            _openapi_file_property(prop)

    _openapi_fix_schema(schema.get("paths", {}))

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

# ==========================================================
# CORS
# ==========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================================
# SERVICES
# ==========================================================
pipeline = RAGPipeline()

vector_store = VectorStore()

retriever = Retriever(
    vector_store
)

llm = GroqLLM()


# ==========================================================
# ROOT
# ==========================================================
@app.get("/")
def root():

    return {
        "status": "success",
        "message": "RAG API Running"
    }


# ==========================================================
# HEALTH
# ==========================================================
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==========================================================
# CHAT REQUEST
# ==========================================================
class ChatRequest(
    BaseModel
):
    query: str


# ==========================================================
# UPLOAD
# ==========================================================

@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...)
):

    try:

        # ==================================================
        # STORAGE VARIABLES
        # ==================================================
        indexed_files = []
        duplicate_files = []
        chunks_to_store = []

        upload_dir = Path(Config.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # ==================================================
        # STEP 1: FILTER DUPLICATES FIRST
        # ==================================================
        for file in files:

            if (
                Config.ENABLE_DUPLICATE_CHECK
                and vector_store.file_already_indexed(file.filename)
            ):
                duplicate_files.append(file.filename)
                continue

            indexed_files.append(file)

        # ==================================================
        # IF ALL FILES ARE DUPLICATES
        # ==================================================
        if not indexed_files:

            return {
                "status": "duplicate",
                "message": "All files already exist in database.",
                "duplicates": duplicate_files
            }

        # ==================================================
        # STEP 2: INGEST ONLY NEW FILES
        # ==================================================
        chunks_to_store = pipeline.ingest_files(
            files=indexed_files,
            upload_dir=upload_dir,
            supported_ext=Config.SUPPORTED_EXTENSIONS
        )

        # ==================================================
        # VALIDATION
        # ==================================================
        if not chunks_to_store:

            raise HTTPException(
                status_code=400,
                detail="No text extracted from uploaded files."
            )

        # ==================================================
        # STEP 3: STORE IN VECTOR DB
        # ==================================================
        vector_store.add_documents(chunks_to_store)

        # ==================================================
        # RESPONSE
        # ==================================================
        return {
            "status": "success",
            "files_uploaded": len(indexed_files),
            "duplicates_skipped": duplicate_files,
            "chunks_indexed": len(chunks_to_store)
        }

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(body: ChatRequest):

    try:

        query = body.query

        # ==================================================
        # RETRIEVE DOCS
        # ==================================================
        retrieved_docs = retriever.retrieve(query)

        if not retrieved_docs:
            return {
                "answer": "I could not find this information in the uploaded documents.",
                "sources": []
            }

        # ==================================================
        # BUILD CONTEXT
        # ==================================================
        context_parts = []

        for doc in retrieved_docs:

            context_parts.append(
                f"""
Source: {doc['file_name']}
Page: {doc['page']}
Score: {round(doc['score'], 4)}

Text:
{doc['text']}
"""
            )

        context = "\n\n".join(context_parts)

        # ==================================================
        # LLM GENERATION
        # ==================================================
        answer = llm.generate(
            query=query,
            context=context
        )

        # ==================================================
        # BUILD SOURCES (WITH CHUNK TEXT + SCORE)
        # ==================================================
        sources = []

        for doc in retrieved_docs:

            sources.append({
                "file_name": doc["file_name"],
                "page": doc["page"],
                "score": round(doc["score"], 4),
                "chunk_text": doc["text"][:500]  # safe limit
            })

        # sort best matches first
        sources = sorted(sources, key=lambda x: x["score"], reverse=True)

        # ==================================================
        # RESPONSE
        # ==================================================
        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )