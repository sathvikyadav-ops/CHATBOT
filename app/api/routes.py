# app/main.py

from pathlib import Path
from typing import List
import shutil

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from app.services.loader import load_file
from app.services.chunker import chunk_text
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.llm import GrokLLM

from app.utils.metadata import (
    create_metadata
)

from app.config import Config


# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0"
)

# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# SERVICES
# ==========================================================

vector_store = VectorStore()

retriever = Retriever()

llm = GrokLLM()

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
# UPLOAD
# ==========================================================

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...)
):

    upload_dir = Path(
        Config.UPLOAD_DIR
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks_to_store = []

    try:

        for file in files:

            extension = (
                Path(
                    file.filename
                ).suffix.lower()
            )

            if (
                extension
                not in Config.SUPPORTED_EXTENSIONS
            ):
                continue

            file_path = (
                upload_dir
                / file.filename
            )

            with open(
                file_path,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            pages = load_file(
                str(file_path)
            )

            for page_data in pages:

                page_number = (
                    page_data["page"]
                )

                page_text = (
                    page_data["text"]
                )

                if not page_text.strip():
                    continue

                chunks = chunk_text(
                    page_text
                )

                for chunk_index, chunk in enumerate(chunks):

                    metadata = (
                        create_metadata(
                            chunk_text=chunk,
                            file_name=file.filename,
                            file_type=extension.replace(
                                ".",
                                ""
                            ),
                            page_number=page_number,
                            chunk_index=chunk_index
                        )
                    )

                    chunks_to_store.append(
                        metadata
                    )

        if not chunks_to_store:

            raise HTTPException(
                status_code=400,
                detail="No text extracted."
            )

        vector_store.add_documents(
            chunks_to_store
        )

        return {
            "status": "success",
            "files_uploaded": len(files),
            "chunks_indexed": len(
                chunks_to_store
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# CHAT
# ==========================================================

@app.post("/chat")
async def chat(
    body: dict
):

    try:

        query = body.get(
            "query"
        )

        if not query:

            raise HTTPException(
                status_code=400,
                detail="Query required"
            )

        retrieved_docs = (
            retriever.retrieve(
                query
            )
        )

        if not retrieved_docs:

            return {
                "answer":
                "I could not find this information in the uploaded documents.",
                "sources": []
            }

        context_parts = []

        sources = set()

        for doc in retrieved_docs:

            context_parts.append(
                f"""
Source: {doc['file_name']}
Page: {doc['page']}

{doc['text']}
"""
            )

            sources.add(
                doc["file_name"]
            )

        context = "\n\n".join(
            context_parts
        )

        answer = llm.generate(
            query=query,
            context=context
        )

        return {
            "answer": answer,
            "sources": list(
                sources
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )