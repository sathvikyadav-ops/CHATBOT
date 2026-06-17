# ==========================================================
# IMPORTS
# ==========================================================
import os
from dotenv import load_dotenv

# ==========================================================
# LOAD ENV VARIABLES
# ==========================================================
load_dotenv()


# ==========================================================
# CONFIG CLASS
# ==========================================================
class Config:

    # ======================================================
    # QDRANT
    # ======================================================
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", 300))


    # ======================================================
    # GROQ LLM
    # ======================================================
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    # ======================================================
    # EMBEDDINGS
    # ======================================================
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ======================================================
    # OCR
    # ======================================================
    TESSERACT_PATH = os.getenv(
        "TESSERACT_PATH",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    OCR_DPI = int(os.getenv("OCR_DPI", 200))

    # ======================================================
    # RETRIEVAL (VERY IMPORTANT)
    # ======================================================
    TOP_K = int(os.getenv("TOP_K", 5))


    # ======================================================
    # CHUNKING (CRITICAL FOR RAG QUALITY)
    # ======================================================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # ======================================================
    # STORAGE
    # ======================================================
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
    PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")

    # ======================================================
    # FILE TYPES
    # ======================================================
    SUPPORTED_EXTENSIONS = [
        ".pdf",
        ".docx",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg"
    ]

    # ======================================================
    # VECTOR DB
    # ======================================================
    VECTOR_BATCH_SIZE = int(os.getenv("VECTOR_BATCH_SIZE", 50))

    # ======================================================
    # DUPLICATE DETECTION
    # ======================================================
    ENABLE_DUPLICATE_CHECK = (
        os.getenv("ENABLE_DUPLICATE_CHECK", "true").lower() == "true"
    )

    # ======================================================
    # CHAT MEMORY 
    # ======================================================
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB = os.getenv("MONGO_DB")
    CHAT_COLLECTION = os.getenv("CHAT_COLLECTION")
    CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 5))

    # ======================================================
    # APP SETTINGS
    # ======================================================
    APP_NAME = os.getenv("APP_NAME", "RAG Chatbot API")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # ======================================================
    # VALIDATION
    # ======================================================
    @classmethod
    def validate(cls):

        missing = []

        if not cls.QDRANT_URL:
            missing.append("QDRANT_URL")

        if not cls.QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")

        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return True