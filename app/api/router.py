# ==========================================================
# IMPORTS
# ==========================================================
from fastapi import APIRouter

from app.api.root import router as root_router
from app.api.health import router as health_router
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router


# ==========================================================
# API ROUTER
# ==========================================================
api_router = APIRouter()

api_router.include_router(
    root_router
)

api_router.include_router(
    health_router
)

api_router.include_router(
    upload_router
)

api_router.include_router(
    chat_router
)