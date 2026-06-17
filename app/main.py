# ==========================================================
# IMPORTS
# ==========================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.config import Config
from app.api.router import api_router

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


app.include_router(
    api_router
)