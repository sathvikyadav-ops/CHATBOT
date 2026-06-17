# ==========================================================
# IMPORTS
# ==========================================================
from pathlib import Path
import hashlib
import traceback

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.config import Config
from app.models.schemas import UploadResponse
from app.utils.dependencies import (
    vector_store,
    pipeline
)


# ==========================================================
# ROUTER
# ==========================================================
router = APIRouter()


# ==========================================================
# UPLOAD FILES
# ==========================================================
@router.post(
    "/upload",
    response_model=UploadResponse
)
async def upload_files(
    files: list[UploadFile] = File(...)
):

    try:

        print(
            "\n========== FILE UPLOAD =========="
        )

        upload_dir = Path(
            Config.UPLOAD_DIR
        )

        upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        indexed_files = []
        duplicate_files = []

        # ==================================================
        # DUPLICATE CHECK
        # ==================================================
        for file in files:

            content = await file.read()

            file_hash = hashlib.sha256(
                content
            ).hexdigest()

            # Reset file pointer
            file.file.seek(0)

            # ----------------------------------------------
            # Check if file already exists
            # ----------------------------------------------
            if (
                Config.ENABLE_DUPLICATE_CHECK
                and
                vector_store.file_already_indexed(
                    file_hash
                )
            ):

                print(
                    f"Duplicate File: "
                    f"{file.filename}"
                )

                duplicate_files.append(
                    file.filename
                )

                continue

            indexed_files.append(
                (
                    file,
                    file_hash
                )
            )

        # ==================================================
        # ALL FILES ARE DUPLICATES
        # ==================================================
        if not indexed_files:

            return UploadResponse(
                status="duplicate",
                message=(
                    "Files already exist "
                    "in database."
                ),
                files_uploaded=0,
                duplicates_skipped=duplicate_files,
                chunks_indexed=0
            )

        # ==================================================
        # INGEST FILES
        # ==================================================
        chunks_to_store = (
            pipeline.ingest_files(
                files=[
                    file
                    for file, _
                    in indexed_files
                ],
                upload_dir=upload_dir,
                supported_ext=(
                    Config
                    .SUPPORTED_EXTENSIONS
                )
            )
        )

        # ==================================================
        # NO TEXT EXTRACTED
        # ==================================================
        if not chunks_to_store:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No text extracted "
                    "from uploaded files."
                )
            )

        # ==================================================
        # ADD FILE HASH TO CHUNKS
        # ==================================================
        file_hash_map = {
            file.filename: file_hash
            for file, file_hash
            in indexed_files
        }

        for chunk in chunks_to_store:

            chunk["file_hash"] = (
                file_hash_map.get(
                    chunk["file_name"]
                )
            )

        print(
            f"Chunks To Index: "
            f"{len(chunks_to_store)}"
        )

        # ==================================================
        # STORE IN QDRANT
        # ==================================================
        vector_store.add_documents(
            chunks_to_store
        )

        print(
            "Indexing Completed."
        )

        # ==================================================
        # RESPONSE
        # ==================================================
        return UploadResponse(
            status="success",
            message=(
                "Files uploaded and "
                "indexed successfully."
            ),
            files_uploaded=len(
                indexed_files
            ),
            duplicates_skipped=duplicate_files,
            chunks_indexed=len(
                chunks_to_store
            )
        )

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )