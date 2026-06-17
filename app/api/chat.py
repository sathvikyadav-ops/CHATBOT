import uuid
from fastapi import (
    APIRouter,
    HTTPException,
    Header,
    Response
)

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Source
)

from app.utils.dependencies import (
    retriever,
    llm,
    chat_memory
)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(
    body: ChatRequest,
    response: Response,
    x_session_id: str | None = Header(
        default=None,
        include_in_schema=False
    )
):

    try:

        query = body.query.strip()

        # ==========================================
        # SESSION ID
        # ==========================================
        session_id = (
            x_session_id
            or str(uuid.uuid4())
        )

        # Send session back only in header
        response.headers[
            "X-Session-Id"
        ] = session_id

        # ==========================================
        # CHAT HISTORY
        # ==========================================
        history = (
            chat_memory
            .get_history(session_id)[-5:]
        )

        history_text = ""

        for item in history:

            history_text += (
                f"User: {item.get('query','')}\n"
                f"Assistant: {item.get('answer','')}\n\n"
            )

        # ==========================================
        # RETRIEVAL
        # ==========================================
        retrieved_docs = (
            retriever.retrieve(query)
            or []
        )

        if not retrieved_docs:

            answer = (
                "I could not find relevant "
                "information in the uploaded documents."
            )

            chat_memory.save_message(
                session_id,
                query,
                answer
            )

            return ChatResponse(
                answer=answer,
                sources=[]
            )

        # ==========================================
        # CONTEXT
        # ==========================================
        context_parts = []

        for doc in retrieved_docs[:5]:

            context_parts.append(
                f"""
Document: {doc['file_name']}
Page: {doc['page']}

{doc['text']}
"""
            )

        context = f"""
Previous Conversation:
{history_text}

Retrieved Documents:
{''.join(context_parts)}
"""

        # ==========================================
        # LLM
        # ==========================================
        answer = llm.generate(
            query=query,
            context=context
        )

        # ==========================================
        # SAVE MEMORY
        # ==========================================
        chat_memory.save_message(
            session_id,
            query,
            answer
        )

        # ==========================================
        # SOURCES
        # ==========================================
        sources = [

            Source(
                file_name=d["file_name"],
                page=d["page"],
                score=round(
                    d["score"],
                    4
                ),
                chunk_text=d["text"][:500]
            )

            for d in retrieved_docs[:5]
        ]

        return ChatResponse(
            answer=answer,
            sources=sources
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )