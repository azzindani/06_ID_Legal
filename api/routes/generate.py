"""
Generate Routes

Endpoints for answer generation.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

router = APIRouter()


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    stream: bool = Field(False, description="Enable streaming response")


class Citation(BaseModel):
    regulation_type: str
    regulation_number: str
    year: str
    about: str


class GenerateResponse(BaseModel):
    answer: str
    query: str
    session_id: Optional[str]
    citations: List[Citation]
    metadata: Dict[str, Any]


@router.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    """
    Generate answer for a legal question

    Uses RAG pipeline to retrieve relevant documents and generate an answer.
    """
    from ..server import get_pipeline, get_conversation_manager

    try:
        pipeline = get_pipeline()
        manager = get_conversation_manager()

        # Get conversation context if session provided
        context = None
        if request.session_id:
            context = manager.get_context_for_query(request.session_id)

        # Generate answer
        result = pipeline.query(
            request.query,
            conversation_history=context,
            stream=False
        )

        # Extract citations
        citations = []
        for citation in result.get('metadata', {}).get('citations', []):
            citations.append(Citation(
                regulation_type=citation.get('regulation_type', 'N/A'),
                regulation_number=citation.get('regulation_number', 'N/A'),
                year=citation.get('year', 'N/A'),
                about=citation.get('about', 'N/A')
            ))

        # Add to conversation history if session provided
        if request.session_id:
            manager.add_turn(
                session_id=request.session_id,
                query=request.query,
                answer=result['answer'],
                metadata=result.get('metadata')
            )

        return GenerateResponse(
            answer=result['answer'],
            query=request.query,
            session_id=request.session_id,
            citations=citations,
            metadata=result.get('metadata', {})
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def generate_answer_stream(request: GenerateRequest):
    """
    Generate answer with streaming response

    Returns answer chunks as server-sent events.
    """
    from ..server import get_pipeline, get_conversation_manager

    async def generate():
        try:
            pipeline = get_pipeline()
            manager = get_conversation_manager()

            # Get conversation context
            context = None
            if request.session_id:
                context = manager.get_context_for_query(request.session_id)

            # Stream response
            full_answer = ""
            for chunk in pipeline.query(
                request.query,
                conversation_history=context,
                stream=True
            ):
                if isinstance(chunk, dict):
                    # Final result
                    full_answer = chunk.get('answer', full_answer)
                    yield f"data: {json.dumps({'type': 'done', 'answer': full_answer})}\n\n"
                else:
                    # Text chunk
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            # Save to history
            if request.session_id and full_answer:
                manager.add_turn(
                    session_id=request.session_id,
                    query=request.query,
                    answer=full_answer
                )

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
