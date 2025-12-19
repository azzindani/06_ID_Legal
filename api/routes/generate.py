"""
Generate Routes

Endpoints for answer generation.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import json

from ..validators import validate_query, validate_session_id

router = APIRouter()


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    session_id: Optional[str] = Field(None, max_length=100, description="Session ID for context")
    stream: bool = Field(False, description="Enable streaming response")

    @validator('query')
    def validate_query_field(cls, v):
        """Validate query using shared validator"""
        return validate_query(v)

    @validator('session_id')
    def validate_session_id_field(cls, v):
        """Validate session ID using shared validator"""
        return validate_session_id(v)


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
async def generate_answer(gen_request: GenerateRequest, request: Request):
    """
    Generate answer for a legal question

    Uses RAG pipeline to retrieve relevant documents and generate an answer.
    """
    from ..server import get_pipeline, get_conversation_manager

    try:
        pipeline = get_pipeline(request)
        manager = get_conversation_manager(request)

        # Get conversation context if session provided
        context = None
        if gen_request.session_id:
            context = manager.get_context_for_query(gen_request.session_id)

        # Generate answer
        result = pipeline.query(
            gen_request.query,
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
        if gen_request.session_id:
            manager.add_turn(
                session_id=gen_request.session_id,
                query=gen_request.query,
                answer=result['answer'],
                metadata=result.get('metadata')
            )

        return GenerateResponse(
            answer=result['answer'],
            query=gen_request.query,
            session_id=gen_request.session_id,
            citations=citations,
            metadata=result.get('metadata', {})
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def generate_answer_stream(gen_request: GenerateRequest, request: Request):
    """
    Generate answer with streaming response

    Returns answer chunks as server-sent events.
    """
    from ..server import get_pipeline, get_conversation_manager

    async def generate():
        try:
            pipeline = get_pipeline(request)
            manager = get_conversation_manager(request)

            # Get conversation context
            context = None
            if gen_request.session_id:
                context = manager.get_context_for_query(gen_request.session_id)

            # Stream response
            full_answer = ""
            for chunk in pipeline.query(
                gen_request.query,
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
            if gen_request.session_id and full_answer:
                manager.add_turn(
                    session_id=gen_request.session_id,
                    query=gen_request.query,
                    answer=full_answer
                )

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
