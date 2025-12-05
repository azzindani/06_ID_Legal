"""
Session Routes

Endpoints for conversation session management.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import re

router = APIRouter()


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(None, max_length=100, description="Custom session ID")

    @validator('session_id')
    def validate_session_id(cls, v):
        """Validate session ID format"""
        if v is None:
            return v
        # Session ID should be alphanumeric with hyphens/underscores only
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Session ID must contain only alphanumeric characters, hyphens, and underscores")
        return v


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    total_turns: int


class SessionSummary(BaseModel):
    session_id: str
    total_turns: int
    total_queries: int
    total_tokens: int
    total_time: float


class TurnData(BaseModel):
    turn_number: int
    timestamp: str
    query: str
    answer: str


class SessionHistory(BaseModel):
    session_id: str
    turns: List[TurnData]


class ExportRequest(BaseModel):
    format: str = Field("json", description="Export format: md, json, html")

    @validator('format')
    def validate_format(cls, v):
        """Enforce whitelist of export formats"""
        allowed_formats = ['md', 'json', 'html', 'markdown']
        if v.lower() not in allowed_formats:
            raise ValueError(f"Export format must be one of: {', '.join(allowed_formats)}")
        # Normalize markdown variants
        if v.lower() == 'markdown':
            return 'md'
        return v.lower()


@router.post("/sessions", response_model=SessionResponse)
async def create_session(session_request: SessionCreateRequest, request: Request):
    """
    Create a new conversation session

    Returns session ID for tracking conversation history.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager(request)
        session_id = manager.start_session(session_request.session_id)

        session = manager.get_session(session_id)

        return SessionResponse(
            session_id=session_id,
            created_at=session.get('created_at', ''),
            total_turns=len(session.get('turns', []))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(request: Request):
    """
    List all active sessions

    Returns summary of all conversation sessions.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager(request)
        sessions = manager.list_sessions()

        return [
            SessionSummary(
                session_id=s['id'],
                total_turns=s.get('total_turns', 0),
                total_queries=s.get('total_queries', 0),
                total_tokens=s.get('total_tokens', 0),
                total_time=s.get('total_time', 0.0)
            )
            for s in sessions
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str, request: Request):
    """
    Get session summary

    Returns detailed summary of a specific session.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager(request)
        summary = manager.get_session_summary(session_id)

        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionSummary(
            session_id=summary['session_id'],
            total_turns=summary.get('total_turns', 0),
            total_queries=summary.get('total_queries', 0),
            total_tokens=summary.get('total_tokens', 0),
            total_time=summary.get('total_time', 0.0)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history", response_model=SessionHistory)
async def get_session_history(session_id: str, request: Request, max_turns: Optional[int] = None):
    """
    Get conversation history

    Returns all turns in a session.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager(request)
        history = manager.get_history(session_id, max_turns)

        if history is None:
            raise HTTPException(status_code=404, detail="Session not found")

        turns = [
            TurnData(
                turn_number=t['turn_number'],
                timestamp=t.get('timestamp', ''),
                query=t['query'],
                answer=t['answer']
            )
            for t in history
        ]

        return SessionHistory(
            session_id=session_id,
            turns=turns
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """
    End and delete a session

    Returns final session data before deletion.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager(request)
        result = manager.end_session(session_id)

        if not result:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"message": "Session deleted", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/export")
async def export_session(session_id: str, export_request: ExportRequest, request: Request):
    """
    Export session to file

    Exports conversation to specified format.
    """
    from ..server import get_conversation_manager
    from conversation import MarkdownExporter, JSONExporter, HTMLExporter

    try:
        manager = get_conversation_manager(request)
        session_data = manager.get_session(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        exporters = {
            'md': MarkdownExporter,
            'json': JSONExporter,
            'html': HTMLExporter
        }

        exporter_class = exporters.get(export_request.format, JSONExporter)
        exporter = exporter_class()

        content = exporter.export(session_data)

        return {
            "format": export_request.format,
            "content": content,
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
