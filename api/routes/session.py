"""
Session Routes

Endpoints for conversation session management.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

router = APIRouter()


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Custom session ID")


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


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Create a new conversation session

    Returns session ID for tracking conversation history.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager()
        session_id = manager.start_session(request.session_id)

        session = manager.get_session(session_id)

        return SessionResponse(
            session_id=session_id,
            created_at=session.get('created_at', ''),
            total_turns=len(session.get('turns', []))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    """
    List all active sessions

    Returns summary of all conversation sessions.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager()
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
async def get_session(session_id: str):
    """
    Get session summary

    Returns detailed summary of a specific session.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager()
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
async def get_session_history(session_id: str, max_turns: Optional[int] = None):
    """
    Get conversation history

    Returns all turns in a session.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager()
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
async def delete_session(session_id: str):
    """
    End and delete a session

    Returns final session data before deletion.
    """
    from ..server import get_conversation_manager

    try:
        manager = get_conversation_manager()
        result = manager.end_session(session_id)

        if not result:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"message": "Session deleted", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/export")
async def export_session(session_id: str, request: ExportRequest):
    """
    Export session to file

    Exports conversation to specified format.
    """
    from ..server import get_conversation_manager
    from conversation import MarkdownExporter, JSONExporter, HTMLExporter

    try:
        manager = get_conversation_manager()
        session_data = manager.get_session(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        exporters = {
            'md': MarkdownExporter,
            'json': JSONExporter,
            'html': HTMLExporter
        }

        exporter_class = exporters.get(request.format, JSONExporter)
        exporter = exporter_class()

        content = exporter.export(session_data)

        return {
            "format": request.format,
            "content": content,
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
