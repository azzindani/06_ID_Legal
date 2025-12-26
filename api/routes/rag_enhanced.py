"""
Enhanced RAG Routes

Specialized endpoints for advanced legal intelligence services.
Provides retrieval-only, deep research, and conversational capabilities.

File: api/routes/rag_enhanced.py
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..validators import validate_query, validate_session_id
from security import sanitize_query
from utils.logger_utils import get_logger
from utils.formatting import format_sources_info as format_detailed_sources_info, format_all_documents
from utils.research_transparency import format_detailed_research_process

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RetrievalRequest(BaseModel):
    """Request for pure retrieval (no LLM generation)"""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance score")
    
    @validator('query')
    def validate_query_field(cls, v):
        return validate_query(v)


class ResearchRequest(BaseModel):
    """Request for deep research mode"""
    query: str = Field(..., min_length=1, max_length=2000, description="Legal question")
    thinking_level: str = Field('high', description="Thinking depth: low, medium, high")
    team_size: int = Field(4, ge=1, le=5, description="Research team size")
    max_tokens: Optional[int] = Field(None, description="Override max tokens")
    
    @validator('query')
    def validate_query_field(cls, v):
        return validate_query(v)
    
    @validator('thinking_level')
    def validate_thinking_level(cls, v):
        allowed = ['low', 'medium', 'high']
        if v.lower() not in allowed:
            raise ValueError(f"thinking_level must be one of: {', '.join(allowed)}")
        return v.lower()


class ChatRequest(BaseModel):
    """Request for conversational RAG"""
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, max_length=100, description="Session ID for context")
    thinking_level: str = Field('low', description="Thinking depth: low, medium, high")
    stream: bool = Field(False, description="Enable streaming response")
    # Config parameters from UI
    top_k: int = Field(3, ge=1, le=30, description="Number of documents to retrieve")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(2048, ge=256, le=8192, description="Max tokens to generate")
    team_size: int = Field(3, ge=1, le=5, description="Research team size")
    
    @validator('query')
    def validate_query_field(cls, v):
        return validate_query(v)
    
    @validator('session_id')
    def validate_session_id_field(cls, v):
        return validate_session_id(v)
    
    @validator('thinking_level')
    def validate_thinking_level(cls, v):
        allowed = ['low', 'medium', 'high']
        if v.lower() not in allowed:
            raise ValueError(f"thinking_level must be one of: {', '.join(allowed)}")
        return v.lower()


class LegalDocument(BaseModel):
    """Legal document metadata"""
    regulation_type: str
    regulation_number: str
    year: str
    about: str
    chapter: Optional[str] = None
    article: Optional[str] = None
    effective_date: Optional[str] = None
    content_preview: Optional[str] = None
    score: float


class RetrievalResponse(BaseModel):
    """Response for retrieval endpoint"""
    query: str
    documents: List[LegalDocument]
    total_retrieved: int
    search_time: float
    metadata: Dict[str, Any]


class ResearchResponse(BaseModel):
    """Response for research endpoint"""
    answer: str
    legal_references: str
    research_process: str
    all_retrieved_documents: str
    query: str
    thinking_level: str
    citations: List[LegalDocument]
    metadata: Dict[str, Any]
    research_time: float


class ChatResponse(BaseModel):
    """Response for chat endpoint"""
    answer: str
    legal_references: str
    research_process: str
    all_retrieved_documents: str
    query: str
    session_id: Optional[str]
    citations: List[LegalDocument]
    metadata: Dict[str, Any]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Note: format_legal_references removed - use utils.formatting.format_sources_info instead


def extract_documents(result: Dict) -> List[LegalDocument]:
    """Extract and format legal documents from pipeline result"""
    documents = []
    
    # Get citations from metadata
    citations = result.get('citations') or result.get('metadata', {}).get('citations', [])
    
    for citation in citations:
        record = citation if isinstance(citation, dict) else citation.get('record', {})
        
        documents.append(LegalDocument(
            regulation_type=record.get('regulation_type', 'N/A'),
            regulation_number=record.get('regulation_number', 'N/A'),
            year=record.get('year', 'N/A'),
            about=record.get('about', 'N/A'),
            chapter=record.get('chapter') or record.get('bab'),
            article=record.get('article') or record.get('pasal'),
            effective_date=record.get('effective_date') or record.get('tanggal_penetapan'),
            content_preview=record.get('content', '')[:200] if record.get('content') else None,
            score=citation.get('score', 0.0) if isinstance(citation, dict) else 0.0
        ))
    
    return documents


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/rag/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(req: RetrievalRequest, request: Request):
    """
    Pure retrieval service - returns ranked legal documents without LLM generation
    
    **Use Case**: Fast document search without answer generation
    - No LLM overhead
    - Direct access to hybrid search + reranking
    - Ideal for building custom UIs or bulk retrieval
    
    **Parameters**:
    - `query`: Legal search query
    - `top_k`: Number of results (1-10)
    - `min_score`: Minimum relevance threshold (0.0-1.0)
    """
    from ..server import get_pipeline
    import time
    
    start_time = time.time()
    
    try:
        pipeline = get_pipeline(request)
        
        # Use retrieval-only mode
        # We'll call the orchestrator directly to bypass LLM
        if not hasattr(pipeline, 'orchestrator') or pipeline.orchestrator is None:
            raise HTTPException(
                status_code=500,
                detail="Retrieval system not initialized"
            )
        
        # Run orchestrator for retrieval with top_k parameter
        logger.info(f"Retrieve called with top_k={req.top_k}")
        rag_result = pipeline.orchestrator.run(
            query=req.query,
            conversation_history=[],
            top_k=req.top_k  # Pass top_k to orchestrator
        )
        
        final_results = rag_result.get('final_results', [])
        
        # Filter by minimum score
        if req.min_score > 0:
            final_results = [
                r for r in final_results
                if r.get('scores', {}).get('final', 0) >= req.min_score
            ]
        
        # Limit to top_k
        final_results = final_results[:req.top_k]
        
        # Convert to LegalDocument format
        documents = []
        for result in final_results:
            record = result.get('record', {})
            scores = result.get('scores', {})
            
            documents.append(LegalDocument(
                regulation_type=record.get('regulation_type', 'N/A'),
                regulation_number=record.get('regulation_number', 'N/A'),
                year=record.get('year', 'N/A'),
                about=record.get('about', 'N/A'),
                chapter=record.get('chapter') or record.get('bab'),
                article=record.get('article') or record.get('pasal'),
                effective_date=record.get('effective_date') or record.get('tanggal_penetapan'),
                content_preview=record.get('content', '')[:200] if record.get('content') else None,
                score=scores.get('final', 0.0)
            ))
        
        search_time = time.time() - start_time
        
        return RetrievalResponse(
            query=req.query,
            documents=documents,
            total_retrieved=len(documents),
            search_time=search_time,
            metadata={
                'retrieval_only': True,
                'min_score': req.min_score,
                'total_candidates': len(rag_result.get('final_results', []))
            }
        )
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/research", response_model=ResearchResponse)
async def deep_research(req: ResearchRequest, request: Request):
    """
    Deep research service - comprehensive legal analysis with high thinking mode
    
    **Use Case**: Complex legal questions requiring thorough analysis
    - Automatically uses 'high' thinking level (overridable)
    - Configurable research team size
    - Includes formatted legal references
    
    **Parameters**:
    - `query`: Complex legal question
    - `thinking_level`: Depth of analysis (default: high)
    - `team_size`: Number of research team members (1-5)
    - `max_tokens`: Optional token override
    """
    from ..server import get_pipeline
    import time
    
    start_time = time.time()
    
    try:
        pipeline = get_pipeline(request)
        
        # Build config overrides
        config_overrides = {
            'research_team_size': req.team_size
        }
        
        if req.max_tokens:
            config_overrides['max_new_tokens'] = req.max_tokens
        
        # Update pipeline config
        pipeline.update_config(**config_overrides)
        
        # Execute with specified thinking level
        result = pipeline.query(
            question=req.query,
            conversation_history=None,
            stream=False,
            thinking_mode=req.thinking_level
        )
        
        if not result.get('success', True):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Research failed')
            )
        
        # Extract documents
        documents = extract_documents(result)
        
        # Format legal references
        citations_data = result.get('citations', [])
        legal_refs = format_detailed_sources_info(citations_data, {})
        
        # Format research process transparency
        research_process = format_detailed_research_process(result, show_content=False)
        
        # Format all documents dump
        all_docs_dump = format_all_documents(result, max_docs=50)
        
        research_time = time.time() - start_time
        
        return ResearchResponse(
            answer=result.get('answer', ''),
            legal_references=legal_refs,
            research_process=research_process,
            all_retrieved_documents=all_docs_dump,
            query=req.query,
            thinking_level=req.thinking_level,
            citations=documents,
            metadata=result.get('metadata', {}),
            research_time=research_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/chat")
async def conversational_rag(req: ChatRequest, request: Request):
    """
    Conversational RAG service - multi-turn legal consultation
    
    **Use Case**: Interactive legal consultation with context
    - Maintains conversation history via session_id
    - Supports streaming responses
    - Remembers previous queries and answers
    
    **Parameters**:
    - `query`: User question
    - `session_id`: Optional session for multi-turn context
    - `thinking_level`: Thinking depth (default: low for speed)
    - `stream`: Enable SSE streaming
    """
    from ..server import get_pipeline, get_conversation_manager
    from conversation import create_conversational_service
    
    try:
        pipeline = get_pipeline(request)
        manager = get_conversation_manager(request)
        
        # Create service
        service = create_conversational_service(
            pipeline=pipeline,
            conversation_manager=manager,
            current_provider='local'
        )
        
        # Get conversation context
        context = None
        if req.session_id:
            context = manager.get_context_for_query(req.session_id)
        
        # Handle streaming
        if req.stream:
            async def generate():
                full_answer = ""
                final_result = None
                
                # Build config dict with ALL parameters from request
                config_dict = {
                    'thinking_mode': req.thinking_level,
                    'final_top_k': req.top_k,
                    'temperature': req.temperature,
                    'max_new_tokens': req.max_tokens,
                    'research_team_size': req.team_size
                }
                
                for event in service.process_query(
                    message=req.query,
                    session_id=req.session_id or 'default',
                    config_dict=config_dict,
                    thinking_mode=req.thinking_level
                ):
                    event_type = event.get('type')
                    data = event.get('data', {})
                    
                    if event_type == 'progress':
                        yield f"data: {json.dumps({'type': 'progress', 'message': data.get('message', '')})}\n\n"
                    
                    elif event_type == 'streaming_chunk':
                        chunk = data.get('chunk', '')
                        full_answer += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    
                    elif event_type == 'thinking_chunk':
                        chunk = data.get('chunk', '')
                        yield f"data: {json.dumps({'type': 'thinking', 'content': chunk})}\n\n"
                    
                    elif event_type == 'final_result':
                        final_result = data
                        full_answer = data.get('answer', full_answer)
                
                # Send final message with detailed research info
                if final_result:
                    citations = final_result.get('citations', [])
                    legal_refs = format_detailed_sources_info(citations, {})
                    research_proc = format_detailed_research_process(final_result, show_content=False)
                    all_docs = format_all_documents(final_result, max_docs=20)
                    
                    done_data = {
                        'type': 'done', 
                        'answer': full_answer, 
                        'legal_references': legal_refs,
                        'research_process': research_proc,
                        'all_retrieved_documents': all_docs
                    }
                    yield f"data: {json.dumps(done_data)}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        else:
            # Non-streaming - update pipeline config first
            pipeline.update_config(
                final_top_k=req.top_k,
                temperature=req.temperature,
                max_new_tokens=req.max_tokens,
                research_team_size=req.team_size
            )
            
            result = pipeline.query(
                question=req.query,
                conversation_history=context,
                stream=False,
                thinking_mode=req.thinking_level
            )
            
            # Save to history
            if req.session_id:
                manager.add_turn(
                    session_id=req.session_id,
                    query=req.query,
                    answer=result.get('answer', ''),
                    metadata=result.get('metadata')
                )
            
            # Extract and format
            documents = extract_documents(result)
            legal_refs = format_detailed_sources_info(result.get('citations', []), {})
            research_proc = format_detailed_research_process(result, show_content=False)
            all_docs = format_all_documents(result, max_docs=50)
            
            return ChatResponse(
                answer=result.get('answer', ''),
                legal_references=legal_refs,
                research_process=research_proc,
                all_retrieved_documents=all_docs,
                query=req.query,
                session_id=req.session_id,
                citations=documents,
                metadata=result.get('metadata', {})
            )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
