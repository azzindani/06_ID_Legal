"""
Search Routes

Endpoints for document search functionality.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(10, ge=1, le=50, description="Maximum results")
    semantic_weight: float = Field(0.7, ge=0, le=1, description="Semantic search weight")
    keyword_weight: float = Field(0.3, ge=0, le=1, description="Keyword search weight")


class SearchResult(BaseModel):
    id: str
    content: str
    regulation_type: str
    regulation_number: str
    year: str
    about: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search legal documents

    Performs hybrid semantic and keyword search on the legal document corpus.
    """
    from ..server import get_pipeline
    import time

    start_time = time.time()

    try:
        pipeline = get_pipeline()

        # Use pipeline's internal search
        result = pipeline.query(request.query)

        # Extract search results from metadata
        sources = result.get('metadata', {}).get('sources', [])

        search_results = []
        for i, source in enumerate(sources[:request.max_results]):
            search_results.append(SearchResult(
                id=source.get('id', f'doc-{i}'),
                content=source.get('content', '')[:500],
                regulation_type=source.get('regulation_type', 'N/A'),
                regulation_number=source.get('regulation_number', 'N/A'),
                year=source.get('year', 'N/A'),
                about=source.get('about', 'N/A'),
                score=source.get('score', 0.0),
                rank=i + 1
            ))

        search_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time=search_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
