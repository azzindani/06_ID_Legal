"""
Legal RAG API Client

HTTP client for interacting with the Legal RAG API server.
Supports both sync and streaming operations.

File: ui/services/api_client.py
"""

import os
import json
import time
import requests
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass
from utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Legal document from retrieval"""
    regulation_type: str
    regulation_number: str
    year: str
    about: str
    score: float
    content_preview: Optional[str] = None
    chapter: Optional[str] = None
    article: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        return cls(
            regulation_type=data.get('regulation_type', 'N/A'),
            regulation_number=data.get('regulation_number', 'N/A'),
            year=data.get('year', 'N/A'),
            about=data.get('about', 'N/A'),
            score=data.get('score', 0.0),
            content_preview=data.get('content_preview'),
            chapter=data.get('chapter'),
            article=data.get('article')
        )


@dataclass 
class HealthStatus:
    """API health status"""
    healthy: bool
    ready: bool
    message: str
    components: Dict[str, bool]


class LegalRAGAPIClient:
    """
    HTTP client for Legal RAG API
    
    Supports:
    - Document retrieval (search only)
    - Deep research (full RAG)
    - Conversational chat with sessions
    - Streaming responses via SSE
    """
    
    def __init__(
        self, 
        base_url: str = "http://127.0.0.1:8000/api/v1",
        api_key: Optional[str] = None,
        timeout: int = 600
    ):
        """
        Initialize API client
        
        Args:
            base_url: API server URL (default: localhost:8000)
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.environ.get('LEGAL_API_KEY', '')
        self.timeout = timeout
        
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }
        
        logger.info(f"API client initialized | base_url={self.base_url}")
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        stream: bool = False
    ) -> requests.Response:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to API at {self.base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise PermissionError("Invalid API key")
            elif e.response.status_code == 429:
                raise RuntimeError("Rate limit exceeded")
            else:
                raise RuntimeError(f"API error: {e.response.text}")
    
    # =========================================================================
    # Health & Status
    # =========================================================================
    
    def health_check(self) -> HealthStatus:
        """Check API health status"""
        try:
            response = self._request('GET', '/health')
            data = response.json()
            
            ready_response = self._request('GET', '/ready')
            ready_data = ready_response.json()
            
            return HealthStatus(
                healthy=data.get('status') == 'healthy',
                ready=ready_data.get('ready', False),
                message=ready_data.get('message', ''),
                components=ready_data.get('components', {})
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                ready=False,
                message=str(e),
                components={}
            )
    
    def is_ready(self) -> bool:
        """Quick check if API is ready"""
        try:
            response = self._request('GET', '/ready')
            return response.json().get('ready', False)
        except:
            return False
    
    # =========================================================================
    # Document Retrieval (Search Only)
    # =========================================================================
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        min_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Retrieve documents without LLM generation
        
        Args:
            query: Search query
            top_k: Number of results (1-10)
            min_score: Minimum relevance score
            
        Returns:
            Dict with documents, search_time, metadata
        """
        response = self._request('POST', '/rag/retrieve', {
            'query': query,
            'top_k': top_k,
            'min_score': min_score
        })
        
        data = response.json()
        
        return {
            'query': data.get('query', query),
            'documents': [Document.from_dict(d) for d in data.get('documents', [])],
            'total_retrieved': data.get('total_retrieved', 0),
            'search_time': data.get('search_time', 0),
            'metadata': data.get('metadata', {})
        }
    
    # =========================================================================
    # Deep Research (Full RAG)
    # =========================================================================
    
    def research(
        self,
        query: str,
        thinking_level: str = 'medium',
        team_size: int = 3
    ) -> Dict[str, Any]:
        """
        Deep research with full RAG pipeline
        
        Args:
            query: Legal question
            thinking_level: low, medium, high
            team_size: 1-5 research agents
            
        Returns:
            Dict with answer, references, research process
        """
        response = self._request('POST', '/rag/research', {
            'query': query,
            'thinking_level': thinking_level,
            'team_size': team_size
        })
        
        return response.json()
    
    # =========================================================================
    # Conversational Chat
    # =========================================================================
    
    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        thinking_level: str = 'low',
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Conversational chat (non-streaming)
        
        Args:
            query: User question
            session_id: Session ID for context
            thinking_level: low, medium, high
            stream: Must be False for this method
            
        Returns:
            Dict with answer, session_id, references
        """
        response = self._request('POST', '/rag/chat', {
            'query': query,
            'session_id': session_id,
            'thinking_level': thinking_level,
            'stream': False
        })
        
        return response.json()
    
    def chat_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        thinking_level: str = 'low'
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming chat via Server-Sent Events
        
        Args:
            query: User question
            session_id: Session ID for context
            thinking_level: low, medium, high
            
        Yields:
            SSE events: {type: 'progress'|'thinking'|'chunk'|'done', ...}
        """
        response = self._request('POST', '/rag/chat', {
            'query': query,
            'session_id': session_id,
            'thinking_level': thinking_level,
            'stream': True
        }, stream=True)
        
        for line in response.iter_lines():
            if line:
                content = line.decode('utf-8')
                if content.startswith('data: '):
                    try:
                        event = json.loads(content[6:])
                        yield event
                    except json.JSONDecodeError:
                        continue
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        response = self._request('POST', '/sessions', {
            'session_id': session_id
        })
        return response.json().get('session_id', '')
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        response = self._request('GET', '/sessions')
        return response.json()
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session summary"""
        response = self._request('GET', f'/sessions/{session_id}')
        return response.json()
    
    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        url = f'/sessions/{session_id}/history'
        if max_turns:
            url += f'?max_turns={max_turns}'
        response = self._request('GET', url)
        return response.json().get('turns', [])
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        try:
            self._request('DELETE', f'/sessions/{session_id}')
            return True
        except:
            return False
    
    def export_session(
        self, 
        session_id: str, 
        format: str = 'md'
    ) -> str:
        """Export session to specified format"""
        response = self._request('POST', f'/sessions/{session_id}/export', {
            'format': format
        })
        return response.json().get('content', '')


# =============================================================================
# Factory function
# =============================================================================

def create_api_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> LegalRAGAPIClient:
    """
    Create API client with environment defaults
    
    Environment variables:
        LEGAL_API_URL: API base URL
        LEGAL_API_KEY: API key
    """
    url = base_url or os.environ.get('LEGAL_API_URL', 'http://127.0.0.1:8000/api/v1')
    key = api_key or os.environ.get('LEGAL_API_KEY', '')
    
    return LegalRAGAPIClient(base_url=url, api_key=key)
