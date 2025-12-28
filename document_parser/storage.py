"""
Document Storage - SQLite-based storage for extracted document text

Stores extracted text alongside session data for persistence and multi-user support.

File: document_parser/storage.py
"""

import sqlite3
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

from utils.logger_utils import get_logger


class DocumentStorage:
    """
    SQLite-based storage for extracted document text.
    
    Documents are tied to sessions and automatically expire.
    Uses the existing sessions.db database.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("DocumentStorage")
        self.config = config or {}
        
        # Get database path - use same location as session_storage
        if 'db_path' in self.config:
            self.db_path = self.config['db_path']
        else:
            data_dir = Path(".data")
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "sessions.db")
        
        # Settings
        self.document_ttl_hours = self.config.get('document_ttl_hours', 24)
        self.max_documents_per_session = self.config.get('max_documents_per_session', 5)
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize storage - create tables if not exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create uploaded_documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS uploaded_documents (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        original_size_bytes INTEGER,
                        content_hash TEXT,
                        extracted_text TEXT NOT NULL,
                        char_count INTEGER,
                        page_count INTEGER DEFAULT 1,
                        format TEXT NOT NULL,
                        extraction_method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                
                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_documents_session 
                    ON uploaded_documents(session_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_documents_hash 
                    ON uploaded_documents(content_hash)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_documents_expires 
                    ON uploaded_documents(expires_at)
                """)
                
                conn.commit()
                
            self._initialized = True
            self.logger.info(f"Document storage initialized: {self.db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
            return False
    
    def store_document(
        self,
        session_id: str,
        filename: str,
        extracted_text: str,
        format: str,
        original_size_bytes: int = 0,
        page_count: int = 1,
        extraction_method: str = None
    ) -> Dict[str, Any]:
        """
        Store extracted document text.
        
        Args:
            session_id: Session this document belongs to
            filename: Original filename
            extracted_text: Extracted text content
            format: File format (pdf, docx, etc.)
            original_size_bytes: Original file size
            page_count: Number of pages (if applicable)
            extraction_method: Method used for extraction
            
        Returns:
            Document record dict with id
        """
        # Check document limit
        current_count = self.get_document_count(session_id)
        if current_count >= self.max_documents_per_session:
            from .exceptions import DocumentLimitExceededError
            raise DocumentLimitExceededError(current_count, self.max_documents_per_session)
        
        # Generate ID and hash
        doc_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(extracted_text.encode()).hexdigest()
        
        # Check for duplicate
        existing = self.get_by_hash(session_id, content_hash)
        if existing:
            self.logger.info(f"Document already exists (hash match): {existing['id']}")
            return existing
        
        # Calculate expiry
        expires_at = datetime.now() + timedelta(hours=self.document_ttl_hours)
        
        # Store
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO uploaded_documents 
                    (id, session_id, filename, original_size_bytes, content_hash,
                     extracted_text, char_count, page_count, format, 
                     extraction_method, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, session_id, filename, original_size_bytes, content_hash,
                    extracted_text, len(extracted_text), page_count, format,
                    extraction_method, expires_at.isoformat()
                ))
                conn.commit()
            
            self.logger.info(f"Stored document: {filename} ({len(extracted_text)} chars)")
            
            return {
                'id': doc_id,
                'session_id': session_id,
                'filename': filename,
                'format': format,
                'char_count': len(extracted_text),
                'page_count': page_count,
                'created_at': datetime.now().isoformat(),
                'expires_at': expires_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to store document: {e}")
            from .exceptions import StorageError
            raise StorageError("insert", str(e))
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM uploaded_documents WHERE id = ?
                """, (document_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get document: {e}")
            return None
    
    def get_by_hash(self, session_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get document by content hash (for deduplication)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM uploaded_documents 
                    WHERE session_id = ? AND content_hash = ?
                """, (session_id, content_hash))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get document by hash: {e}")
            return None
    
    def get_session_documents(self, session_id: str, include_text: bool = False) -> List[Dict[str, Any]]:
        """
        Get all documents for a session.
        
        Args:
            session_id: Session ID
            include_text: Include extracted text (for pipeline use)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if include_text:
                    cursor.execute("""
                        SELECT id, filename, format, char_count, page_count, 
                               extracted_text, created_at, expires_at
                        FROM uploaded_documents 
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                    """, (session_id,))
                else:
                    cursor.execute("""
                        SELECT id, filename, format, char_count, page_count, 
                               created_at, expires_at
                        FROM uploaded_documents 
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                    """, (session_id,))
                    
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get session documents: {e}")
            return []
    
    def get_documents_text(self, document_ids: List[str]) -> List[Dict[str, str]]:
        """Get extracted text for multiple documents"""
        if not document_ids:
            return []
            
        try:
            placeholders = ','.join(['?' for _ in document_ids])
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT id, filename, extracted_text, char_count
                    FROM uploaded_documents 
                    WHERE id IN ({placeholders})
                """, document_ids)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get documents text: {e}")
            return []
    
    def get_document_count(self, session_id: str) -> int:
        """Get count of documents in session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM uploaded_documents WHERE session_id = ?
                """, (session_id,))
                return cursor.fetchone()[0]
        except:
            return 0
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM uploaded_documents WHERE id = ?
                """, (document_id,))
                conn.commit()
                deleted = cursor.rowcount > 0
                
            if deleted:
                self.logger.info(f"Deleted document: {document_id}")
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete document: {e}")
            return False
    
    def delete_session_documents(self, session_id: str) -> int:
        """Delete all documents for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM uploaded_documents WHERE session_id = ?
                """, (session_id,))
                conn.commit()
                count = cursor.rowcount
                
            self.logger.info(f"Deleted {count} documents for session: {session_id}")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to delete session documents: {e}")
            return 0
    
    def cleanup_expired(self) -> int:
        """Delete expired documents"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM uploaded_documents 
                    WHERE expires_at < datetime('now')
                """)
                conn.commit()
                count = cursor.rowcount
                
            if count > 0:
                self.logger.info(f"Cleaned up {count} expired documents")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired documents: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM uploaded_documents")
                total_docs = cursor.fetchone()[0]
                
                cursor.execute("SELECT SUM(char_count) FROM uploaded_documents")
                total_chars = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(DISTINCT session_id) FROM uploaded_documents")
                sessions_with_docs = cursor.fetchone()[0]
                
                return {
                    'total_documents': total_docs,
                    'total_characters': total_chars,
                    'sessions_with_documents': sessions_with_docs,
                    'max_per_session': self.max_documents_per_session,
                    'ttl_hours': self.document_ttl_hours
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
