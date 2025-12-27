"""
Document Routes - API endpoints for document upload and management

Endpoints:
- POST /documents/upload - Upload and parse document
- POST /documents/url - Extract content from URL
- GET /documents - List session documents
- GET /documents/{document_id} - Get specific document
- DELETE /documents/{document_id} - Delete document
- DELETE /documents - Clear all session documents

File: api/routes/documents.py
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import os

from utils.logger_utils import get_logger

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    success: bool
    document_id: str
    filename: str
    format: str
    char_count: int
    page_count: int
    preview: str
    message: str


class URLExtractRequest(BaseModel):
    """Request for URL extraction"""
    url: str = Field(..., description="URL to extract content from")
    session_id: str = Field(..., description="Session ID")


class DocumentInfo(BaseModel):
    """Document information"""
    id: str
    filename: str
    format: str
    char_count: int
    page_count: int
    created_at: str
    preview: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for document list"""
    success: bool
    documents: List[DocumentInfo]
    total_count: int
    session_id: str


class DocumentDetailResponse(BaseModel):
    """Response for document detail"""
    success: bool
    document: Dict[str, Any]


class DeleteResponse(BaseModel):
    """Response for delete operations"""
    success: bool
    message: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_document_parser_or_error(request: Request):
    """Get document parser or raise error if not initialized"""
    if not getattr(request.app.state, 'document_parser_initialized', False):
        raise HTTPException(
            status_code=503,
            detail="Document parser not initialized. Upload feature unavailable."
        )
    import document_parser
    return document_parser


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload and parse a document file.
    
    Supported formats: PDF, DOCX, DOC, TXT, MD, HTML, JSON, CSV, XML, RTF, images (OCR)
    
    **Parameters**:
    - `file`: Document file to upload
    - `session_id`: Session ID to associate document with
    
    **Returns**: Document ID, metadata, and text preview
    """
    doc_parser = get_document_parser_or_error(request)
    
    logger.info(f"Document upload request", {
        "filename": file.filename,
        "session_id": session_id,
        "content_type": file.content_type
    })
    
    try:
        # Get parser instance
        parser = doc_parser.get_parser()
        
        # Save uploaded file to temp location
        suffix = os.path.splitext(file.filename)[1] if file.filename else ''
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Parse the file
            doc_info = parser.parse_file(tmp_path, session_id, original_filename=file.filename)
            
            logger.success(f"Document parsed successfully", {
                "document_id": doc_info.get('id'),
                "char_count": doc_info.get('char_count')
            })
            
            return DocumentUploadResponse(
                success=True,
                document_id=doc_info['id'],
                filename=doc_info['filename'],
                format=doc_info['format'],
                char_count=doc_info['char_count'],
                page_count=doc_info.get('page_count', 1),
                preview=doc_info.get('preview', '')[:500],
                message=f"Document '{file.filename}' uploaded successfully"
            )
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except doc_parser.DocumentLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except doc_parser.FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except doc_parser.UnsupportedFormatError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except doc_parser.ExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/documents/url", response_model=DocumentUploadResponse)
async def extract_from_url(
    request: Request,
    req: URLExtractRequest
):
    """
    Extract content from a URL and store as document.
    
    Supports: Web pages (HTML), PDF links, DOCX links, JSON APIs
    
    **Parameters**:
    - `url`: URL to extract content from
    - `session_id`: Session ID to associate document with
    
    **Security**: Blocks private IPs, has timeout and size limits
    """
    doc_parser = get_document_parser_or_error(request)
    
    if not doc_parser.is_url_extraction_enabled():
        raise HTTPException(
            status_code=503,
            detail="URL extraction is disabled"
        )
    
    logger.info(f"URL extraction request", {
        "url": req.url[:100],
        "session_id": req.session_id
    })
    
    try:
        # Extract from URL
        doc_info = doc_parser.extract_from_url(req.url, req.session_id)
        
        logger.success(f"URL content extracted", {
            "document_id": doc_info.get('id'),
            "char_count": doc_info.get('char_count')
        })
        
        return DocumentUploadResponse(
            success=True,
            document_id=doc_info['id'],
            filename=req.url,
            format='url',
            char_count=doc_info.get('char_count', 0),
            page_count=doc_info.get('page_count', 1),
            preview=doc_info.get('preview', '')[:500],
            message=f"Content extracted from URL successfully"
        )
        
    except doc_parser.URLBlockedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except doc_parser.URLExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except doc_parser.DocumentLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"URL extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    session_id: str
):
    """
    List all documents for a session.
    
    **Parameters**:
    - `session_id`: Session ID to list documents for
    
    **Returns**: List of document metadata
    """
    doc_parser = get_document_parser_or_error(request)
    
    try:
        storage = doc_parser.get_storage()
        documents = storage.get_session_documents(session_id)
        
        doc_infos = [
            DocumentInfo(
                id=doc['id'],
                filename=doc['filename'],
                format=doc['format'],
                char_count=doc.get('char_count', len(doc.get('extracted_text', ''))),
                page_count=doc.get('page_count', 1),
                created_at=doc.get('created_at', ''),
                preview=doc.get('extracted_text', '')[:200] if doc.get('extracted_text') else None
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            success=True,
            documents=doc_infos,
            total_count=len(doc_infos),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    request: Request,
    document_id: str
):
    """
    Get a specific document by ID.
    
    **Returns**: Full document content and metadata
    """
    doc_parser = get_document_parser_or_error(request)
    
    try:
        storage = doc_parser.get_storage()
        document = storage.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentDetailResponse(
            success=True,
            document=document
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    request: Request,
    document_id: str
):
    """
    Delete a specific document.
    """
    doc_parser = get_document_parser_or_error(request)
    
    try:
        storage = doc_parser.get_storage()
        deleted = storage.delete_document(document_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DeleteResponse(
            success=True,
            message=f"Document {document_id} deleted"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents", response_model=DeleteResponse)
async def clear_session_documents(
    request: Request,
    session_id: str
):
    """
    Delete all documents for a session.
    """
    doc_parser = get_document_parser_or_error(request)
    
    try:
        storage = doc_parser.get_storage()
        storage.delete_session_documents(session_id)
        
        return DeleteResponse(
            success=True,
            message=f"All documents for session {session_id} deleted"
        )
        
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
