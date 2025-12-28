"""
Document Parser Exceptions

Custom exceptions for document parsing operations.

File: document_parser/exceptions.py
"""


class DocumentParserError(Exception):
    """Base exception for document parser module"""
    pass


class UnsupportedFormatError(DocumentParserError):
    """Raised when file format is not supported"""
    
    def __init__(self, extension: str, supported: list = None):
        self.extension = extension
        self.supported = supported or []
        msg = f"Unsupported file format: {extension}"
        if supported:
            msg += f". Supported: {', '.join(supported)}"
        super().__init__(msg)


class ExtractionError(DocumentParserError):
    """Raised when text extraction fails"""
    
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to extract text from '{filename}': {reason}")


class FileTooLargeError(DocumentParserError):
    """Raised when file exceeds size limit"""
    
    def __init__(self, filename: str, size_bytes: int, max_bytes: int):
        self.filename = filename
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        size_mb = size_bytes / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(
            f"File '{filename}' ({size_mb:.1f}MB) exceeds limit ({max_mb:.1f}MB)"
        )


class DocumentLimitExceededError(DocumentParserError):
    """Raised when session document limit is exceeded"""
    
    def __init__(self, current_count: int, max_count: int):
        self.current_count = current_count
        self.max_count = max_count
        super().__init__(
            f"Document limit exceeded: {current_count}/{max_count} documents in session"
        )


class OCRError(DocumentParserError):
    """Raised when OCR processing fails"""
    
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"OCR failed for '{filename}': {reason}")


class OCRNotAvailableError(DocumentParserError):
    """Raised when OCR is requested but not available"""
    
    def __init__(self, provider: str, install_hint: str = None):
        self.provider = provider
        self.install_hint = install_hint
        msg = f"OCR provider '{provider}' is not available"
        if install_hint:
            msg += f". Install: {install_hint}"
        super().__init__(msg)


class StorageError(DocumentParserError):
    """Raised when document storage operation fails"""
    
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Storage {operation} failed: {reason}")


class URLExtractionError(DocumentParserError):
    """Raised when URL content extraction fails"""
    
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to extract content from URL '{url}': {reason}")


class URLValidationError(DocumentParserError):
    """Raised when URL fails security validation"""
    
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"URL validation failed for '{url}': {reason}")


class URLBlockedError(DocumentParserError):
    """Raised when URL is blocked (private IP, blocked domain, etc.)"""
    
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"URL blocked '{url}': {reason}")
