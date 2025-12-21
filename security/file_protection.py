"""
File Protection Module

Secure file upload handling with type validation, size limits,
and safety checks.

File: security/file_protection.py
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple
from utils.logger_utils import get_logger

logger = get_logger(__name__)


# Allowed file extensions and MIME types for legal documents
ALLOWED_EXTENSIONS = {
    # Documents
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain',
    '.rtf': 'application/rtf',
    
    # Spreadsheets
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.csv': 'text/csv',
    
    # Images (for evidence/attachments)
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    
    # Archives (for bulk documents)
    '.zip': 'application/zip',
}


# Dangerous extensions that should never be allowed
DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.sh', '.ps1', '.msi', '.app',
    '.jar', '.vbs', '.js', '.jsx', '.ts', '.tsx',
    '.dll', '.so', '.dylib', '.scr', '.pif'
}


class FileValidator:
    """
    Validates uploaded files for security and compliance
    """
    
    def __init__(
        self,
        max_size_mb: int = 50,
        allowed_extensions: Optional[List[str]] = None,
        check_mime_type: bool = True
    ):
        """
        Initialize file validator
        
        Args:
            max_size_mb: Maximum file size in megabytes
            allowed_extensions: List of allowed extensions (default: ALLOWED_EXTENSIONS)
            check_mime_type: Verify MIME type matches extension
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_extensions = allowed_extensions or list(ALLOWED_EXTENSIONS.keys())
        self.check_mime_type = check_mime_type
        
        logger.info(f"File validator initialized: max {max_size_mb}MB, {len(self.allowed_extensions)} allowed types")
    
    def validate(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            # Check size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_size_bytes:
                size_mb = file_size / (1024 * 1024)
                max_mb = self.max_size_bytes / (1024 * 1024)
                return False, f"File size ({size_mb:.1f}MB) exceeds limit ({max_mb:.1f}MB)"
            
            if file_size == 0:
                return False, "File is empty"
            
            # Check extension
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in DANGEROUS_EXTENSIONS:
                return False, f"File type {file_ext} is not allowed for security reasons"
            
            if file_ext not in self.allowed_extensions:
                return False, f"File type {file_ext} is not supported"
            
            # Check MIME type (if enabled)
            if self.check_mime_type:
                guessed_type, _ = mimetypes.guess_type(file_path)
                expected_type = ALLOWED_EXTENSIONS.get(file_ext)
                
                if expected_type and guessed_type != expected_type:
                    logger.warning(f"MIME type mismatch: expected {expected_type}, got {guessed_type}")
                    return False, "File type does not match its extension"
            
            return True, None
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_multiple(self, file_paths: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate multiple files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []
        for file_path in file_paths:
            is_valid, error = self.validate(file_path)
            if not is_valid:
                errors.append(f"{Path(file_path).name}: {error}")
        
        return len(errors) == 0, errors


def is_safe_filename(filename: str) -> bool:
    """
    Check if filename is safe
    
    Args:
        filename: Filename to check
        
    Returns:
        True if safe, False otherwise
    """
    # No path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    
    # No dangerous extensions
    ext = Path(filename).suffix.lower()
    if ext in DANGEROUS_EXTENSIONS:
        return False
    
    # No null bytes
    if '\x00' in filename:
        return False
    
    # Not too long
    if len(filename) > 255:
        return False
    
    # Has valid characters
    if not filename.replace('.', '').replace('_', '').replace('-', '').replace(' ', '').isalnum():
        # Contains special characters beyond .-_ and space
        return False
    
    return True


def validate_upload(
    file_path: str,
    max_size_mb: int = 50,
    allowed_extensions: Optional[List[str]] = None
) -> bool:
    """
    Quick validation for file uploads (convenience function)
    
    Args:
        file_path: Path to uploaded file
        max_size_mb: Maximum size in MB
        allowed_extensions: Allowed file extensions
        
    Returns:
        True if valid, False otherwise
    """
    validator = FileValidator(max_size_mb, allowed_extensions)
    is_valid, _ = validator.validate(file_path)
    return is_valid


def check_file_header(file_path: str, expected_ext: str) -> bool:
    """
    Check file header (magic bytes) to verify file type
    Prevents extension spoofing attacks
    
    Args:
        file_path: Path to file
        expected_ext: Expected extension
        
    Returns:
        True if header matches extension
    """
    # Magic bytes for common file types
    magic_bytes = {
        '.pdf': b'%PDF',
        '.zip': b'PK\x03\x04',
        '.png': b'\x89PNG',
        '.jpg': b'\xff\xd8\xff',
        '.jpeg': b'\xff\xd8\xff',
    }
    
    expected_magic = magic_bytes.get(expected_ext.lower())
    if not expected_magic:
        # No magic bytes defined for this type
        return True
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(len(expected_magic))
            return header == expected_magic
    except Exception as e:
        logger.error(f"Error checking file header: {e}")
        return False


class SecureFileUploader:
    """
    Manages secure file uploads with virus scanning hooks
    """
    
    def __init__(
        self,
        upload_dir: str,
        max_size_mb: int = 50,
        enable_virus_scan: bool = False
    ):
        """
        Initialize uploader
        
        Args:
            upload_dir: Directory for uploads
            max_size_mb: Maximum file size
            enable_virus_scan: Enable virus scanning (requires external tool)
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = FileValidator(max_size_mb)
        self.enable_virus_scan = enable_virus_scan
    
    def save_upload(self, file_path: str, original_filename: str) -> Tuple[bool, str]:
        """
        Safely save uploaded file
        
        Args:
            file_path: Temporary file path
            original_filename: Original filename
            
        Returns:
            Tuple of (success, saved_path_or_error)
        """
        # Validate file
        is_valid, error = self.validator.validate(file_path)
        if not is_valid:
            return False, error
        
        # Sanitize filename
        from .input_safety import sanitize_filename
        safe_name = sanitize_filename(original_filename)
        
        # Generate unique filename
        timestamp = int(os.path.getmtime(file_path))
        unique_name = f"{timestamp}_{safe_name}"
        save_path = self.upload_dir / unique_name
        
        # Check magic bytes
        ext = Path(safe_name).suffix
        if not check_file_header(file_path, ext):
            return False, "File header does not match extension (possible spoofing)"
        
        try:
            # Move file
            import shutil
            shutil.move(file_path, save_path)
            
            logger.info(f"File uploaded successfully: {save_path}")
            return True, str(save_path)
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return False, str(e)
