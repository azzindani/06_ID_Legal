"""
Image Extractor - Extract text from images using OCR

Supports Tesseract (default) and EasyOCR providers.

File: document_parser/extractors/image_extractor.py
"""

from typing import Dict, Any, Optional
from pathlib import Path
from .base import BaseExtractor
from utils.logger_utils import get_logger


class ImageExtractor(BaseExtractor):
    """Extract text from images using OCR"""
    
    SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    EXTRACTOR_NAME = "image"
    
    def __init__(self, provider: str = 'tesseract', languages: list = None):
        """
        Initialize image extractor.
        
        Args:
            provider: OCR provider ('tesseract' or 'easyocr')
            languages: Languages to use for OCR (default: ['ind', 'eng'])
        """
        super().__init__()
        self.logger = get_logger("ImageExtractor")
        self.provider = provider
        self.languages = languages or ['ind', 'eng']
        self._tesseract_available = None
        self._easyocr_available = None
    
    def _check_dependencies(self) -> bool:
        """Check if OCR providers are available"""
        if self.provider == 'tesseract':
            return self._check_tesseract()
        elif self.provider == 'easyocr':
            return self._check_easyocr()
        return self._check_tesseract() or self._check_easyocr()
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available"""
        if self._tesseract_available is not None:
            return self._tesseract_available
        
        try:
            import pytesseract
            from PIL import Image
            # Try to get tesseract version (checks if binary is installed)
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception:
            self._tesseract_available = False
        
        return self._tesseract_available
    
    def _check_easyocr(self) -> bool:
        """Check if EasyOCR is available"""
        if self._easyocr_available is not None:
            return self._easyocr_available
        
        try:
            import easyocr
            self._easyocr_available = True
        except ImportError:
            self._easyocr_available = False
        
        return self._easyocr_available
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from image using OCR.
        
        Uses configured provider or falls back to available one.
        """
        self.validate_file(file_path)
        
        # Try preferred provider first
        if self.provider == 'tesseract' and self._check_tesseract():
            return self._extract_with_tesseract(file_path)
        elif self.provider == 'easyocr' and self._check_easyocr():
            return self._extract_with_easyocr(file_path)
        
        # Fallback
        if self._check_tesseract():
            return self._extract_with_tesseract(file_path)
        elif self._check_easyocr():
            return self._extract_with_easyocr(file_path)
        
        from ..exceptions import OCRNotAvailableError
        raise OCRNotAvailableError(
            self.provider,
            "pip install pytesseract (requires tesseract binary) or pip install easyocr"
        )
    
    def _extract_with_tesseract(self, file_path: str) -> Dict[str, Any]:
        """Extract text using Tesseract OCR"""
        import pytesseract
        from PIL import Image
        
        # Map language codes for tesseract
        lang_map = {'ind': 'ind', 'eng': 'eng', 'id': 'ind', 'en': 'eng'}
        tesseract_langs = [lang_map.get(l, l) for l in self.languages]
        lang_str = '+'.join(tesseract_langs)
        
        try:
            image = Image.open(file_path)
            
            # Preprocess image for better OCR
            image = self._preprocess_image(image)
            
            # Run OCR
            text = pytesseract.image_to_string(image, lang=lang_str)
            
            # Get image dimensions for metadata
            width, height = image.size
            
            return {
                'text': text.strip(),
                'page_count': 1,
                'method': 'tesseract',
                'metadata': {
                    'provider': 'tesseract',
                    'languages': tesseract_langs,
                    'image_size': f"{width}x{height}",
                    'format': Path(file_path).suffix
                }
            }
            
        except Exception as e:
            from ..exceptions import OCRError
            raise OCRError(file_path, str(e))
    
    def _extract_with_easyocr(self, file_path: str) -> Dict[str, Any]:
        """Extract text using EasyOCR"""
        import easyocr
        
        # Map language codes for EasyOCR
        lang_map = {'ind': 'id', 'eng': 'en', 'id': 'id', 'en': 'en'}
        easyocr_langs = [lang_map.get(l, l) for l in self.languages]
        
        # Remove duplicates while preserving order
        seen = set()
        easyocr_langs = [l for l in easyocr_langs if not (l in seen or seen.add(l))]
        
        try:
            # Create reader (will download models if needed)
            reader = easyocr.Reader(easyocr_langs, gpu=False)
            
            # Read text
            results = reader.readtext(file_path)
            
            # Extract text from results
            text_lines = [text for _, text, _ in results]
            text = '\n'.join(text_lines)
            
            return {
                'text': text.strip(),
                'page_count': 1,
                'method': 'easyocr',
                'metadata': {
                    'provider': 'easyocr',
                    'languages': easyocr_langs,
                    'regions_detected': len(results)
                }
            }
            
        except Exception as e:
            from ..exceptions import OCRError
            raise OCRError(file_path, str(e))
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        from PIL import Image, ImageFilter, ImageEnhance
        
        # Convert to RGB if necessary
        if image.mode not in ('L', 'RGB'):
            image = image.convert('RGB')
        
        # Convert to grayscale
        if image.mode == 'RGB':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        return image
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get available OCR providers"""
        return {
            'tesseract': self._check_tesseract(),
            'easyocr': self._check_easyocr()
        }
