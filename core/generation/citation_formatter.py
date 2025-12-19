"""
Citation Formatter for Indonesian Legal Documents
Formats legal citations according to Indonesian standards

File: core/generation/citation_formatter.py
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from utils.logger_utils import get_logger


class CitationFormatter:
    """
    Formats legal citations according to Indonesian legal citation standards
    Extracts and validates citation information from documents
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("CitationFormatter")
        self.config = config
        
        # Citation formats
        self.citation_styles = {
            'standard': self._format_standard,
            'short': self._format_short,
            'inline': self._format_inline,
            'bluebook': self._format_bluebook
        }
        
        self.logger.info("CitationFormatter initialized")
    
    def format_citation(
        self,
        record: Dict[str, Any],
        style: str = 'standard',
        include_article: bool = True
    ) -> str:
        """
        Format a single citation
        
        Args:
            record: Document record
            style: Citation style ('standard', 'short', 'inline', 'bluebook')
            include_article: Whether to include article/chapter info
            
        Returns:
            Formatted citation string
        """
        formatter = self.citation_styles.get(style, self._format_standard)
        return formatter(record, include_article)
    
    def format_citations(
        self,
        results: List[Dict[str, Any]],
        style: str = 'standard',
        numbered: bool = True
    ) -> str:
        """
        Format multiple citations
        
        Args:
            results: List of retrieved results
            style: Citation style
            numbered: Whether to number citations
            
        Returns:
            Formatted citations string
        """
        if not results:
            return ""
        
        citations = []
        
        for idx, result in enumerate(results, 1):
            record = result.get('record', {})
            citation = self.format_citation(record, style)
            
            if numbered:
                citations.append(f"{idx}. {citation}")
            else:
                citations.append(citation)
        
        return "\n".join(citations)
    
    def _format_standard(
        self,
        record: Dict[str, Any],
        include_article: bool = True
    ) -> str:
        """
        Standard Indonesian legal citation format
        Example: Undang-Undang Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik
        """
        parts = []
        
        # Regulation type
        reg_type = record.get('regulation_type', '').title()
        if reg_type and reg_type != 'Unknown':
            parts.append(reg_type)
        
        # Number
        reg_number = record.get('regulation_number', '')
        if reg_number and reg_number != 'N/A':
            parts.append(f"Nomor {reg_number}")
        
        # Year
        year = record.get('year', '')
        if year and year != 'N/A':
            parts.append(f"Tahun {year}")
        
        # About
        about = record.get('about', '')
        if about:
            # Truncate if too long
            if len(about) > 100:
                about = about[:100] + "..."
            parts.append(f"tentang {about}")
        
        citation = " ".join(parts)
        
        # Add article/chapter if requested
        if include_article:
            article_info = self._format_article_info(record)
            if article_info:
                citation += f", {article_info}"
        
        return citation
    
    def _format_short(
        self,
        record: Dict[str, Any],
        include_article: bool = True
    ) -> str:
        """
        Short citation format
        Example: UU No. 11/2008
        """
        reg_type = record.get('regulation_type', '')
        reg_number = record.get('regulation_number', 'N/A')
        year = record.get('year', 'N/A')
        
        # Abbreviate regulation type
        abbreviations = {
            'undang-undang': 'UU',
            'peraturan pemerintah': 'PP',
            'peraturan presiden': 'Perpres',
            'peraturan menteri': 'Permen',
            'peraturan daerah': 'Perda',
            'keputusan presiden': 'Keppres'
        }
        
        reg_abbr = abbreviations.get(reg_type.lower(), reg_type)
        
        citation = f"{reg_abbr} No. {reg_number}/{year}"
        
        if include_article:
            article = record.get('article', '')
            if article and article != 'N/A':
                citation += f" Pasal {article}"
        
        return citation
    
    def _format_inline(
        self,
        record: Dict[str, Any],
        include_article: bool = True
    ) -> str:
        """
        Inline citation format for use within text
        Example: (UU No. 11/2008, Pasal 27)
        """
        short_cite = self._format_short(record, include_article)
        return f"({short_cite})"
    
    def _format_bluebook(
        self,
        record: Dict[str, Any],
        include_article: bool = True
    ) -> str:
        """
        Bluebook-style citation (adapted for Indonesian law)
        Example: Law No. 11 of 2008 on Information and Electronic Transactions (Indon.)
        """
        reg_type = record.get('regulation_type', 'Regulation').title()
        reg_number = record.get('regulation_number', 'N/A')
        year = record.get('year', 'N/A')
        about = record.get('about', '')
        
        citation = f"{reg_type} No. {reg_number} of {year}"
        
        if about:
            # Truncate and translate "tentang" to "on"
            if len(about) > 80:
                about = about[:80] + "..."
            citation += f" on {about}"
        
        citation += " (Indon.)"
        
        if include_article:
            article = record.get('article', '')
            if article and article != 'N/A':
                citation += f" art. {article}"
        
        return citation
    
    def _format_article_info(self, record: Dict[str, Any]) -> str:
        """Format article/chapter information"""
        parts = []
        
        chapter = record.get('chapter', '')
        if chapter and chapter != 'N/A':
            parts.append(f"Bab {chapter}")
        
        article = record.get('article', '')
        if article and article != 'N/A':
            parts.append(f"Pasal {article}")
        
        return ", ".join(parts)
    
    def extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract legal citations from text
        
        Args:
            text: Text containing potential citations
            
        Returns:
            List of extracted citation dictionaries
        """
        citations = []
        
        # Pattern for Indonesian legal citations
        patterns = [
            # UU No. X/YYYY or UU Nomor X Tahun YYYY
            r'(?:UU|Undang-Undang)\s+(?:No\.|Nomor)\s*(\d+)(?:/|\s+Tahun\s+)(\d{4})',
            # PP No. X/YYYY
            r'(?:PP|Peraturan Pemerintah)\s+(?:No\.|Nomor)\s*(\d+)(?:/|\s+Tahun\s+)(\d{4})',
            # Perpres No. X/YYYY
            r'(?:Perpres|Peraturan Presiden)\s+(?:No\.|Nomor)\s*(\d+)(?:/|\s+Tahun\s+)(\d{4})',
            # Permen X No. Y/ZZZZ
            r'(?:Permen\w*|Peraturan Menteri\s+\w+)\s+(?:No\.|Nomor)\s*(\d+)(?:/|\s+Tahun\s+)(\d{4})',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                citation = {
                    'regulation_number': match.group(1),
                    'year': match.group(2),
                    'matched_text': match.group(0),
                    'start_pos': match.start(),
                    'end_pos': match.end()
                }
                citations.append(citation)
        
        self.logger.debug("Extracted citations from text", {
            "citations_found": len(citations)
        })
        
        return citations
    
    def validate_citation(self, citation_text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate citation format
        
        Args:
            citation_text: Citation string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check basic structure
        if not citation_text or len(citation_text) < 10:
            return False, "Citation too short"
        
        # Check for essential components
        has_number = bool(re.search(r'(?:No\.|Nomor)\s*\d+', citation_text, re.IGNORECASE))
        has_year = bool(re.search(r'(?:Tahun|/)\s*\d{4}', citation_text, re.IGNORECASE))
        
        if not has_number:
            return False, "Missing regulation number"
        
        if not has_year:
            return False, "Missing year"
        
        return True, None
    
    def format_inline_references(
        self,
        text: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Add inline references to text
        
        Args:
            text: Generated text
            results: Retrieved results for reference
            
        Returns:
            Text with inline references
        """
        # Create mapping of regulations to document numbers
        reg_map = {}
        for idx, result in enumerate(results, 1):
            record = result.get('record', {})
            reg_type = record.get('regulation_type', '').lower()
            reg_number = record.get('regulation_number', '')
            year = record.get('year', '')
            
            key = f"{reg_type}_{reg_number}_{year}"
            reg_map[key] = idx
        
        # Find mentions of regulations in text and add references
        for key, doc_num in reg_map.items():
            reg_type, reg_number, year = key.split('_')
            
            # Pattern to match regulation mentions
            pattern = rf'\b{re.escape(reg_type)}.*?{re.escape(reg_number)}.*?{re.escape(year)}\b'
            
            # Replace with citation
            replacement = f"\\g<0> [Dok. {doc_num}]"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def generate_bibliography(
        self,
        results: List[Dict[str, Any]],
        style: str = 'standard'
    ) -> str:
        """
        Generate bibliography section
        
        Args:
            results: Retrieved results
            style: Citation style
            
        Returns:
            Formatted bibliography
        """
        bibliography = ["DAFTAR PUSTAKA", "=" * 50, ""]
        
        # Format citations
        citations = self.format_citations(results, style, numbered=True)
        bibliography.append(citations)
        
        return "\n".join(bibliography)
    
    def format_reference_list(
        self,
        results: List[Dict[str, Any]],
        max_items: int = 10
    ) -> str:
        """
        Format reference list with metadata
        
        Args:
            results: Retrieved results
            max_items: Maximum items to include
            
        Returns:
            Formatted reference list
        """
        reference_list = ["REFERENSI", "=" * 50, ""]
        
        for idx, result in enumerate(results[:max_items], 1):
            record = result.get('record', {})
            
            # Standard citation
            citation = self.format_citation(record, 'standard')
            reference_list.append(f"{idx}. {citation}")
            
            # Add metadata
            enacting_body = record.get('enacting_body', '')
            if enacting_body and enacting_body != 'Unknown':
                reference_list.append(f"   Lembaga: {enacting_body}")
            
            effective_date = record.get('effective_date', '')
            if effective_date:
                reference_list.append(f"   Tanggal Berlaku: {effective_date}")
            
            # Add score
            score = result.get('final_score', result.get('rerank_score', 0))
            reference_list.append(f"   Skor Relevansi: {score:.3f}")
            
            reference_list.append("")
        
        return "\n".join(reference_list)