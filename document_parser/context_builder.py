"""
Context Builder - Build prompt context from uploaded documents

Combines uploaded document text with RAG context for LLM prompts.

File: document_parser/context_builder.py
"""

from typing import List, Dict, Any, Optional
from utils.logger_utils import get_logger


class DocumentContextBuilder:
    """
    Builds context from uploaded documents for use in LLM prompts.
    
    Handles:
    - Multiple documents
    - Character limits
    - Context formatting
    - Priority ordering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("DocumentContextBuilder")
        self.config = config or {}
        self.max_total_chars = self.config.get('max_chars_total', 100000)
        self.max_docs_in_context = self.config.get('max_docs_in_context', 5)
    
    def build_context(
        self,
        documents: List[Dict[str, str]],
        max_chars: Optional[int] = None
    ) -> str:
        """
        Build context string from document list.
        
        Args:
            documents: List of dicts with 'filename' and 'extracted_text'
            max_chars: Override max chars limit
            
        Returns:
            Formatted context string for prompt
        """
        if not documents:
            return ""
        
        max_chars = max_chars or self.max_total_chars
        
        # Limit number of documents
        docs_to_use = documents[:self.max_docs_in_context]
        
        # Calculate available chars per document
        total_header_overhead = sum(
            len(f"\n--- Dokumen: {d.get('filename', 'unknown')} ---\n") 
            for d in docs_to_use
        )
        available_chars = max_chars - total_header_overhead
        chars_per_doc = available_chars // len(docs_to_use)
        
        context_parts = []
        
        for doc in docs_to_use:
            filename = doc.get('filename', 'Dokumen')
            text = doc.get('extracted_text', '')
            
            # Truncate if needed
            if len(text) > chars_per_doc:
                text = text[:chars_per_doc]
                text += "\n[... Konten dipotong ...]"
            
            part = f"\n--- Dokumen yang Diunggah: {filename} ---\n{text}\n"
            context_parts.append(part)
        
        return '\n'.join(context_parts)
    
    def build_prompt_section(
        self,
        documents: List[Dict[str, str]],
        header: str = "DOKUMEN YANG DIUNGGAH PENGGUNA",
        max_chars: Optional[int] = None
    ) -> str:
        """
        Build a formatted prompt section for uploaded documents.
        
        Args:
            documents: List of document dicts
            header: Section header
            max_chars: Max chars limit
            
        Returns:
            Formatted section for prompt
        """
        if not documents:
            return ""
        
        context = self.build_context(documents, max_chars)
        
        return f"""
=== {header} ===
Pengguna telah mengunggah {len(documents)} dokumen berikut untuk dianalisis:
{context}
=== AKHIR DOKUMEN PENGGUNA ===

Gunakan informasi dari dokumen yang diunggah di atas bersama dengan dokumen hukum yang ditemukan untuk menjawab pertanyaan pengguna.
"""
    
    def get_document_summary(self, documents: List[Dict[str, Any]]) -> str:
        """
        Create a brief summary of uploaded documents.
        
        For display in UI or logs.
        """
        if not documents:
            return "Tidak ada dokumen yang diunggah"
        
        summaries = []
        for doc in documents:
            filename = doc.get('filename', 'unknown')
            char_count = doc.get('char_count', len(doc.get('extracted_text', '')))
            page_count = doc.get('page_count', 1)
            
            # Format size
            if char_count > 10000:
                size_str = f"{char_count/1000:.1f}K karakter"
            else:
                size_str = f"{char_count} karakter"
            
            summaries.append(f"📄 {filename} ({page_count} hal, {size_str})")
        
        return "\n".join(summaries)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses rough approximation: ~4 chars per token for Indonesian.
        """
        return len(text) // 4
    
    def should_chunk_documents(
        self, 
        documents: List[Dict[str, str]], 
        max_context_tokens: int = 8000
    ) -> bool:
        """
        Check if documents are too large and need chunking.
        
        Args:
            documents: Document list
            max_context_tokens: Max tokens for context
            
        Returns:
            True if chunking is recommended
        """
        total_chars = sum(len(d.get('extracted_text', '')) for d in documents)
        estimated_tokens = self.estimate_tokens(total_chars)
        
        return estimated_tokens > max_context_tokens
