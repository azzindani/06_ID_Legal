"""
Prompt Builder for Indonesian Legal RAG System
Constructs context-aware prompts with retrieved documents and citations

File: core/generation/prompt_builder.py
"""

from typing import Dict, Any, List, Optional
from logger_utils import get_logger
from config import SYSTEM_PROMPT


class PromptBuilder:
    """
    Builds optimized prompts for legal question answering
    Handles context formatting, citation preparation, and prompt templates
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("PromptBuilder")
        self.config = config
        
        # System prompt from config
        self.system_prompt = config.get('system_prompt', SYSTEM_PROMPT)
        
        # Prompt templates
        self.templates = {
            'rag_qa': self._get_rag_qa_template(),
            'followup': self._get_followup_template(),
            'clarification': self._get_clarification_template(),
            'procedural': self._get_procedural_template(),
            'comparison': self._get_comparison_template()
        }
        
        self.logger.info("PromptBuilder initialized", {
            "templates": len(self.templates)
        })
    
    def build_prompt(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        template_type: str = 'rag_qa'
    ) -> str:
        """
        Build complete prompt with system instructions, context, and query
        
        Args:
            query: User query
            retrieved_results: Retrieved and ranked documents
            query_analysis: Optional query analysis from QueryDetector
            conversation_history: Optional conversation context
            template_type: Type of prompt template to use
            
        Returns:
            Complete formatted prompt
        """
        self.logger.info("Building prompt", {
            "query_length": len(query),
            "num_results": len(retrieved_results),
            "template": template_type
        })
        
        # Format context from retrieved results
        context = self._format_context(retrieved_results)
        
        # Build conversation context if provided
        conv_context = ""
        if conversation_history:
            conv_context = self._format_conversation_history(conversation_history)
        
        # Get appropriate template
        template = self.templates.get(template_type, self.templates['rag_qa'])
        
        # Format prompt
        prompt = template.format(
            system_prompt=self.system_prompt,
            conversation_context=conv_context,
            context=context,
            query=query
        )
        
        self.logger.debug("Prompt built", {
            "prompt_length": len(prompt),
            "context_docs": len(retrieved_results)
        })
        
        return prompt
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved results into context string with citations
        
        Args:
            results: Retrieved results with records and scores
            
        Returns:
            Formatted context string
        """
        if not results:
            return "Tidak ada dokumen relevan yang ditemukan."
        
        context_parts = []
        
        for idx, result in enumerate(results, 1):
            record = result.get('record', {})
            
            # Build document header
            reg_type = record.get('regulation_type', 'N/A')
            reg_number = record.get('regulation_number', 'N/A')
            year = record.get('year', 'N/A')
            
            header = f"[Dokumen {idx}] {reg_type} Nomor {reg_number} Tahun {year}"
            
            # Add enacting body
            enacting_body = record.get('enacting_body', '')
            if enacting_body and enacting_body != 'Unknown':
                header += f" - {enacting_body}"
            
            # Add about/title
            about = record.get('about', '')
            if about:
                header += f"\nTentang: {about}"
            
            # Add article/chapter info if available
            article = record.get('article', '')
            chapter = record.get('chapter', '')
            
            if article and article != 'N/A':
                header += f"\nPasal: {article}"
            
            if chapter and chapter != 'N/A':
                header += f"\nBab: {chapter}"
            
            # Add content
            content = record.get('content', '')
            if content:
                # Truncate if too long
                max_content_length = 1000
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                header += f"\n\nIsi:\n{content}"
            
            # Add relevance score (optional)
            score = result.get('final_score', result.get('rerank_score', 0))
            header += f"\n\n[Skor Relevansi: {score:.3f}]"
            
            context_parts.append(header)
        
        # Join all documents
        context = "\n\n" + "="*80 + "\n\n"
        context += "\n\n" + "="*80 + "\n\n".join(context_parts)
        context += "\n\n" + "="*80 + "\n"
        
        return context
    
    def _format_conversation_history(
        self,
        history: List[Dict[str, str]],
        max_turns: int = 5
    ) -> str:
        """
        Format conversation history
        
        Args:
            history: List of conversation turns
            max_turns: Maximum turns to include
            
        Returns:
            Formatted conversation string
        """
        if not history:
            return ""
        
        # Take last N turns
        recent_history = history[-max_turns:] if len(history) > max_turns else history
        
        conv_parts = ["Riwayat Percakapan:"]
        
        for turn in recent_history:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            
            if role == 'user':
                conv_parts.append(f"Pengguna: {content}")
            else:
                conv_parts.append(f"Asisten: {content}")
        
        return "\n".join(conv_parts) + "\n\n"
    
    def _get_rag_qa_template(self) -> str:
        """Get standard RAG QA template"""
        return """{system_prompt}

{conversation_context}Berdasarkan dokumen peraturan perundang-undangan berikut:

{context}

Pertanyaan: {query}

Instruksi:
1. Jawab pertanyaan berdasarkan dokumen yang disediakan
2. Kutip peraturan yang relevan dengan format: [Dokumen X]
3. Jika informasi tidak lengkap dalam dokumen, sebutkan secara jelas
4. Berikan penjelasan yang mudah dipahami
5. Jika ada pasal atau ayat spesifik, kutip dengan tepat
6. Berikan rekomendasi untuk konsultasi dengan ahli hukum jika diperlukan

Jawaban:"""
    
    def _get_followup_template(self) -> str:
        """Get template for follow-up questions"""
        return """{system_prompt}

{conversation_context}Konteks dokumen:

{context}

Pertanyaan lanjutan: {query}

Instruksi:
1. Jawab dengan mempertimbangkan konteks percakapan sebelumnya
2. Rujuk ke peraturan yang sudah dibahas jika relevan
3. Berikan informasi tambahan yang diminta
4. Gunakan referensi: [Dokumen X]

Jawaban:"""
    
    def _get_clarification_template(self) -> str:
        """Get template for clarification requests"""
        return """{system_prompt}

{conversation_context}Dokumen referensi:

{context}

Permintaan klarifikasi: {query}

Instruksi:
1. Perjelas informasi yang diminta
2. Berikan definisi atau penjelasan lebih detail
3. Gunakan contoh jika membantu pemahaman
4. Rujuk ke pasal atau ayat yang relevan

Jawaban:"""
    
    def _get_procedural_template(self) -> str:
        """Get template for procedural questions"""
        return """{system_prompt}

{conversation_context}Dokumen prosedur:

{context}

Pertanyaan prosedur: {query}

Instruksi:
1. Jelaskan prosedur atau tata cara secara bertahap
2. Identifikasi persyaratan yang diperlukan
3. Sebutkan batas waktu jika ada
4. Rujuk ke peraturan yang mengatur: [Dokumen X]
5. Berikan peringatan jika ada sanksi atau konsekuensi

Jawaban:"""
    
    def _get_comparison_template(self) -> str:
        """Get template for comparative analysis"""
        return """{system_prompt}

{conversation_context}Dokumen untuk perbandingan:

{context}

Permintaan perbandingan: {query}

Instruksi:
1. Bandingkan ketentuan dari berbagai peraturan
2. Identifikasi persamaan dan perbedaan
3. Jelaskan hierarki peraturan jika relevan
4. Sebutkan peraturan mana yang lebih baru atau lebih tinggi
5. Berikan kesimpulan yang jelas

Jawaban:"""

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation)
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for Indonesian
        return len(text) // 4
    
    def truncate_context(
        self,
        results: List[Dict[str, Any]],
        max_tokens: int = 6000
    ) -> List[Dict[str, Any]]:
        """
        Truncate context to fit within token limit
        
        Args:
            results: Retrieved results
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated results list
        """
        truncated = []
        current_tokens = 0
        
        for result in results:
            # Estimate tokens for this result
            record = result.get('record', {})
            result_text = f"{record.get('about', '')} {record.get('content', '')}"
            result_tokens = self.estimate_tokens(result_text)
            
            if current_tokens + result_tokens > max_tokens:
                self.logger.warning("Context truncated due to token limit", {
                    "included_docs": len(truncated),
                    "total_docs": len(results)
                })
                break
            
            truncated.append(result)
            current_tokens += result_tokens
        
        return truncated
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get information about available templates"""
        return {
            'available_templates': list(self.templates.keys()),
            'default_template': 'rag_qa',
            'system_prompt_length': len(self.system_prompt)
        }