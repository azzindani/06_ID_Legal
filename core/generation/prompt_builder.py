"""
Prompt Builder for Indonesian Legal RAG System
Constructs context-aware prompts with retrieved documents and citations

File: core/generation/prompt_builder.py
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from logger_utils import get_logger
from config import SYSTEM_PROMPT


class ThinkingMode(Enum):
    """Thinking mode enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PromptBuilder:
    """
    Builds optimized prompts for legal question answering
    Handles context formatting, citation preparation, and prompt templates
    Includes integrated thinking mode management for multi-level analysis
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("PromptBuilder")
        self.config = config

        # System prompt from config
        self.system_prompt = config.get('system_prompt', SYSTEM_PROMPT)

        # Thinking mode settings
        self.enable_thinking_pipeline = config.get('enable_thinking_pipeline', True)

        # Define thinking mode configurations
        self.thinking_configs = {
            ThinkingMode.LOW: {
                'name': 'Low Thinking Mode',
                'description': 'Basic analysis for straightforward queries',
                'min_tokens': 2048,
                'max_tokens': 4096,
                'thinking_depth': 'basic',
                'instruction_template': self._get_low_thinking_template()
            },
            ThinkingMode.MEDIUM: {
                'name': 'Medium Thinking Mode',
                'description': 'Deep thinking for moderate complexity',
                'min_tokens': 4096,
                'max_tokens': 8192,
                'thinking_depth': 'deep',
                'instruction_template': self._get_medium_thinking_template()
            },
            ThinkingMode.HIGH: {
                'name': 'High Thinking Mode',
                'description': 'Iterative & recursive thinking for complex analysis',
                'min_tokens': 8192,
                'max_tokens': 16384,
                'thinking_depth': 'iterative_recursive',
                'instruction_template': self._get_high_thinking_template()
            }
        }

        # Prompt templates
        self.templates = {
            'rag_qa': self._get_rag_qa_template(),
            'followup': self._get_followup_template(),
            'clarification': self._get_clarification_template(),
            'procedural': self._get_procedural_template(),
            'comparison': self._get_comparison_template()
        }

        self.logger.info("PromptBuilder initialized", {
            "templates": len(self.templates),
            "thinking_pipeline": self.enable_thinking_pipeline,
            "thinking_modes": len(self.thinking_configs)
        })

    def build_prompt(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        template_type: str = 'rag_qa',
        thinking_mode: str = 'low'
    ) -> str:
        """
        Build complete prompt with system instructions, context, and query

        Args:
            query: User query
            retrieved_results: Retrieved and ranked documents
            query_analysis: Optional query analysis from QueryDetector
            conversation_history: Optional conversation context
            template_type: Type of prompt template to use
            thinking_mode: Thinking mode ('low', 'medium', 'high')

        Returns:
            Complete formatted prompt
        """
        self.logger.info("Building prompt", {
            "query_length": len(query),
            "num_results": len(retrieved_results),
            "template": template_type,
            "thinking_mode": thinking_mode
        })

        # Get thinking instructions if enabled and replace placeholder
        thinking_applied = False
        if self.enable_thinking_pipeline:
            thinking_config = self._get_thinking_instructions(
                mode=thinking_mode,
                query_complexity=query_analysis.get('complexity', 0.5) if query_analysis else None
            )

            self.logger.info("Thinking mode applied", {
                "mode": thinking_config['mode'],
                "token_range": f"{thinking_config['min_tokens']}-{thinking_config['max_tokens']}"
            })

            # Replace placeholder in <think> tag with actual thinking instructions
            enhanced_system_prompt = self.system_prompt.replace(
                "[Mode-specific thinking instructions are provided based on thinking mode]",
                thinking_config['instructions']
            )
            thinking_applied = True
        else:
            # Use system prompt as-is if thinking pipeline disabled
            enhanced_system_prompt = self.system_prompt

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
            system_prompt=enhanced_system_prompt,
            conversation_context=conv_context,
            context=context,
            query=query
        )

        self.logger.debug("Prompt built", {
            "prompt_length": len(prompt),
            "context_docs": len(retrieved_results),
            "thinking_enhanced": thinking_applied
        })

        return prompt

    def _get_thinking_instructions(
        self,
        mode: str = "low",
        query_complexity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get thinking instructions based on mode.

        Args:
            mode: Thinking mode ('low', 'medium', 'high')
            query_complexity: Optional complexity score (0-1) for auto-selection

        Returns:
            Dictionary with thinking instructions and configuration
        """
        # Convert string to enum
        try:
            thinking_mode = ThinkingMode(mode.lower())
        except ValueError:
            self.logger.warning(f"Invalid thinking mode '{mode}', defaulting to LOW")
            thinking_mode = ThinkingMode.LOW

        config = self.thinking_configs[thinking_mode]

        return {
            'mode': thinking_mode.value,
            'name': config['name'],
            'description': config['description'],
            'min_tokens': config['min_tokens'],
            'max_tokens': config['max_tokens'],
            'thinking_depth': config['thinking_depth'],
            'instructions': config['instruction_template']
        }

    def _get_low_thinking_template(self) -> str:
        """Low thinking mode: Basic analysis"""
        return """Dalam tag <think>, lakukan analisis DASAR dengan struktur berikut:

1. **PEMAHAMAN PERTANYAAN** (1-2 paragraf)
   - Identifikasi inti pertanyaan
   - Tentukan jenis pertanyaan (definitional, procedural, sanctions, dll)
   - Catat kata kunci penting

2. **ANALISIS DOKUMEN** (2-3 paragraf)
   - Evaluasi relevansi setiap dokumen yang disediakan
   - Identifikasi pasal dan ayat yang relevan
   - Catat hierarki peraturan (UU > PP > Permen, dll)

3. **SINTESIS INFORMASI** (2-3 paragraf)
   - Gabungkan informasi dari berbagai sumber
   - Identifikasi konsistensi atau konflik antar peraturan
   - Tentukan peraturan yang paling authoritative

4. **KESIMPULAN ANALISIS** (1-2 paragraf)
   - Ringkas temuan utama
   - Identifikasi informasi yang masih kurang (jika ada)
   - Tentukan arah jawaban"""

    def _get_medium_thinking_template(self) -> str:
        """Medium thinking mode: Deep thinking"""
        return """Dalam tag <think>, lakukan deep thinking:
1. Analisis pertanyaan dari berbagai sudut pandang & konteks hukum
2. Evaluasi dokumen: ranking relevansi, pasal spesifik, hierarki, status peraturan
3. Cross-reference: referensi silang, konsistensi, lex specialis/generalis
4. Sintesis komprehensif: integrasikan semua sumber, analisis konflik
5. Validasi: cross-check pasal, verifikasi hierarki & interpretasi
6. Kesimpulan: tingkat kepastian, struktur jawaban optimal"""

    def _get_high_thinking_template(self) -> str:
        """High thinking mode: Iterative & recursive thinking"""
        return """Dalam tag <think>, lakukan iterative & recursive thinking:

**ANALISIS:**
1. Dekonstruksi pertanyaan → sub-pertanyaan, scope, konteks
2. Reading: baca semua dokumen, identifikasi section kunci
3. Analysis: ekstrak pasal/ayat detail, definisi istilah, struktur argumen
4. Relationship mapping: hierarki, referensi, peraturan terkait

**EVALUASI:**
5. Multi-perspective: legal researcher, KG specialist, procedural expert, devil's advocate
6. Conflict check: inkonsistensi, resolusi konflik (lex superior/specialis/posterior)
7. Validation loop: semua pertanyaan terjawab? kutipan akurat? interpretasi konsisten?

**REFINEMENT:**
8. Re-check dokumen untuk info terlewat
9. Bottom-up verification: detail → keseluruhan
10. Top-down validation: kesimpulan → evidence
11. Cross-validation: klaim × supporting evidence

**FINAL:**
12. Quality assessment, gap analysis, final synthesis
13. Checklist: kutipan akurat, hierarki benar, referensi konsisten, disclaimer included"""

    def estimate_thinking_tokens(
        self,
        mode: str,
        num_documents: int,
        query_length: int
    ) -> Dict[str, int]:
        """
        Estimate thinking token usage based on mode and context.

        Args:
            mode: Thinking mode ('low', 'medium', 'high')
            num_documents: Number of retrieved documents
            query_length: Length of query in characters

        Returns:
            Dictionary with token estimates
        """
        try:
            thinking_mode = ThinkingMode(mode.lower())
        except ValueError:
            thinking_mode = ThinkingMode.LOW

        config = self.thinking_configs[thinking_mode]

        # Base estimate from mode
        base_estimate = (config['min_tokens'] + config['max_tokens']) // 2

        # Adjust based on context complexity
        doc_factor = min(num_documents / 3.0, 2.0)  # Max 2x for many docs
        query_factor = min(query_length / 200.0, 1.5)  # Max 1.5x for long queries

        estimated_tokens = int(base_estimate * doc_factor * query_factor)

        # Clamp to mode's range
        estimated_tokens = max(config['min_tokens'],
                              min(estimated_tokens, config['max_tokens']))

        return {
            'mode': thinking_mode.value,
            'estimated_tokens': estimated_tokens,
            'min_tokens': config['min_tokens'],
            'max_tokens': config['max_tokens'],
            'documents_factor': round(doc_factor, 2),
            'query_factor': round(query_factor, 2)
        }

    def get_available_thinking_modes(self) -> list:
        """Get list of available thinking modes with descriptions"""
        return [
            {
                'value': mode.value,
                'name': config['name'],
                'description': config['description'],
                'token_range': f"{config['min_tokens']}-{config['max_tokens']}"
            }
            for mode, config in self.thinking_configs.items()
        ]

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
            'system_prompt_length': len(self.system_prompt),
            'thinking_modes': self.get_available_thinking_modes()
        }
