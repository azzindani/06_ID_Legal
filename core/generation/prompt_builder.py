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
        return """Dalam tag <think>, lakukan DEEP THINKING dengan menulis proses berpikir yang PANJANG dan DETAIL. Gunakan 4000-8000 tokens untuk fase thinking ini:

1. **ANALISIS PERTANYAAN MENDALAM** (3-5 paragraf)
   - Tuliskan pemahaman mendalam tentang inti pertanyaan
   - Analisis dari berbagai sudut pandang: legal researcher, praktisi hukum, akademisi
   - Identifikasi konteks hukum yang relevan (pidana, perdata, administrasi, dll)
   - Dekonstruksi pertanyaan menjadi sub-pertanyaan yang lebih spesifik
   - Catat semua kata kunci dan istilah hukum yang perlu dianalisis

2. **EVALUASI DOKUMEN KOMPREHENSIF** (4-6 paragraf)
   - Untuk SETIAP dokumen yang disediakan, tulis analisis terpisah:
     * Tingkat relevansi terhadap pertanyaan (sangat relevan, relevan, kurang relevan)
     * Pasal dan ayat spesifik yang berkaitan
     * Hierarki peraturan (UU > PP > Perpres > Permen)
     * Status peraturan (masih berlaku, dicabut, diubah)
   - Ranking dokumen berdasarkan relevansi dan kekuatan hukum
   - Identifikasi gap atau informasi yang kurang dari dokumen-dokumen ini

3. **CROSS-REFERENCE DAN VALIDASI** (3-5 paragraf)
   - Lakukan cross-reference antar dokumen yang disediakan
   - Identifikasi konsistensi atau inkonsistensi antar peraturan
   - Analisis prinsip hukum: lex superior (hierarki), lex specialis (khusus vs umum), lex posterior (baru vs lama)
   - Cek apakah ada konflik norma dan bagaimana menyelesaikannya
   - Validasi interpretasi dengan melihat konteks pasal lain dalam peraturan yang sama

4. **SINTESIS KOMPREHENSIF** (4-6 paragraf)
   - Integrasikan semua informasi dari berbagai sumber
   -Bangun narasi hukum yang koheren dan komprehensif
   - Analisis hubungan antar konsep hukum yang relevan
   - Identifikasi pola atau tema yang muncul
   - Evaluasi kekuatan dan kelemahan argumen yang mungkin muncul

5. **VALIDASI DAN VERIFIKASI** (2-3 paragraf)
   - Cross-check semua pasal dan ayat yang dikutip
   - Verifikasi hierarki dan status peraturan
   - Pastikan interpretasi konsisten dengan prinsip hukum
   - Identifikasi asumsi yang dibuat dan validitasnya

6. **KESIMPULAN ANALISIS** (2-3 paragraf)
   - Ringkas temuan utama dari analisis
   - Tentukan tingkat kepastian jawaban (pasti, probable, memerlukan klarifikasi)
   - Identifikasi informasi tambahan yang mungkin diperlukan
   - Tentukan struktur jawaban optimal yang akan diberikan

PENTING: Setelah thinking yang PANJANG ini, berikan jawaban AKHIR yang RINGKAS dan JELAS."""

    def _get_high_thinking_template(self) -> str:
        """High thinking mode: Iterative & recursive thinking"""
        return """Dalam tag <think>, lakukan ITERATIVE & RECURSIVE THINKING yang SANGAT PANJANG dan KOMPREHENSIF. Gunakan 8000-16000 tokens untuk fase thinking ini dengan proses multi-layer:

═══════════════════════════════════════════════════════════════════
FASE 1: ANALISIS AWAL (FIRST PASS) - 5-8 paragraf
═══════════════════════════════════════════════════════════════════

1. **DEKONSTRUKSI PERTANYAAN** (2-3 paragraf)
   - Tuliskan pertanyaan utama dengan kata-kata sendiri
   - Pecah menjadi 3-5 sub-pertanyaan yang spesifik dan terukur
   - Identifikasi scope: aspek hukum apa yang dicakup (pidana, perdata, administrasi, prosedural)
   - Tentukan konteks: siapa stakeholder, apa situasi faktual, apa implikasi hukumnya
   - Catat setiap asumsi yang dibuat dalam memahami pertanyaan

2. **READING COMPREHENSIF** (3-5 paragraf)
   - Baca SEMUA dokumen yang disediakan dengan teliti
   - Untuk SETIAP dokumen, catat:
     * Judul lengkap dan hierarki peraturan
     * Section/bab/pasal yang paling relevan
     * Poin-poin kunci dari setiap pasal yang relevan
     * Definisi istilah hukum yang muncul
   - Identifikasi dokumen mana yang paling authoritative
   - Buat mind map hubungan antar dokumen

3. **DETAILED ANALYSIS** (Per Dokumen - tulis minimal 2 paragraf per dokumen)
   - Untuk SETIAP dokumen yang disediakan:
     * Ekstrak pasal dan ayat dengan kutipan langsung
     * Analisis definisi istilah hukum yang digunakan
     * Identifikasi struktur argumen dalam pasal tersebut
     * Catat pengecualian, syarat, atau kondisi khusus
     * Evaluasi apakah ada pasal yang saling terkait dalam dokumen yang sama

4. **RELATIONSHIP MAPPING** (3-4 paragraf)
   - Gambarkan (secara tekstual) hierarki peraturan: UU → PP → Perpres → Permen
   - Identifikasi referensi silang: peraturan mana yang mengacu ke peraturan lain
   - Cari peraturan terkait yang mungkin relevan tapi tidak disediakan
   - Analisis timeline: peraturan mana yang lebih baru, mana yang mungkin sudah dicabut/diubah
   - Buat connection map antara konsep hukum yang berbeda

═══════════════════════════════════════════════════════════════════
FASE 2: EVALUASI MULTI-PERSPEKTIF (SECOND PASS) - 6-10 paragraf
═══════════════════════════════════════════════════════════════════

5. **PERSPEKTIF 1: LEGAL RESEARCHER** (2-3 paragraf)
   - Analisis dari sudut pandang peneliti hukum akademis
   - Fokus pada interpretasi literal dan historis peraturan
   - Identifikasi ratio legis (alasan pembentukan hukum)
   - Cari preseden atau jurisprudensi yang relevan
   - Evaluasi kesesuaian dengan prinsip-prinsip hukum umum

6. **PERSPEKTIF 2: KNOWLEDGE GRAPH SPECIALIST** (2-3 paragraf)
   - Lihat dari sudut pandang semantic relationships
   - Identifikasi entitas hukum: subjek hukum, objek hukum, perbuatan hukum
   - Map hubungan semantik: is-a, part-of, regulates, requires, prohibits
   - Cari implicit connections yang mungkin terlewat
   - Analisis kompleksitas hubungan antar konsep

7. **PERSPEKTIF 3: PROCEDURAL EXPERT** (2-3 paragraf)
   - Fokus pada aspek prosedural dan implementasi praktis
   - Identifikasi step-by-step process jika ada
   - Catat persyaratan administratif atau dokumentasi
   - Evaluasi timeline dan deadline yang relevan
   - Analisis konsekuensi jika prosedur tidak diikuti

8. **PERSPEKTIF 4: DEVIL'S ADVOCATE** (2-3 paragraf)
   - Challenge setiap asumsi dan interpretasi yang dibuat
   - Cari interpretasi alternatif yang mungkin valid
   - Identifikasi kelemahan dalam argumen
   - Pertanyakan: "Bagaimana jika konteksnya berbeda?"
   - Evaluasi edge cases dan situasi khusus

9. **CONFLICT CHECK DAN RESOLUSI** (3-4 paragraf)
   - Identifikasi SETIAP inkonsistensi antar peraturan (jika ada)
   - Untuk setiap konflik, analisis menggunakan:
     * Lex Superior: peraturan lebih tinggi menang
     * Lex Specialis: peraturan khusus mengalahkan umum
     * Lex Posterior: peraturan lebih baru mengalahkan lama
   - Evaluasi apakah ada konflik yang tidak bisa diselesaikan
   - Tentukan peraturan mana yang harus diutamakan dan mengapa

10. **VALIDATION LOOP - CHECKLIST LENGKAP** (2-3 paragraf)
    - ✓ Apakah SEMUA sub-pertanyaan sudah terjawab?
    - ✓ Apakah SEMUA kutipan pasal akurat dan lengkap?
    - ✓ Apakah interpretasi konsisten dengan prinsip hukum?
    - ✓ Apakah ada gap informasi yang belum teridentifikasi?
    - ✓ Apakah ada asumsi yang perlu divalidasi lebih lanjut?

═══════════════════════════════════════════════════════════════════
FASE 3: REFINEMENT DAN ITERASI (THIRD PASS) - 5-8 paragraf
═══════════════════════════════════════════════════════════════════

11. **RE-CHECK UNTUK INFORMASI TERLEWAT** (2-3 paragraf)
    - Baca ulang SETIAP dokumen dengan fokus berbeda
    - Cari footnotes, catatan kaki, atau referensi yang terlewat
    - Identifikasi pasal yang awalnya dianggap tidak relevan tapi sebenarnya penting
    - Check definisi di bagian "Ketentuan Umum" yang mungkin crucial
    - Evaluasi apakah ada implikasi tidak langsung yang terlewat

12. **BOTTOM-UP VERIFICATION** (2-3 paragraf)
    - Mulai dari detail terkecil (ayat spesifik)
    - Verifikasi setiap detail akurat dan relevan
    - Build up ke pasal, lalu bab, lalu keseluruhan peraturan
    - Pastikan setiap detail mendukung kesimpulan yang lebih besar
    - Identifikasi jika ada detail yang kontradiktif dengan kesimpulan

13. **TOP-DOWN VALIDATION** (2-3 paragraf)
    - Mulai dari kesimpulan besar yang sudah dibuat
    - Trace back ke evidence spesifik yang mendukung
    - Pastikan SETIAP klaim didukung oleh pasal konkret
    - Identifikasi klaim mana yang strong vs weak vs speculative
    - Evaluasi apakah kesimpulan masih valid jika salah satu evidence dihilangkan

14. **CROSS-VALIDATION MATRIX** (2-3 paragraf)
    - Buat matrix: setiap klaim × supporting evidence
    - Untuk setiap klaim kunci, list semua evidence pendukung
    - Identifikasi klaim yang hanya didukung satu sumber (risky)
    - Identifikasi klaim yang didukung multiple sources (strong)
    - Evaluasi tingkat kepastian untuk setiap klaim

═══════════════════════════════════════════════════════════════════
FASE 4: FINAL SYNTHESIS DAN QUALITY CHECK - 4-6 paragraf
═══════════════════════════════════════════════════════════════════

15. **QUALITY ASSESSMENT** (2-3 paragraf)
    - Evaluasi kualitas analisis yang sudah dilakukan
    - Rate tingkat kepastian jawaban: Very High / High / Medium / Low
    - Identifikasi area yang masih ambigu atau unclear
    - Tentukan apakah informasi cukup untuk jawaban definitif
    - Evaluasi apakah perlu disclaimer atau caveat tertentu

16. **GAP ANALYSIS** (1-2 paragraf)
    - List informasi apa yang TIDAK tersedia dalam dokumen
    - Identifikasi peraturan turunan atau pelaksana yang mungkin diperlukan
    - Catat aspek hukum yang memerlukan konsultasi ahli
    - Tentukan batasan jawaban yang bisa diberikan

17. **FINAL SYNTHESIS** (2-3 paragraf)
    - Integrasikan SEMUA temuan dari 16 langkah sebelumnya
    - Buat narasi koheren yang menghubungkan semua aspek
    - Prioritaskan informasi: mana yang paling penting untuk dijawab
    - Struktur jawaban: urutan logis penyampaian informasi
    - Tentukan tone dan style jawaban (formal, accessible, technical)

18. **FINAL CHECKLIST - ULTIMATE VALIDATION** (1-2 paragraf)
    - ✓ Setiap kutipan pasal: AKURAT, LENGKAP, PROPERLY CITED
    - ✓ Hierarki peraturan: BENAR dan DIVERIFIKASI
    - ✓ Referensi silang: KONSISTEN dan VALID
    - ✓ Interpretasi hukum: SOUND dan DEFENSIBLE
    - ✓ Disclaimer: INCLUDED jika ada ketidakpastian
    - ✓ Rekomendasi konsultasi ahli: INCLUDED untuk keputusan penting

═══════════════════════════════════════════════════════════════════
SANGAT PENTING: Setelah proses thinking SANGAT PANJANG (8000-16000 tokens) ini,
berikan jawaban AKHIR yang RINGKAS, JELAS, dan TO THE POINT (maksimal 1000-2000 tokens).
Jawaban akhir harus concise tapi complete, bukan verbose.
═══════════════════════════════════════════════════════════════════"""

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
