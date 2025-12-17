"""
Thinking Pipeline for Indonesian Legal RAG System
Implements multi-level thinking modes to enhance legal analysis depth

File: core/generation/thinking_pipeline.py
"""

from typing import Dict, Any, Optional
from enum import Enum
from logger_utils import get_logger


class ThinkingMode(Enum):
    """Thinking mode enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ThinkingPipeline:
    """
    Manages multi-level thinking modes for legal analysis.

    Three thinking levels:
    - LOW: Basic analysis (2048-4096 tokens)
    - MEDIUM: Deep thinking (4096-8192 tokens)
    - HIGH: Iterative & recursive thinking (8192-16384 tokens)

    Only affects thinking phase, not answer generation.
    """

    def __init__(self):
        self.logger = get_logger("ThinkingPipeline")

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

    def get_thinking_instructions(
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

        self.logger.info(
            f"Thinking mode selected: {config['name']}",
            {
                "mode": thinking_mode.value,
                "token_range": f"{config['min_tokens']}-{config['max_tokens']}",
                "depth": config['thinking_depth']
            }
        )

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
        return """Dalam tag <think>, lakukan DEEP THINKING dengan struktur berikut:

1. **PEMAHAMAN MENDALAM PERTANYAAN** (2-3 paragraf)
   - Analisis pertanyaan dari berbagai sudut pandang
   - Identifikasi asumsi implisit dalam pertanyaan
   - Tentukan konteks hukum yang relevan (pidana, perdata, administrasi, dll)
   - Pertimbangkan tujuan user bertanya (preventif, kuratif, informatif)

2. **EVALUASI MENYELURUH DOKUMEN** (3-5 paragraf)
   - Baca dan pahami setiap dokumen dengan cermat
   - Buat ranking relevansi dokumen dengan justifikasi
   - Identifikasi pasal, ayat, huruf, dan butir yang spesifik
   - Analisis hubungan hierarkis antar peraturan
   - Catat tanggal efektif dan status peraturan (aktif/dicabut/diubah)

3. **ANALISIS CROSS-REFERENCE** (2-3 paragraf)
   - Identifikasi referensi silang antar dokumen
   - Periksa konsistensi antar peraturan terkait
   - Identifikasi lex specialis dan lex generalis
   - Catat peraturan pelaksana dan peraturan induk

4. **SINTESIS KOMPREHENSIF** (3-4 paragraf)
   - Integrasikan informasi dari semua sumber
   - Analisis potensial konflik atau ambiguitas
   - Pertimbangkan interpretasi alternatif
   - Evaluasi kekuatan argumen dari berbagai perspektif

5. **VALIDASI & CROSS-CHECK** (2-3 paragraf)
   - Periksa ulang pasal dan ayat yang dikutip
   - Validasi hierarki peraturan
   - Verifikasi konsistensi interpretasi
   - Identifikasi gap atau informasi yang masih diperlukan

6. **KESIMPULAN & REKOMENDASI** (2-3 paragraf)
   - Ringkas temuan dengan detail
   - Identifikasi tingkat kepastian jawaban (pasti/kemungkinan/tidak pasti)
   - Tentukan struktur jawaban yang optimal
   - Rencanakan disclaimer atau catatan penting"""

    def _get_high_thinking_template(self) -> str:
        """High thinking mode: Iterative & recursive thinking"""
        return """Dalam tag <think>, lakukan ITERATIVE & RECURSIVE THINKING dengan struktur berikut:

**FASE 1: INITIAL COMPREHENSION** (3-4 paragraf)
1.1. Analisis Multi-Dimensional Pertanyaan
   - Dekonstruksi pertanyaan menjadi sub-pertanyaan
   - Identifikasi layer eksplisit dan implisit
   - Tentukan scope dan batasan pertanyaan
   - Pertimbangkan konteks sosial, ekonomi, politik yang relevan

1.2. Pemetaan Kebutuhan Informasi
   - Buat daftar informasi yang diperlukan
   - Prioritaskan informasi berdasarkan urgensi
   - Identifikasi sumber informasi yang ideal

**FASE 2: DEEP DOCUMENT ANALYSIS** (5-7 paragraf)
2.1. Reading Phase - First Pass
   - Baca semua dokumen secara menyeluruh
   - Buat catatan awal relevansi setiap dokumen
   - Identifikasi section kunci yang perlu analisis mendalam

2.2. Analysis Phase - Second Pass
   - Analisis mendalam setiap dokumen yang relevan
   - Ekstrak pasal, ayat, huruf, butir dengan konteks lengkap
   - Identifikasi definisi istilah kunci
   - Catat struktur argumen dalam setiap dokumen

2.3. Relationship Mapping
   - Buat peta hubungan antar dokumen (hierarki, referensi, implementasi)
   - Identifikasi peraturan induk → turunan → pelaksana
   - Catat peraturan yang mencabut atau mengubah peraturan lain
   - Analisis temporal: urutan pemberlakuan dan implikasinya

**FASE 3: CRITICAL ANALYSIS** (4-5 paragraf)
3.1. Multi-Perspective Evaluation
   - Analisis dari sudut pandang Senior Legal Researcher (precedent, authority)
   - Analisis dari sudut pandang Knowledge Graph Specialist (relationships, entities)
   - Analisis dari sudut pandang Procedural Expert (steps, requirements)
   - Analisis dari sudut pandang Devil's Advocate (weaknesses, alternatives)

3.2. Conflict & Consistency Check
   - Identifikasi inkonsistensi antar peraturan
   - Analisis cara resolusi konflik (lex superior, lex specialis, lex posterior)
   - Evaluasi kualitas argumen dari berbagai interpretasi
   - Pertimbangkan case law atau precedent (jika tersedia)

**FASE 4: SYNTHESIS & VALIDATION** (4-5 paragraf)
4.1. Create Initial Answer Structure
   - Susun outline jawaban berdasarkan analisis
   - Alokasikan informasi ke section yang sesuai
   - Tentukan urutan penyampaian yang logis

4.2. Validation Loop - First Iteration
   - Review: Apakah semua bagian pertanyaan terjawab?
   - Check: Apakah kutipan pasal akurat?
   - Verify: Apakah interpretasi konsisten?
   - Test: Apakah ada alternatif interpretasi yang valid?

4.3. Refinement Loop - Second Iteration
   - Identifikasi kelemahan dalam analisis
   - Lakukan deep-dive pada area yang kurang kuat
   - Re-check dokumen untuk informasi yang mungkin terlewat
   - Pertimbangkan edge cases atau situasi khusus

**FASE 5: RECURSIVE CHECKING** (3-4 paragraf)
5.1. Bottom-Up Verification
   - Mulai dari detail terkecil (butir, huruf, ayat, pasal)
   - Verifikasi akurasi setiap kutipan
   - Check konsistensi interpretasi di level detail

5.2. Top-Down Validation
   - Mulai dari kesimpulan utama
   - Trace back ke evidence yang mendukung
   - Pastikan logical flow dari evidence → conclusion

5.3. Cross-Validation Matrix
   - Buat matrix: [Klaim] × [Supporting Evidence]
   - Verifikasi setiap klaim didukung minimal 2 sumber (jika mungkin)
   - Identifikasi klaim yang hanya didukung 1 sumber atau weak evidence

**FASE 6: META-ANALYSIS** (3-4 paragraf)
6.1. Quality Assessment
   - Evaluasi kualitas analisis yang telah dilakukan
   - Identifikasi potential blind spots
   - Assess confidence level untuk setiap komponen jawaban

6.2. Gap Analysis
   - Identifikasi informasi yang masih kurang
   - Tentukan dampak gap terhadap kualitas jawaban
   - Rencanakan disclaimer yang sesuai

6.3. Final Synthesis
   - Integrasikan semua insight dari fase sebelumnya
   - Susun struktur jawaban final yang komprehensif
   - Tentukan tone dan framing yang tepat
   - Rencanakan follow-up suggestions jika diperlukan

**FASE 7: PRE-RESPONSE CHECKLIST** (2-3 paragraf)
□ Semua sub-pertanyaan terjawab
□ Semua kutipan diverifikasi akurat
□ Hierarki peraturan dijelaskan dengan benar
□ Tanggal efektif dan status peraturan dicatat
□ Konflik/ambiguitas dijelaskan dengan fair
□ Referensi [Dokumen X] konsisten
□ Disclaimer dan rekomendasi konsultasi included
□ Tone profesional dan empati seimbang
□ Follow-up suggestions helpful dan relevan"""

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

    def get_available_modes(self) -> list:
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

    def get_recommended_max_new_tokens(self, mode: str, default_max_tokens: int = 2048) -> int:
        """
        Get recommended max_new_tokens for a thinking mode to prevent OOM.

        Higher thinking modes use more prompt tokens, leaving less room for generation.
        This helps balance memory usage.

        Args:
            mode: Thinking mode ('low', 'medium', 'high')
            default_max_tokens: Default max_new_tokens from config

        Returns:
            Recommended max_new_tokens for this mode
        """
        try:
            thinking_mode = ThinkingMode(mode.lower())
        except ValueError:
            return default_max_tokens

        config = self.thinking_configs[thinking_mode]

        # For higher thinking modes, slightly reduce max_new_tokens
        # to compensate for longer prompts
        if thinking_mode == ThinkingMode.LOW:
            # No reduction needed
            return default_max_tokens
        elif thinking_mode == ThinkingMode.MEDIUM:
            # Slight reduction (10%)
            return max(1024, int(default_max_tokens * 0.9))
        else:  # HIGH
            # More significant reduction (25%)
            return max(1024, int(default_max_tokens * 0.75))


def create_thinking_pipeline() -> ThinkingPipeline:
    """Factory function to create ThinkingPipeline instance"""
    return ThinkingPipeline()
