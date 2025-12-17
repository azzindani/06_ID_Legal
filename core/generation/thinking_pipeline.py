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


def create_thinking_pipeline() -> ThinkingPipeline:
    """Factory function to create ThinkingPipeline instance"""
    return ThinkingPipeline()
