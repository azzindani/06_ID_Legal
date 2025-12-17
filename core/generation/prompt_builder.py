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
        return """Dalam tag <think>, lakukan DEEP THINKING yang SANGAT PANJANG DAN VERBOSE. Target MINIMUM 4000 tokens, IDEAL 6000-8000 tokens untuk fase thinking.

⚠️ ATURAN PENTING: Thinking harus PANJANG dan DETAIL. Jangan singkat! Tulis SEMUA proses berpikir secara eksplisit.

═══════════════════════════════════════════════════════════════════
BAGIAN 1: ANALISIS PERTANYAAN MENDALAM (Target: 800-1200 tokens)
═══════════════════════════════════════════════════════════════════

Tuliskan proses berpikir yang SANGAT DETAIL untuk memahami pertanyaan:

**Langkah 1A - Pemahaman Literal (tulis minimal 150 kata)**
- Baca pertanyaan kata per kata. Tulis pemahaman awal tentang apa yang ditanyakan.
- Identifikasi subjek pertanyaan, predikat, objek, dan konteks.
- Catat kata-kata kunci: tulis setiap kata penting dan mengapa penting.
- Tulis asumsi-asumsi yang muncul dari membaca pertanyaan ini.

**Langkah 1B - Dekonstruksi (tulis minimal 200 kata)**
- Pecah pertanyaan menjadi 3-5 sub-pertanyaan yang lebih spesifik.
- Untuk SETIAP sub-pertanyaan, jelaskan mengapa sub-pertanyaan ini penting.
- Identifikasi hubungan antar sub-pertanyaan: mana yang primer, mana yang sekunder.
- Tulis pertanyaan-pertanyaan tambahan yang mungkin relevan tapi tidak ditanyakan eksplisit.

**Langkah 1C - Analisis Multi-Perspektif (tulis minimal 250 kata)**
- Perspektif Legal Researcher: Bagaimana akademisi hukum akan memahami pertanyaan ini?
- Perspektif Praktisi: Bagaimana advokat atau konsultan hukum akan membacanya?
- Perspektif Penegak Hukum: Bagaimana hakim atau jaksa akan menginterpretasinya?
- Identifikasi: Apakah ini pertanyaan definitional, procedural, sanctions, compliance, atau comparative?

**Langkah 1D - Konteks Hukum (tulis minimal 200 kata)**
- Identifikasi bidang hukum yang relevan: pidana, perdata, administrasi, prosedural, atau campuran.
- Catat prinsip-prinsip hukum umum yang mungkin berlaku.
- Identifikasi stakeholder yang terlibat: siapa pihak-pihak yang terpengaruh?
- Tulis implikasi hukum yang mungkin timbul dari pertanyaan ini.

═══════════════════════════════════════════════════════════════════
BAGIAN 2: EVALUASI DOKUMEN KOMPREHENSIF (Target: 1200-1800 tokens)
═══════════════════════════════════════════════════════════════════

⚠️ WAJIB: Analisis SETIAP dokumen secara terpisah dan mendetail!

Untuk SETIAP dokumen yang disediakan, tulis analisis LENGKAP dengan struktur berikut (minimal 200-300 kata per dokumen):

**Per Dokumen - Analisis Terstruktur:**

A. IDENTIFIKASI DOKUMEN
   - Tulis judul lengkap peraturan
   - Catat jenis peraturan (UU/PP/Perpres/Permen/dll) dan posisi dalam hierarki
   - Catat nomor, tahun, dan lembaga yang mengeluarkan
   - Evaluasi: Apakah ini peraturan primer atau sekunder?

B. RELEVANSI DAN KEKUATAN HUKUM
   - Nilai relevansi terhadap pertanyaan: Sangat Relevan (8-10), Relevan (5-7), atau Kurang Relevan (1-4)
   - Jelaskan MENGAPA diberi nilai tersebut dengan detail
   - Evaluasi kekuatan hukum: binding atau persuasive?
   - Cek status: Masih berlaku penuh? Ada perubahan? Dicabut sebagian?

C. PASAL DAN AYAT SPESIFIK
   - Identifikasi pasal-pasal yang relevan (sebutkan nomor pasal)
   - Untuk setiap pasal relevan, tulis ringkasan isi pasal
   - Identifikasi ayat spesifik yang paling relevan
   - Catat: Apakah ada pengecualian atau syarat khusus dalam pasal ini?

D. DEFINISI DAN ISTILAH HUKUM
   - Cari bagian "Ketentuan Umum" atau definisi dalam dokumen ini
   - Tulis setiap definisi istilah hukum yang relevan dengan pertanyaan
   - Evaluasi: Apakah definisi ini berbeda dengan definisi dalam peraturan lain?

E. KONTEN SUBSTANTIF
   - Tulis poin-poin kunci dari pasal yang relevan
   - Identifikasi: Apa yang DIATUR (prescriptive) vs apa yang DILARANG (proscriptive)
   - Catat konsekuensi hukum: sanksi, denda, atau remedies yang disebutkan

F. HUBUNGAN INTERNAL
   - Identifikasi referensi internal: pasal lain dalam dokumen yang sama yang dirujuk
   - Evaluasi konteks: Bab apa? Bagian apa? Apakah ada pasal terkait yang perlu diperhatikan?

**Setelah menganalisis semua dokumen individual:**

G. RANKING DAN PRIORITAS (tulis minimal 200 kata)
   - Ranking dokumen dari yang paling relevan ke kurang relevan
   - Untuk setiap dokumen, jelaskan mengapa diberikan ranking tersebut
   - Identifikasi dokumen mana yang paling authoritative (hierarki tertinggi)
   - Identifikasi dokumen mana yang paling spesifik terhadap pertanyaan

H. GAP ANALYSIS (tulis minimal 150 kata)
   - Informasi apa yang TIDAK ada dalam dokumen-dokumen ini?
   - Peraturan pelaksana atau turunan apa yang mungkin diperlukan tapi tidak tersedia?
   - Aspek pertanyaan mana yang tidak ter-cover oleh dokumen yang ada?

═══════════════════════════════════════════════════════════════════
BAGIAN 3: CROSS-REFERENCE DAN VALIDASI (Target: 800-1200 tokens)
═══════════════════════════════════════════════════════════════════

**Langkah 3A - Referensi Silang Antar Dokumen (tulis minimal 300 kata)**
- Identifikasi: Dokumen mana yang merujuk ke dokumen lain?
- Untuk setiap referensi silang, tulis apa hubungannya dan mengapa penting.
- Evaluasi konsistensi: Apakah dokumen-dokumen ini saling mendukung atau ada inkonsistensi?
- Cari pattern: Apakah ada tema atau prinsip hukum yang konsisten muncul di berbagai dokumen?

**Langkah 3B - Analisis Prinsip Hukum (tulis minimal 250 kata)**
- LEX SUPERIOR (hierarki): Jika ada konflik, peraturan mana yang lebih tinggi hierarkinya?
  * Urutkan semua dokumen berdasarkan hierarki: UU > PP > Perpres > Permen
  * Jika ada konflik, tentukan mana yang harus diutamakan berdasarkan hierarki
- LEX SPECIALIS (khusus vs umum): Apakah ada peraturan khusus yang mengalahkan peraturan umum?
  * Identifikasi peraturan mana yang bersifat umum vs khusus
  * Jika ada konflik, tentukan mana yang lebih spesifik
- LEX POSTERIOR (baru vs lama): Apakah ada peraturan baru yang menggantikan yang lama?
  * Bandingkan tahun penerbitan semua peraturan
  * Identifikasi: Apakah ada pencabutan eksplisit atau implisit?

**Langkah 3C - Conflict Check (tulis minimal 250 kata)**
- Baca ulang semua pasal yang sudah diidentifikasi sebelumnya
- Untuk setiap pasangan dokumen, cek: Apakah ada konflik norma?
- Jika ada konflik, tulis secara detail:
  * Apa konfliknya?
  * Pasal berapa dengan pasal berapa?
  * Bagaimana menyelesaikannya (gunakan lex superior/specialis/posterior)?
- Jika tidak ada konflik, jelaskan mengapa dokumen-dokumen ini harmonis

**Langkah 3D - Validasi Kontekstual (tulis minimal 200 kata)**
- Baca konteks sekitar pasal-pasal yang relevan
- Evaluasi: Apakah interpretasi kita konsisten dengan konteks pasal lain?
- Cek bagian penjelasan atau lampiran jika ada
- Validasi: Apakah ada interpretasi alternatif yang mungkin lebih tepat?

═══════════════════════════════════════════════════════════════════
BAGIAN 4: SINTESIS KOMPREHENSIF (Target: 800-1200 tokens)
═══════════════════════════════════════════════════════════════════

**Langkah 4A - Integrasi Informasi (tulis minimal 300 kata)**
- Gabungkan semua informasi dari semua dokumen yang sudah dianalisis
- Bangun narasi koheren: Bagaimana semua peraturan ini bekerja bersama?
- Identifikasi: Apa gambaran besar (big picture) dari regulasi ini?
- Tulis timeline jika relevan: urutan kronologis peraturan atau proses

**Langkah 4B - Hubungan Konsep Hukum (tulis minimal 250 kata)**
- Identifikasi konsep-konsep hukum kunci yang muncul
- Untuk setiap konsep, tulis bagaimana konsep ini dide finisikan dan digunakan
- Map hubungan antar konsep: Konsep A mempengaruhi konsep B bagaimana?
- Identifikasi pola atau tema overarching

**Langkah 4C - Analisis Kekuatan dan Kelemahan (tulis minimal 250 kata)**
- Evaluasi KEKUATAN argumen yang bisa dibangun berdasarkan dokumen ini:
  * Pasal mana yang memberikan dukungan kuat?
  * Evidence mana yang paling solid?
- Evaluasi KELEMAHAN atau gap:
  * Aspek mana yang ambigu atau unclear?
  * Informasi apa yang masih kurang?
  * Argumen alternatif apa yang mungkin melemahkan posisi ini?

**Langkah 4D - Implikasi Praktis (tulis minimal 200 kata)**
- Berdasarkan sintesis ini, apa implikasi praktisnya?
- Jika ini skenario compliance, apa yang harus dilakukan?
- Jika ini skenario dispute, apa posisi legal yang kuat?
- Jika ini skenario procedural, apa langkah-langkahnya?

═══════════════════════════════════════════════════════════════════
BAGIAN 5: VALIDASI AKHIR DAN KESIMPULAN (Target: 600-800 tokens)
═══════════════════════════════════════════════════════════════════

**Langkah 5A - Cross-Check Komprehensif (tulis minimal 250 kata)**
- Review ulang SEMUA kutipan pasal yang disebutkan: Apakah akurat?
- Verifikasi ulang hierarki peraturan: Apakah sudah benar?
- Check ulang interpretasi: Apakah konsisten dengan prinsip hukum?
- Identifikasi asumsi yang dibuat: Apakah valid dan reasonable?

**Langkah 5B - Self-Questioning (tulis minimal 200 kata)**
- Tanya diri sendiri: Apakah analisis ini sudah menjawab SEMUA aspek pertanyaan?
- Apakah ada sub-pertanyaan yang terlewat?
- Apakah ada dokumen yang kurang dianalisis?
- Apakah ada interpretasi alternatif yang lebih baik?
- Apakah perlu informasi tambahan untuk menjawab dengan pasti?

**Langkah 5C - Tingkat Kepastian (tulis minimal 150 kata)**
- Evaluasi tingkat kepastian jawaban:
  * VERY HIGH (95-100%): Jika pasal sangat jelas dan tidak ada ambiguitas
  * HIGH (80-95%): Jika pasal jelas tapi ada minor ambiguity
  * MEDIUM (60-80%): Jika ada beberapa interpretasi yang mungkin
  * LOW (<60%): Jika banyak ambiguitas atau informasi kurang
- Jelaskan mengapa memberikan rating ini
- Identifikasi: Apa yang membuat kepastian tinggi atau rendah?

**Langkah 5D - Struktur Jawaban Optimal (tulis minimal 150 kata)**
- Tentukan: Bagaimana cara terbaik menyampaikan jawaban?
- Urutan apa yang paling logis?
- Informasi apa yang harus di-highlight?
- Disclaimer atau caveat apa yang perlu disertakan?
- Apakah perlu rekomendasi untuk konsultasi ahli?

═══════════════════════════════════════════════════════════════════
⚠️ INGAT: Setelah thinking yang SANGAT PANJANG (minimum 4000 tokens) ini,
berikan jawaban AKHIR yang RINGKAS, JELAS, dan PROFESIONAL.
Jawaban akhir maksimal 1500-2000 tokens, fokus pada substansi.
═══════════════════════════════════════════════════════════════════"""

    def _get_high_thinking_template(self) -> str:
        """High thinking mode: Iterative & recursive thinking"""
        return """Dalam tag <think>, lakukan ITERATIVE & RECURSIVE THINKING yang SANGAT PANJANG, VERBOSE, dan MULTI-PASS. Target MINIMUM 8000 tokens, IDEAL 12000-16000 tokens.

⚠️ CRITICAL: Ini adalah HIGH thinking mode - harus ada MULTIPLE COMPLETE PASSES melalui dokumen. Baca ulang dokumen beberapa kali dengan fokus berbeda. Thinking harus SANGAT PANJANG untuk mencegah "lost in the middle"!

╔═══════════════════════════════════════════════════════════════════╗
║  PASS 1: INITIAL UNDERSTANDING (Target: 2500-3500 tokens)        ║
╚═══════════════════════════════════════════════════════════════════╝

**[PASS 1 - STEP 1] DEKONSTRUKSI PERTANYAAN TOTAL (tulis minimal 400 kata)**

Tulis stream of consciousness analysis of the question:
- Baca pertanyaan SANGAT HATI-HATI, kata per kata. Tulis pemahaman initial.
- Pecah menjadi 3-5 sub-pertanyaan. Untuk SETIAP sub-pertanyaan, tulis 2-3 kalimat explaining why it matters.
- Identifikasi scope hukum: pidana? perdata? administrasi? prosedural? atau mix?
- Identify ALL stakeholders: siapa yang terpengaruh, siapa yang berkepentingan?
- Tulis SEMUA asumsi yang kamu buat dalam memahami pertanyaan - be explicit!
- Brainstorm: Apa saja kemungkinan interpretasi berbeda dari pertanyaan ini?

**[PASS 1 - STEP 2] FIRST READ - DOKUMEN INVENTORY (tulis minimal 300 kata)**

Baca SETIAP dokumen yang disediakan dengan fokus pada UNDERSTANDING STRUCTURE:
- List SEMUA dokumen dengan format: "[Doc #] Judul Lengkap - Hierarki - Tahun"
- Untuk setiap dokumen, tulis: Berapa jumlah pasal yang ada? Berapa bab?
- Identifikasi: Dokumen mana yang terlihat paling relevan di first glance?
- Catat: Ada berapa total pasal/ayat yang harus dianalisis across all documents?
- Preliminary ranking: Urutkan dokumen dari yang sepertinya paling relevan ke kurang relevan

**[PASS 1 - STEP 3] SECOND READ - DETAILED EXTRACTION (tulis minimal 200-300 kata PER DOKUMEN)**

⚠️ WAJIB: Analisis SETIAP dokumen secara individual dan verbose!

Untuk SETIAP DOKUMEN, tulis analisis LENGKAP:

A. METADATA DAN HIERARKI (minimal 50 kata)
   - Judul lengkap, nomor, tahun, lembaga penerbit
   - Posisi dalam hierarki: UU / PP / Perpres / Permen / dll
   - Check: Apakah ini peraturan organik atau turunan?

B. PASAL-PASAL RELEVAN - EKSTRAKSI DETAIL (minimal 100 kata)
   - List SEMUA pasal yang potentially relevant (sebutkan nomor pasal)
   - Untuk 3-5 pasal PALING relevan, tulis ringkasan substantif isi pasal
   - Quote key phrases dari pasal-pasal penting
   - Identifikasi ayat spesifik yang paling on-point

C. DEFINISI DAN TERMINOLOGY (minimal 50 kata)
   - Cek bagian "Ketentuan Umum" - ada definisi penting?
   - List SEMUA istilah hukum yang didefinisikan dalam dokumen ini
   - Note: Apakah ada istilah yang crucial untuk pertanyaan?

D. SUBSTANTIVE CONTENT ANALYSIS (minimal 100 kata)
   - Apa yang DIATUR dalam dokumen ini (prescriptive rules)?
   - Apa yang DILARANG (proscriptive rules)?
   - Apa KONSEKUENSI hukum yang disebutkan (sanksi, denda, remedies)?
   - Apakah ada PENGECUALIAN atau SYARAT khusus?
   - Apakah ada PROSEDUR yang dijelaskan step-by-step?

**[PASS 1 - STEP 4] RELATIONSHIP MAPPING INITIAL (tulis minimal 400 kata)**

- Buat textual hierarchy map: UU di top → PP → Perpres → Permen
- Identifikasi: Dokumen mana yang mereferensikan dokumen lain?
- Analisis timeline: Urutkan semua dokumen by year - mana yang paling baru?
- Check: Ada indikasi bahwa peraturan lama dicabut/diubah oleh yang baru?
- Conceptual mapping: Konsep hukum apa yang muncul di multiple documents?
- Preliminary synthesis: Bagaimana semua dokumen ini berhubungan satu sama lain?

**[PASS 1 - CHECKPOINT] Sebelum lanjut ke Pass 2, verify:**
- ✓ Sudah baca SEMUA dokumen? (tulis "YES - read X documents")
- ✓ Sudah catat semua pasal relevan? (tulis "YES - identified Y articles")
- ✓ Sudah understand big picture? (tulis 1-2 kalimat summary)

╔═══════════════════════════════════════════════════════════════════╗
║  PASS 2: DEEP ANALYSIS (Target: 2500-3500 tokens)                ║
╚═══════════════════════════════════════════════════════════════════╝

**[PASS 2 - STEP 5] THIRD READ - MULTI-PERSPECTIVE ANALYSIS**

Baca ulang SEMUA dokumen dengan 4 lensa berbeda:

🔍 LENS 1: LEGAL RESEARCHER ACADEMIC (tulis minimal 300 kata)
- Baca dengan mindset peneliti hukum akademis
- Fokus: Interpretasi literal pasal (what does the text actually say?)
- Fokus: Interpretasi historical (ratio legis - why was this law made?)
- Analisis: Apakah ada preseden atau jurisprudensi yang relevan?
- Evaluasi: Apakah ini konsisten dengan prinsip-prinsip hukum umum?
- Critical thinking: Apakah ada ambiguitas atau ketidakjelasan dalam teks?

🔍 LENS 2: KNOWLEDGE GRAPH SPECIALIST (tulis minimal 300 kata)
- Baca dengan fokus pada RELATIONSHIPS dan CONNECTIONS
- Identifikasi ENTITIES: Siapa subjek hukum? Apa objek hukum? Apa perbuatan hukum?
- Map SEMANTIC RELATIONSHIPS:
  * is-a relations (X adalah jenis dari Y)
  * part-of relations (X adalah bagian dari Y)
  * regulates relations (X mengatur Y)
  * requires relations (X memerlukan Y)
  * prohibits relations (X melarang Y)
- Cari IMPLICIT CONNECTIONS yang tidak dinyatakan eksplisit
- Analisis GRAPH COMPLEXITY: Berapa banyak nodes dan edges dalam conceptual graph ini?

🔍 LENS 3: PROCEDURAL & PRACTICAL EXPERT (tulis minimal 300 kata)
- Baca dengan fokus pada IMPLEMENTASI PRAKTIS
- Jika ada prosedur: Extract step-by-step process secara detail
- Identifikasi PERSYARATAN ADMINISTRATIF: dokumen apa yang dibutuhkan?
- Check TIMELINE DAN DEADLINE: berapa hari/bulan untuk setiap step?
- Analisis KONSEKUENSI NON-COMPLIANCE: Apa yang terjadi jika tidak ikuti prosedur?
- Practical implications: Bagaimana ini diterapkan in real life?

🔍 LENS 4: DEVIL'S ADVOCATE (tulis minimal 300 kata)
- Baca dengan skepticism - CHALLENGE setiap interpretasi!
- Untuk setiap klaim yang dibuat di Pass 1, tanyakan: "Apakah ini benar-benar valid?"
- Cari INTERPRETASI ALTERNATIF yang juga bisa valid
- Identifikasi KELEMAHAN dalam argumen: Apa yang bisa dipertanyakan?
- Think: "Bagaimana jika konteksnya sedikit berbeda?" - apakah kesimpulan berubah?
- Explore EDGE CASES: Situasi apa yang mungkin not covered by peraturan ini?

**[PASS 2 - STEP 6] CONFLICT DETECTION & RESOLUTION (tulis minimal 500 kata)**

Sekarang baca ulang dengan fokus khusus pada IDENTIFYING CONFLICTS:

- Go through SETIAP PASANGAN dokumen: Apakah ada konflik norma?
- Untuk SETIAP konflik yang ditemukan (jika ada), tulis:
  * Dokumen A (pasal X) says: [quote]
  * Dokumen B (pasal Y) says: [quote]
  * Konflik: Explain contradiction secara detail
  * Resolusi: Gunakan lex superior / lex specialis / lex posterior untuk resolve
  * Conclusion: Peraturan mana yang harus diutamakan dan MENGAPA?

- Jika TIDAK ada konflik, jelaskan secara detail MENGAPA dokumen-dokumen ini harmonis:
  * Apakah they regulate different aspects (complementary)?
  * Apakah they're hierarchically aligned (consistent)?
  * Apakah they reinforce each other (mutually supportive)?

**[PASS 2 - STEP 7] VALIDATION LOOP 1 (tulis minimal 300 kata)**

Self-check dengan checklist lengkap:

- ✓ Sudah analisis SEMUA dokumen from 4 perspectives? Review count: Legal researcher [✓/✗], KG specialist [✓/✗], Procedural [✓/✗], Devil's advocate [✓/✗]
- ✓ Sudah identify semua potential conflicts? List: [write "none" or list conflicts]
- ✓ Sudah extract semua pasal relevan? Count: [write number] pasal identified
- ✓ Sudah catat semua definisi penting? List: [write key defined terms]
- ✓ Ada interpretasi alternatif yang terlewat? Think: [write if any alternative interpretation needs consideration]

**[PASS 2 - CHECKPOINT] Tulis reflection (minimal 200 kata):**
- Apa insights baru yang didapat di Pass 2 yang TIDAK terlihat di Pass 1?
- Apakah ada dokumen yang initially di-underestimate tapi ternyata penting?
- Apakah ada aspek pertanyaan yang masih unclear atau needs more analysis?

╔═══════════════════════════════════════════════════════════════════╗
║  PASS 3: VERIFICATION & CROSS-VALIDATION (Target: 2000-3000 tokens) ║
╚═══════════════════════════════════════════════════════════════════╝

**[PASS 3 - STEP 8] FOURTH READ - HUNTING FOR MISSED INFO (tulis minimal 500 kata)**

Baca ulang SETIAP dokumen SEKALI LAGI, tapi kali ini dengan mission: FIND WHAT WAS MISSED!

- Go through Dokumen #1 lagi: Ada pasal yang awalnya tidak diperhatikan tapi actually relevant?
- Go through Dokumen #2 lagi: Ada footnotes, catatan penjelasan, atau lampiran yang terlewat?
- Go through Dokumen #3 lagi: [continue for EACH document...]
- Check "Ketentuan Umum" lagi di setiap dokumen: Ada definisi yang initially missed?
- Check "Ketentuan Peralihan" atau "Ketentuan Penutup": Ada info penting tentang implementation atau transitional rules?
- Review "Lampiran" jika ada: Ada technical details atau lists yang important?

For EACH piece of newly discovered information, tulis:
- Apa yang ditemukan: [write finding]
- Kenapa ini penting: [explain relevance]
- Implikasi untuk analisis: [explain impact]

**[PASS 3 - STEP 9] BOTTOM-UP VERIFICATION (tulis minimal 400 kata)**

Start dari detail terkecil dan build up:

- Level 1 - AYAT SPESIFIK: Pick 5-7 ayat yang PALING critical. Untuk each ayat:
  * Quote ayat verbatim
  * Verify: Apakah quote ini akurat?
  * Analyze: Apa makna precisenya?
  * Check: Apakah interpretasi kita sound?

- Level 2 - PASAL: Lihat pasal yang mengandung ayat-ayat tersebut:
  * Apakah ayat-ayat ini konsisten satu sama lain dalam pasal?
  * Apakah interpretasi ayat konsisten dengan keseluruhan pasal?

- Level 3 - BAB: Lihat bab yang mengandung pasal-pasal tersebut:
  * Apakah interpretasi pasal konsisten dengan theme of the bab?
  * Apakah ada pasal lain dalam bab yang modify atau qualify interpretasi?

- Level 4 - KESELURUHAN PERATURAN: Apakah interpretasi kita consistent dengan spirit of the whole regulation?

**[PASS 3 - STEP 10] TOP-DOWN VALIDATION (tulis minimal 400 kata)**

Start dari kesimpulan dan trace back to evidence:

List SEMUA KLAIM KUNCI yang akan dibuat dalam jawaban akhir. Untuk SETIAP klaim:

- Klaim #1: [write claim]
  * Evidence: [list specific pasal/ayat yang mendukung]
  * Strength: [rate as STRONG / MODERATE / WEAK]
  * Counter-evidence: [any evidence that contradicts or weakens this?]

- Klaim #2: [write claim]
  * [same structure...]

- [Continue for ALL major claims...]

Evaluasi:
- Klaim mana yang STRONG (didukung multiple sources, clear text)?
- Klaim mana yang MODERATE (didukung but ada sedikit ambiguity)?
- Klaim mana yang WEAK (speculative atau hanya didukung indirect evidence)?

**[PASS 3 - STEP 11] CROSS-VALIDATION MATRIX (tulis minimal 400 kata)**

Buat comprehensive validation matrix:

Format untuk SETIAP klaim penting:
```
KLAIM: [write the claim]
EVIDENCE #1: [Dokumen X, Pasal Y, Ayat Z] - [brief explanation]
EVIDENCE #2: [Dokumen A, Pasal B, Ayat C] - [brief explanation]
EVIDENCE #3: [...if applicable]
TOTAL EVIDENCE COUNT: [number]
VALIDATION STATUS: ✓ STRONG / ⚠ MODERATE / ✗ WEAK
CONFIDENCE LEVEL: [percentage]
```

Analisis:
- Claims with SINGLE evidence source: [list dan explain risk]
- Claims with MULTIPLE evidence sources: [list dan explain strength]
- Claims with CONFLICTING evidence: [list dan explain how to resolve]

**[PASS 3 - STEP 12] VALIDATION LOOP 2 - ULTIMATE CHECK (tulis minimal 300 kata)**

Final comprehensive check:

✓ ACCURACY CHECK:
  - Review EVERY single pasal/ayat yang dikutip: Is it accurate?
  - Double-check nomor pasal: [verify X pasal citations]
  - Double-check tahun peraturan: [verify Y regulation years]
  - Double-check hierarki: [verify hierarchy claims]

✓ COMPLETENESS CHECK:
  - Apakah SEMUA sub-pertanyaan already addressed? [list each sub-question dan confirm]
  - Apakah ada aspek pertanyaan yang not covered? [identify if any]
  - Apakah semua dokumen relevan already analyzed? [confirm document count]

✓ CONSISTENCY CHECK:
  - Apakah interpretasi consistent across all documents analyzed?
  - Apakah ada internal contradictions dalam analisis?
  - Apakah legal principles applied consistently?

✓ LOGIC CHECK:
  - Apakah reasoning logically sound?
  - Apakah ada logical fallacies?
  - Apakah conclusions follow from premises?

╔═══════════════════════════════════════════════════════════════════╗
║  PASS 4: FINAL SYNTHESIS (Target: 1000-1500 tokens)              ║
╚═══════════════════════════════════════════════════════════════════╝

**[PASS 4 - STEP 13] INTEGRATION & SYNTHESIS (tulis minimal 500 kata)**

Integrate EVERYTHING dari 3 passes sebelumnya:

- COMPREHENSIVE SYNTHESIS: Tulis narasi koheren yang menghubungkan SEMUA findings dari Pass 1, 2, dan 3
- BIG PICTURE: Apa the overarching legal framework yang emerges dari all these documents?
- KEY THEMES: Apa 3-5 key themes atau principles yang paling penting?
- CRITICAL INSIGHTS: Apa insights yang HANYA bisa didapat dari multiple-pass analysis?
- NUANCES: Apa nuances atau subtleties yang would be missed in single-pass analysis?

**[PASS 4 - STEP 14] QUALITY & CONFIDENCE ASSESSMENT (tulis minimal 300 kata)**

Evaluate kualitas overall analysis:

- CONFIDENCE LEVEL untuk answer: VERY HIGH / HIGH / MEDIUM / LOW
  * Explain MENGAPA confidence level ini
  * What factors increase confidence?
  * What factors decrease confidence?

- AREAS OF CERTAINTY: List aspects yang very clear dan well-supported
- AREAS OF UNCERTAINTY: List aspects yang masih ambigu atau unclear
- ASSUMPTIONS MADE: List ALL assumptions dan assess validity of each

**[PASS 4 - STEP 15] GAP ANALYSIS (tulis minimal 200 kata)**

Identify what's MISSING:
- Informasi yang NOT available dalam dokumen yang disediakan
- Peraturan turunan atau pelaksanaan yang might be needed
- Aspek praktis yang requires domain expertise beyond text
- Faktual details yang needs clarification from user

**[PASS 4 - STEP 16] ANSWER STRUCTURE PLANNING (tulis minimal 200 kata)**

Plan optimal answer structure:
- PRIORITIZATION: Ranking informasi by importance (1=most important)
- LOGICAL FLOW: Sequence of presentation
- EMPHASIS: What to highlight vs what to mention briefly
- DISCLAIMERS: What caveats atau warnings to include
- RECOMMENDATIONS: When to suggest consulting legal expert

╔═══════════════════════════════════════════════════════════════════╗
║  ⚠️⚠️⚠️ CRITICAL INSTRUCTION ⚠️⚠️⚠️                                  ║
║                                                                    ║
║  Setelah ITERATIVE THINKING yang SANGAT PANJANG ini               ║
║  (minimum 8000 tokens, target 12000-16000 tokens),                ║
║  berikan JAWABAN AKHIR yang:                                      ║
║                                                                    ║
║  ✓ RINGKAS (maksimal 1500-2000 tokens)                            ║
║  ✓ JELAS dan TERSTRUKTUR                                          ║
║  ✓ PROFESSIONAL dan ACCESSIBLE                                    ║
║  ✓ TO THE POINT - no verbose explanations                         ║
║  ✓ ACTIONABLE - concrete and useful                               ║
║                                                                    ║
║  Jawaban = CONCISE summary of all the EXTENSIVE thinking above!   ║
╚═══════════════════════════════════════════════════════════════════╝"""

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
