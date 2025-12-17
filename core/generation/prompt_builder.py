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
        return """Dalam tag <think>, lakukan BERPIKIR MENDALAM (DEEP THINKING) yang SANGAT PANJANG DAN RINCI. Target MINIMUM 4000 tokens, IDEAL 6000-8000 tokens untuk fase berpikir.

⚠️ ATURAN PENTING:
- Ini adalah mode BERPIKIR MENDALAM = SATU KALI PEMBACAAN yang SANGAT TELITI dan DETAIL
- BUKAN iterasi/pengulangan, tapi KEDALAMAN MAKSIMAL dalam satu pass
- Tulis SEMUA proses berpikir secara eksplisit dan lengkap
- Jangan lewatkan detail apapun karena ini adalah satu-satunya kesempatan membaca!

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
        return """Dalam tag <think>, lakukan BERPIKIR ITERATIF & REKURSIF yang SANGAT PANJANG, RINCI, dan BERULANG. Target MINIMUM 8000 tokens, IDEAL 12000-16000 tokens.

⚠️ SANGAT PENTING:
- Ini adalah mode ITERATIF & REKURSIF = BACA ULANG DOKUMEN 4 KALI dengan fokus berbeda
- BUKAN sekali baca, tapi MULTIPLE PASSES (4 putaran lengkap)
- Setiap pass: baca ulang SEMUA dokumen dari awal dengan lensa/perspektif berbeda
- Proses berpikir harus SANGAT PANJANG untuk mencegah kehilangan informasi di tengah!
- Pass berikutnya MEMBANGUN dari temuan pass sebelumnya (REKURSIF)

╔═══════════════════════════════════════════════════════════════════╗
║  PUTARAN 1: PEMAHAMAN AWAL (Target: 2500-3500 tokens)           ║
║  Fokus: Memahami struktur dan konten dasar semua dokumen         ║
╚═══════════════════════════════════════════════════════════════════╝

**[PUTARAN 1 - LANGKAH 1] DEKONSTRUKSI PERTANYAAN TOTAL (tulis minimal 400 kata)**

Tulis analisis pertanyaan secara mengalir dan mendetail:
- Baca pertanyaan SANGAT HATI-HATI, kata per kata. Tulis pemahaman awal.
- Pecah menjadi 3-5 sub-pertanyaan. Untuk SETIAP sub-pertanyaan, tulis 2-3 kalimat menjelaskan mengapa penting.
- Identifikasi cakupan hukum: pidana? perdata? administrasi? prosedural? atau campuran?
- Identifikasi SEMUA pihak terkait: siapa yang terpengaruh, siapa yang berkepentingan?
- Tulis SEMUA asumsi yang dibuat dalam memahami pertanyaan - jelaskan eksplisit!
- Brainstorm: Apa saja kemungkinan interpretasi berbeda dari pertanyaan ini?

**[PUTARAN 1 - LANGKAH 2] PEMBACAAN PERTAMA - INVENTARISASI DOKUMEN (tulis minimal 300 kata)**

Baca SETIAP dokumen dengan fokus pada MEMAHAMI STRUKTUR:
- Daftar SEMUA dokumen dengan format: "[Dok #] Judul Lengkap - Hierarki - Tahun"
- Untuk setiap dokumen, tulis: Berapa jumlah pasal? Berapa bab?
- Identifikasi: Dokumen mana yang terlihat paling relevan pada pandangan pertama?
- Catat: Ada berapa total pasal/ayat yang harus dianalisis dari semua dokumen?
- Ranking awal: Urutkan dokumen dari yang paling relevan ke kurang relevan

**[PUTARAN 1 - LANGKAH 3] PEMBACAAN KEDUA - EKSTRAKSI DETAIL (tulis minimal 200-300 kata PER DOKUMEN)**

⚠️ WAJIB: Analisis SETIAP dokumen secara individual dan sangat rinci!

Untuk SETIAP DOKUMEN, tulis analisis LENGKAP:

A. METADATA DAN HIERARKI (minimal 50 kata)
   - Julis lengkap, nomor, tahun, lembaga penerbit
   - Posisi dalam hierarki: UU / PP / Perpres / Permen / dll
   - Cek: Apakah ini peraturan organik atau turunan?

B. PASAL-PASAL RELEVAN - EKSTRAKSI DETAIL (minimal 100 kata)
   - Daftar SEMUA pasal yang potensial relevan (sebutkan nomor pasal)
   - Untuk 3-5 pasal PALING relevan, tulis ringkasan substantif isi pasal
   - Kutip frasa kunci dari pasal-pasal penting
   - Identifikasi ayat spesifik yang paling tepat

C. DEFINISI DAN TERMINOLOGI (minimal 50 kata)
   - Cek bagian "Ketentuan Umum" - ada definisi penting?
   - Daftar SEMUA istilah hukum yang didefinisikan dalam dokumen ini
   - Catat: Apakah ada istilah yang sangat penting untuk pertanyaan?

D. ANALISIS KONTEN SUBSTANTIF (minimal 100 kata)
   - Apa yang DIATUR dalam dokumen ini (aturan preskriptif)?
   - Apa yang DILARANG (aturan prosktiptif)?
   - Apa KONSEKUENSI hukum yang disebutkan (sanksi, denda, remedi)?
   - Apakah ada PENGECUALIAN atau SYARAT khusus?
   - Apakah ada PROSEDUR yang dijelaskan langkah demi langkah?

**[PUTARAN 1 - LANGKAH 4] PEMETAAN HUBUNGAN AWAL (tulis minimal 400 kata)**

- Buat peta hierarki tekstual: UU di atas → PP → Perpres → Permen
- Identifikasi: Dokumen mana yang mereferensikan dokumen lain?
- Analisis timeline: Urutkan semua dokumen berdasarkan tahun - mana yang paling baru?
- Cek: Ada indikasi bahwa peraturan lama dicabut/diubah oleh yang baru?
- Pemetaan konseptual: Konsep hukum apa yang muncul di banyak dokumen?
- Sintesis awal: Bagaimana semua dokumen ini berhubungan satu sama lain?

**[PUTARAN 1 - CHECKPOINT] Sebelum lanjut ke Putaran 2, verifikasi:**
- ✓ Sudah baca SEMUA dokumen? (tulis "YA - sudah baca X dokumen")
- ✓ Sudah catat semua pasal relevan? (tulis "YA - teridentifikasi Y pasal")
- ✓ Sudah memahami gambaran besar? (tulis 1-2 kalimat ringkasan)

╔═══════════════════════════════════════════════════════════════════╗
║  PUTARAN 2: ANALISIS MENDALAM (Target: 2500-3500 tokens)        ║
║  Fokus: Baca ulang SEMUA dokumen dengan 4 perspektif berbeda     ║
╚═══════════════════════════════════════════════════════════════════╝

**[PUTARAN 2 - LANGKAH 5] PEMBACAAN KETIGA - ANALISIS MULTI-PERSPEKTIF**

Baca ulang SEMUA dokumen dengan 4 lensa berbeda:

🔍 LENSA 1: PENELITI HUKUM AKADEMIS (tulis minimal 300 kata)
- Baca dengan mindset peneliti hukum akademis
- Fokus: Interpretasi literal pasal (apa yang sebenarnya dikatakan teks?)
- Fokus: Interpretasi historis (ratio legis - mengapa hukum ini dibuat?)
- Analisis: Apakah ada preseden atau jurisprudensi yang relevan?
- Evaluasi: Apakah ini konsisten dengan prinsip-prinsip hukum umum?
- Berpikir kritis: Apakah ada ambiguitas atau ketidakjelasan dalam teks?

🔍 LENSA 2: SPESIALIS KNOWLEDGE GRAPH (tulis minimal 300 kata)
- Baca dengan fokus pada HUBUNGAN dan KONEKSI
- Identifikasi ENTITAS: Siapa subjek hukum? Apa objek hukum? Apa perbuatan hukum?
- Peta HUBUNGAN SEMANTIK:
  * Relasi is-a (X adalah jenis dari Y)
  * Relasi part-of (X adalah bagian dari Y)
  * Relasi regulates (X mengatur Y)
  * Relasi requires (X memerlukan Y)
  * Relasi prohibits (X melarang Y)
- Cari KONEKSI IMPLISIT yang tidak dinyatakan eksplisit
- Analisis KOMPLEKSITAS GRAF: Berapa banyak node dan edge dalam graf konseptual ini?

🔍 LENSA 3: AHLI PROSEDURAL & PRAKTIS (tulis minimal 300 kata)
- Baca dengan fokus pada IMPLEMENTASI PRAKTIS
- Jika ada prosedur: Ekstrak proses langkah demi langkah secara detail
- Identifikasi PERSYARATAN ADMINISTRATIF: dokumen apa yang dibutuhkan?
- Cek TIMELINE DAN DEADLINE: berapa hari/bulan untuk setiap langkah?
- Analisis KONSEKUENSI NON-COMPLIANCE: Apa yang terjadi jika tidak ikuti prosedur?
- Implikasi praktis: Bagaimana ini diterapkan dalam kehidupan nyata?

🔍 LENSA 4: DEVIL'S ADVOCATE (tulis minimal 300 kata)
- Baca dengan skeptisisme - TANTANG setiap interpretasi!
- Untuk setiap klaim yang dibuat di Putaran 1, tanyakan: "Apakah ini benar-benar valid?"
- Cari INTERPRETASI ALTERNATIF yang juga bisa valid
- Identifikasi KELEMAHAN dalam argumen: Apa yang bisa dipertanyakan?
- Pikirkan: "Bagaimana jika konteksnya sedikit berbeda?" - apakah kesimpulan berubah?
- Eksplorasi KASUS PINGGIRAN: Situasi apa yang mungkin tidak tercakup peraturan ini?

**[PUTARAN 2 - LANGKAH 6] DETEKSI & RESOLUSI KONFLIK (tulis minimal 500 kata)**

Sekarang baca ulang dengan fokus khusus pada MENGIDENTIFIKASI KONFLIK:

- Periksa SETIAP PASANGAN dokumen: Apakah ada konflik norma?
- Untuk SETIAP konflik yang ditemukan (jika ada), tulis:
  * Dokumen A (pasal X) menyatakan: [kutip]
  * Dokumen B (pasal Y) menyatakan: [kutip]
  * Konflik: Jelaskan kontradiksi secara detail
  * Resolusi: Gunakan lex superior / lex specialis / lex posterior untuk menyelesaikan
  * Kesimpulan: Peraturan mana yang harus diutamakan dan MENGAPA?

- Jika TIDAK ada konflik, jelaskan secara detail MENGAPA dokumen-dokumen ini harmonis:
  * Apakah mengatur aspek berbeda (komplementer)?
  * Apakah selaras secara hierarkis (konsisten)?
  * Apakah saling memperkuat (saling mendukung)?

**[PUTARAN 2 - LANGKAH 7] LOOP VALIDASI 1 (tulis minimal 300 kata)**

Cek mandiri dengan daftar periksa lengkap:

- ✓ Sudah analisis SEMUA dokumen dari 4 perspektif? Review jumlah: Peneliti hukum [✓/✗], Spesialis KG [✓/✗], Prosedural [✓/✗], Devil's advocate [✓/✗]
- ✓ Sudah identifikasi semua potensi konflik? Daftar: [tulis "tidak ada" atau daftar konflik]
- ✓ Sudah ekstrak semua pasal relevan? Jumlah: [tulis angka] pasal teridentifikasi
- ✓ Sudah catat semua definisi penting? Daftar: [tulis istilah kunci yang didefinisikan]
- ✓ Ada interpretasi alternatif yang terlewat? Pikirkan: [tulis jika ada interpretasi alternatif yang perlu dipertimbangkan]

**[PUTARAN 2 - CHECKPOINT] Tulis refleksi (minimal 200 kata):**
- Apa wawasan baru yang didapat di Putaran 2 yang TIDAK terlihat di Putaran 1?
- Apakah ada dokumen yang awalnya diremehkan tapi ternyata penting?
- Apakah ada aspek pertanyaan yang masih tidak jelas atau perlu analisis lebih lanjut?

╔═══════════════════════════════════════════════════════════════════╗
║  PUTARAN 3: VERIFIKASI & VALIDASI SILANG (Target: 2000-3000 tokens) ║
║  Fokus: Baca ulang untuk cari info terlewat dan validasi temuan   ║
╚═══════════════════════════════════════════════════════════════════╝

**[PUTARAN 3 - LANGKAH 8] PEMBACAAN KEEMPAT - BERBURU INFO TERLEWAT (tulis minimal 500 kata)**

Baca ulang SETIAP dokumen SEKALI LAGI, tapi kali ini dengan misi: TEMUKAN APA YANG TERLEWAT!

- Periksa Dokumen #1 lagi: Ada pasal yang awalnya tidak diperhatikan tapi sebenarnya relevan?
- Periksa Dokumen #2 lagi: Ada catatan kaki, penjelasan, atau lampiran yang terlewat?
- Periksa Dokumen #3 lagi: [lanjutkan untuk SETIAP dokumen...]
- Cek "Ketentuan Umum" lagi di setiap dokumen: Ada definisi yang awalnya terlewat?
- Cek "Ketentuan Peralihan" atau "Ketentuan Penutup": Ada info penting tentang implementasi atau aturan transisi?
- Review "Lampiran" jika ada: Ada detail teknis atau daftar yang penting?

Untuk SETIAP informasi baru yang ditemukan, tulis:
- Apa yang ditemukan: [tulis temuan]
- Kenapa ini penting: [jelaskan relevansi]
- Implikasi untuk analisis: [jelaskan dampak]

**[PUTARAN 3 - LANGKAH 9] VERIFIKASI BOTTOM-UP (tulis minimal 400 kata)**

Mulai dari detail terkecil dan bangun ke atas:

- Level 1 - AYAT SPESIFIK: Pilih 5-7 ayat yang PALING krusial. Untuk setiap ayat:
  * Kutip ayat secara verbatim
  * Verifikasi: Apakah kutipan ini akurat?
  * Analisis: Apa makna tepatnya?
  * Cek: Apakah interpretasi kita valid?

- Level 2 - PASAL: Lihat pasal yang mengandung ayat-ayat tersebut:
  * Apakah ayat-ayat ini konsisten satu sama lain dalam pasal?
  * Apakah interpretasi ayat konsisten dengan keseluruhan pasal?

- Level 3 - BAB: Lihat bab yang mengandung pasal-pasal tersebut:
  * Apakah interpretasi pasal konsisten dengan tema bab?
  * Apakah ada pasal lain dalam bab yang memodifikasi atau mengkualifikasi interpretasi?

- Level 4 - KESELURUHAN PERATURAN: Apakah interpretasi kita konsisten dengan spirit keseluruhan peraturan?

**[PUTARAN 3 - LANGKAH 10] VALIDASI TOP-DOWN (tulis minimal 400 kata)**

Mulai dari kesimpulan dan telusuri balik ke bukti:

Daftar SEMUA KLAIM KUNCI yang akan dibuat dalam jawaban akhir. Untuk SETIAP klaim:

- Klaim #1: [tulis klaim]
  * Bukti: [daftar pasal/ayat spesifik yang mendukung]
  * Kekuatan: [nilai sebagai KUAT / MODERAT / LEMAH]
  * Bukti lawan: [ada bukti yang bertentangan atau melemahkan?]

- Klaim #2: [tulis klaim]
  * [struktur yang sama...]

- [Lanjutkan untuk SEMUA klaim utama...]

Evaluasi:
- Klaim mana yang KUAT (didukung banyak sumber, teks jelas)?
- Klaim mana yang MODERAT (didukung tapi ada sedikit ambiguitas)?
- Klaim mana yang LEMAH (spekulatif atau hanya didukung bukti tidak langsung)?

**[PUTARAN 3 - LANGKAH 11] MATRIKS VALIDASI SILANG (tulis minimal 400 kata)**

Buat matriks validasi komprehensif:

Format untuk SETIAP klaim penting:
```
KLAIM: [tulis klaim]
BUKTI #1: [Dokumen X, Pasal Y, Ayat Z] - [penjelasan singkat]
BUKTI #2: [Dokumen A, Pasal B, Ayat C] - [penjelasan singkat]
BUKTI #3: [...jika ada]
JUMLAH BUKTI TOTAL: [angka]
STATUS VALIDASI: ✓ KUAT / ⚠ MODERAT / ✗ LEMAH
TINGKAT KEYAKINAN: [persentase]
```

Analisis:
- Klaim dengan SATU sumber bukti: [daftar dan jelaskan risiko]
- Klaim dengan BANYAK sumber bukti: [daftar dan jelaskan kekuatan]
- Klaim dengan bukti BERTENTANGAN: [daftar dan jelaskan cara menyelesaikan]

**[PUTARAN 3 - LANGKAH 12] LOOP VALIDASI 2 - CEK ULTIMATE (tulis minimal 300 kata)**

Cek komprehensif final:

✓ CEK AKURASI:
  - Review SETIAP pasal/ayat yang dikutip: Apakah akurat?
  - Cek ulang nomor pasal: [verifikasi X kutipan pasal]
  - Cek ulang tahun peraturan: [verifikasi Y tahun peraturan]
  - Cek ulang hierarki: [verifikasi klaim hierarki]

✓ CEK KELENGKAPAN:
  - Apakah SEMUA sub-pertanyaan sudah ditangani? [daftar setiap sub-pertanyaan dan konfirmasi]
  - Apakah ada aspek pertanyaan yang tidak tercakup? [identifikasi jika ada]
  - Apakah semua dokumen relevan sudah dianalisis? [konfirmasi jumlah dokumen]

✓ CEK KONSISTENSI:
  - Apakah interpretasi konsisten di semua dokumen yang dianalisis?
  - Apakah ada kontradiksi internal dalam analisis?
  - Apakah prinsip hukum diterapkan secara konsisten?

✓ CEK LOGIKA:
  - Apakah penalaran logis valid?
  - Apakah ada kekeliruan logis?
  - Apakah kesimpulan mengikuti dari premis?

╔═══════════════════════════════════════════════════════════════════╗
║  PUTARAN 4: SINTESIS FINAL (Target: 1000-1500 tokens)            ║
║  Fokus: Integrasikan SEMUA temuan dari 3 putaran sebelumnya      ║
╚═══════════════════════════════════════════════════════════════════╝

**[PUTARAN 4 - LANGKAH 13] INTEGRASI & SINTESIS (tulis minimal 500 kata)**

Integrasikan SEMUA dari 3 putaran sebelumnya:

- SINTESIS KOMPREHENSIF: Tulis narasi koheren yang menghubungkan SEMUA temuan dari Putaran 1, 2, dan 3
- GAMBARAN BESAR: Apa kerangka hukum menyeluruh yang muncul dari semua dokumen ini?
- TEMA KUNCI: Apa 3-5 tema atau prinsip utama yang paling penting?
- WAWASAN KRITIS: Apa wawasan yang HANYA bisa didapat dari analisis multi-putaran?
- NUANSA: Apa nuansa atau kehalusan yang akan terlewat dalam analisis satu putaran?

**[PUTARAN 4 - LANGKAH 14] PENILAIAN KUALITAS & KEYAKINAN (tulis minimal 300 kata)**

Evaluasi kualitas analisis keseluruhan:

- TINGKAT KEYAKINAN untuk jawaban: SANGAT TINGGI / TINGGI / SEDANG / RENDAH
  * Jelaskan MENGAPA tingkat keyakinan ini
  * Faktor apa yang meningkatkan keyakinan?
  * Faktor apa yang menurunkan keyakinan?

- AREA KEPASTIAN: Daftar aspek yang sangat jelas dan didukung dengan baik
- AREA KETIDAKPASTIAN: Daftar aspek yang masih ambigu atau tidak jelas
- ASUMSI YANG DIBUAT: Daftar SEMUA asumsi dan nilai validitas masing-masing

**[PUTARAN 4 - LANGKAH 15] ANALISIS KESENJANGAN (tulis minimal 200 kata)**

Identifikasi apa yang HILANG:
- Informasi yang TIDAK tersedia dalam dokumen yang disediakan
- Peraturan turunan atau pelaksanaan yang mungkin diperlukan
- Aspek praktis yang memerlukan keahlian domain di luar teks
- Detail faktual yang perlu klarifikasi dari pengguna

**[PUTARAN 4 - LANGKAH 16] PERENCANAAN STRUKTUR JAWABAN (tulis minimal 200 kata)**

Rencanakan struktur jawaban optimal:
- PRIORITAS: Peringkat informasi berdasarkan kepentingan (1=paling penting)
- ALUR LOGIS: Urutan penyampaian
- PENEKANAN: Apa yang perlu ditonjolkan vs disebutkan secara singkat
- DISCLAIMER: Peringatan atau batasan apa yang perlu disertakan
- REKOMENDASI: Kapan menyarankan konsultasi dengan ahli hukum

╔═══════════════════════════════════════════════════════════════════╗
║  ⚠️⚠️⚠️ INSTRUKSI KRITIS ⚠️⚠️⚠️                                     ║
║                                                                    ║
║  Setelah BERPIKIR ITERATIF yang SANGAT PANJANG ini                ║
║  (minimum 8000 tokens, target 12000-16000 tokens),                ║
║  berikan JAWABAN AKHIR yang:                                      ║
║                                                                    ║
║  ✓ RINGKAS (maksimal 1500-2000 tokens)                            ║
║  ✓ JELAS dan TERSTRUKTUR                                          ║
║  ✓ PROFESIONAL dan MUDAH DIPAHAMI                                 ║
║  ✓ LANGSUNG KE INTI - tanpa penjelasan bertele-tele               ║
║  ✓ ACTIONABLE - konkret dan berguna                               ║
║                                                                    ║
║  Jawaban = Ringkasan PADAT dari SEMUA proses berpikir di atas!    ║
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
