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
                'description': 'Analisis dasar untuk pertanyaan sederhana',
                'instruction_template': self._get_low_thinking_template()
            },
            ThinkingMode.MEDIUM: {
                'name': 'Medium Thinking Mode',
                'description': 'Berpikir mendalam untuk kompleksitas sedang',
                'instruction_template': self._get_medium_thinking_template()
            },
            ThinkingMode.HIGH: {
                'name': 'High Thinking Mode',
                'description': 'Berpikir iteratif & rekursif untuk analisis kompleks',
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
                "mode": thinking_config['mode']
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
            'instructions': config['instruction_template']
        }

    def _get_low_thinking_template(self) -> str:
        """Low thinking mode: Basic analysis"""
        return """Dalam tag <think>, lakukan analisis DASAR dengan struktur berikut:

1. **PEMAHAMAN PERTANYAAN**
   - Identifikasi inti pertanyaan
   - Tentukan jenis pertanyaan (definitional, procedural, sanctions, dll)
   - Catat kata kunci penting

2. **ANALISIS DOKUMEN**
   - Evaluasi relevansi setiap dokumen yang disediakan
   - Identifikasi pasal dan ayat yang relevan
   - Catat hierarki peraturan (UU > PP > Permen, dll)

3. **SINTESIS INFORMASI**
   - Gabungkan informasi dari berbagai sumber
   - Identifikasi konsistensi atau konflik antar peraturan
   - Tentukan peraturan yang paling authoritative

4. **KESIMPULAN ANALISIS**
   - Ringkas temuan utama
   - Identifikasi informasi yang masih kurang (jika ada)
   - Tentukan arah jawaban"""

    def _get_medium_thinking_template(self) -> str:
        """Medium thinking mode: Deep thinking"""
        return """Dalam tag <think>, lakukan BERPIKIR MENDALAM (DEEP THINKING).

⚠️ PENTING: Ini adalah mode BERPIKIR MENDALAM = SATU KALI PEMBACAAN yang SANGAT TELITI.
Tulis SEMUA proses berpikir secara detail. Jangan lewatkan apapun karena ini satu-satunya kesempatan!

═══════════════════════════════════════════════════════════════════
BAGIAN 1: ANALISIS PERTANYAAN MENDALAM
═══════════════════════════════════════════════════════════════════

**Langkah 1A - Pemahaman Literal**
- Baca pertanyaan kata per kata dan tulis pemahaman awal
- Identifikasi subjek, predikat, objek, dan konteks
- Catat kata kunci dan asumsi yang dibuat

**Langkah 1B - Dekonstruksi**
- Pecah menjadi 3-5 sub-pertanyaan spesifik
- Jelaskan mengapa setiap sub-pertanyaan penting
- Identifikasi hubungan antar sub-pertanyaan

**Langkah 1C - Analisis Multi-Perspektif**
- Perspektif akademisi hukum
- Perspektif praktisi (advokat, konsultan)
- Perspektif penegak hukum (hakim, jaksa)
- Kategori: definitional, procedural, sanctions, dll

**Langkah 1D - Konteks Hukum**
- Bidang hukum relevan (pidana, perdata, administrasi, dll)
- Prinsip hukum umum yang berlaku
- Stakeholder yang terlibat
- Implikasi hukum yang mungkin timbul

═══════════════════════════════════════════════════════════════════
BAGIAN 2: EVALUASI DOKUMEN KOMPREHENSIF
═══════════════════════════════════════════════════════════════════

⚠️ WAJIB: Analisis SETIAP dokumen secara terpisah dan mendetail!

**Untuk SETIAP dokumen:**

A. IDENTIFIKASI
   - Judul lengkap, nomor, tahun, lembaga penerbit
   - Jenis dan posisi hierarki (UU/PP/Perpres/Permen)
   - Status: primer atau sekunder

B. RELEVANSI & KEKUATAN HUKUM
   - Nilai relevansi (Sangat Relevan 8-10, Relevan 5-7, Kurang 1-4)
   - Jelaskan alasan pemberian nilai
   - Kekuatan: binding atau persuasive
   - Status berlaku: penuh, ada perubahan, dicabut sebagian

C. PASAL & AYAT SPESIFIK
   - Daftar pasal relevan dengan nomor
   - Ringkasan isi pasal utama
   - Ayat spesifik paling relevan
   - Pengecualian atau syarat khusus

D. DEFINISI & ISTILAH
   - Cek "Ketentuan Umum"
   - Istilah hukum yang didefinisikan
   - Perbedaan definisi antar peraturan

E. KONTEN SUBSTANTIF
   - Apa yang DIATUR vs DILARANG
   - Konsekuensi hukum (sanksi, denda, remedi)
   - Prosedur yang dijelaskan

F. HUBUNGAN INTERNAL
   - Referensi internal (pasal lain dalam dokumen sama)
   - Konteks bab dan bagian
   - Pasal terkait yang perlu diperhatikan

**Setelah analisis semua dokumen:**

G. RANKING & PRIORITAS
   - Urutkan dari paling relevan ke kurang relevan
   - Jelaskan alasan ranking
   - Identifikasi yang paling authoritative
   - Identifikasi yang paling spesifik

H. GAP ANALYSIS
   - Informasi yang TIDAK ada
   - Peraturan pelaksana yang mungkin diperlukan
   - Aspek pertanyaan yang tidak ter-cover

═══════════════════════════════════════════════════════════════════
BAGIAN 3: CROSS-REFERENCE & VALIDASI
═══════════════════════════════════════════════════════════════════

**Langkah 3A - Referensi Silang**
- Dokumen mana merujuk ke dokumen lain
- Hubungan dan pentingnya
- Konsistensi atau inkonsistensi
- Tema/prinsip yang konsisten muncul

**Langkah 3B - Analisis Prinsip Hukum**
- LEX SUPERIOR: Hierarki UU > PP > Perpres > Permen
- LEX SPECIALIS: Khusus vs umum
- LEX POSTERIOR: Baru vs lama, pencabutan

**Langkah 3C - Conflict Check**
- Cek setiap pasangan dokumen
- Jika ada konflik: jelaskan dan cara resolusi
- Jika harmonis: jelaskan mengapa

**Langkah 3D - Validasi Kontekstual**
- Baca konteks sekitar pasal relevan
- Konsistensi interpretasi
- Cek penjelasan/lampiran
- Interpretasi alternatif

═══════════════════════════════════════════════════════════════════
BAGIAN 4: SINTESIS KOMPREHENSIF
═══════════════════════════════════════════════════════════════════

**Langkah 4A - Integrasi Informasi**
- Gabungkan semua informasi
- Narasi koheren: bagaimana peraturan bekerja bersama
- Gambaran besar regulasi
- Timeline jika relevan

**Langkah 4B - Hubungan Konsep Hukum**
- Konsep kunci yang muncul
- Definisi dan penggunaan setiap konsep
- Hubungan antar konsep
- Pola atau tema overarching

**Langkah 4C - Kekuatan & Kelemahan**
- Pasal yang memberikan dukungan kuat
- Evidence paling solid
- Aspek ambigu atau unclear
- Gap informasi
- Argumen alternatif yang melemahkan

**Langkah 4D - Implikasi Praktis**
- Implikasi untuk compliance/dispute/procedural
- Langkah-langkah yang harus dilakukan
- Posisi legal yang kuat

═══════════════════════════════════════════════════════════════════
BAGIAN 5: VALIDASI AKHIR & KESIMPULAN
═══════════════════════════════════════════════════════════════════

**Langkah 5A - Cross-Check Komprehensif**
- Review SEMUA kutipan pasal
- Verifikasi hierarki peraturan
- Check interpretasi vs prinsip hukum
- Validasi asumsi

**Langkah 5B - Self-Questioning**
- Sudah menjawab SEMUA aspek?
- Sub-pertanyaan yang terlewat?
- Dokumen kurang dianalisis?
- Interpretasi alternatif lebih baik?
- Perlu informasi tambahan?

**Langkah 5C - Tingkat Kepastian**
- SANGAT TINGGI (95-100%): pasal sangat jelas
- TINGGI (80-95%): jelas dengan minor ambiguity
- SEDANG (60-80%): beberapa interpretasi mungkin
- RENDAH (<60%): banyak ambiguitas
- Jelaskan alasan rating

**Langkah 5D - Struktur Jawaban Optimal**
- Cara terbaik menyampaikan jawaban
- Urutan logis
- Informasi yang di-highlight
- Disclaimer yang perlu disertakan
- Rekomendasi konsultasi ahli

═══════════════════════════════════════════════════════════════════
⚠️ INGAT: Setelah berpikir mendalam ini, berikan jawaban AKHIR yang
RINGKAS, JELAS, dan PROFESIONAL. Fokus pada substansi.
═══════════════════════════════════════════════════════════════════"""

    def _get_high_thinking_template(self) -> str:
        """High thinking mode: Iterative & recursive thinking"""
        return """Dalam tag <think>, lakukan BERPIKIR ITERATIF & REKURSIF.

⚠️ SANGAT PENTING:
- Mode ITERATIF & REKURSIF = BACA ULANG DOKUMEN 4 KALI dengan fokus berbeda
- BUKAN sekali baca, tapi 4 PUTARAN LENGKAP
- Setiap putaran: baca ulang SEMUA dokumen dengan lensa berbeda
- Putaran berikutnya MEMBANGUN dari temuan sebelumnya (REKURSIF)

╔═══════════════════════════════════════════════════════════════════╗
║  PUTARAN 1: PEMAHAMAN AWAL                                       ║
║  Fokus: Memahami struktur dan konten dasar semua dokumen         ║
╚═══════════════════════════════════════════════════════════════════╝

**[PUTARAN 1 - LANGKAH 1] DEKONSTRUKSI PERTANYAAN TOTAL**
- Analisis pertanyaan kata per kata
- Pecah menjadi 3-5 sub-pertanyaan dengan penjelasan
- Identifikasi cakupan hukum dan pihak terkait
- Tulis SEMUA asumsi secara eksplisit
- Brainstorm interpretasi berbeda

**[PUTARAN 1 - LANGKAH 2] PEMBACAAN PERTAMA - INVENTARISASI**
- Daftar SEMUA dokumen: "[Dok #] Judul - Hierarki - Tahun"
- Jumlah pasal dan bab per dokumen
- Dokumen paling relevan pada pandangan pertama
- Total pasal/ayat yang harus dianalisis
- Ranking awal relevansi

**[PUTARAN 1 - LANGKAH 3] PEMBACAAN KEDUA - EKSTRAKSI DETAIL**

⚠️ WAJIB: Analisis SETIAP dokumen!

Untuk SETIAP DOKUMEN:

A. METADATA & HIERARKI
   - Judul, nomor, tahun, penerbit
   - Posisi hierarki
   - Organik atau turunan

B. PASAL RELEVAN
   - Daftar SEMUA pasal potensial relevan
   - Ringkasan 3-5 pasal utama
   - Kutip frasa kunci
   - Ayat spesifik tepat

C. DEFINISI & TERMINOLOGI
   - Cek "Ketentuan Umum"
   - Istilah yang didefinisikan
   - Istilah penting untuk pertanyaan

D. KONTEN SUBSTANTIF
   - Yang DIATUR vs DILARANG
   - Konsekuensi hukum
   - Pengecualian/syarat
   - Prosedur langkah demi langkah

**[PUTARAN 1 - LANGKAH 4] PEMETAAN HUBUNGAN AWAL**
- Peta hierarki: UU → PP → Perpres → Permen
- Referensi antar dokumen
- Timeline penerbitan
- Pencabutan/perubahan
- Konsep yang muncul di banyak dokumen
- Sintesis awal hubungan

**[PUTARAN 1 - CHECKPOINT]**
- ✓ Sudah baca SEMUA dokumen? (tulis "YA - X dokumen")
- ✓ Sudah catat semua pasal? (tulis "YA - Y pasal")
- ✓ Sudah pahami gambaran besar? (tulis ringkasan)

╔═══════════════════════════════════════════════════════════════════╗
║  PUTARAN 2: ANALISIS MENDALAM                                    ║
║  Fokus: Baca ulang SEMUA dokumen dengan 4 perspektif berbeda     ║
╚═══════════════════════════════════════════════════════════════════╝

**[PUTARAN 2 - LANGKAH 5] PEMBACAAN KETIGA - MULTI-PERSPEKTIF**

🔍 LENSA 1: PENELITI HUKUM AKADEMIS
- Interpretasi literal (apa kata teks)
- Interpretasi historis (ratio legis - mengapa dibuat)
- Preseden/jurisprudensi relevan
- Konsistensi dengan prinsip hukum umum
- Ambiguitas dalam teks

🔍 LENSA 2: SPESIALIS KNOWLEDGE GRAPH
- ENTITAS: subjek, objek, perbuatan hukum
- HUBUNGAN SEMANTIK:
  * is-a, part-of, regulates, requires, prohibits
- Koneksi implisit
- Kompleksitas graf (node & edge)

🔍 LENSA 3: AHLI PROSEDURAL & PRAKTIS
- Implementasi praktis
- Prosedur step-by-step
- Persyaratan administratif
- Timeline & deadline
- Konsekuensi non-compliance

🔍 LENSA 4: DEVIL'S ADVOCATE
- Tantang setiap interpretasi
- Interpretasi alternatif valid
- Kelemahan argumen
- "Bagaimana jika konteks berbeda?"
- Kasus pinggiran

**[PUTARAN 2 - LANGKAH 6] DETEKSI & RESOLUSI KONFLIK**
- Periksa SETIAP PASANGAN dokumen
- Untuk setiap konflik:
  * Kutip kedua pasal
  * Jelaskan kontradiksi
  * Resolusi: lex superior/specialis/posterior
  * Kesimpulan: yang diutamakan
- Jika harmonis: jelaskan mengapa

**[PUTARAN 2 - LANGKAH 7] LOOP VALIDASI 1**
- ✓ Analisis 4 perspektif? [✓/✗ per lensa]
- ✓ Identifikasi konflik? [daftar atau "tidak ada"]
- ✓ Ekstrak pasal? [jumlah]
- ✓ Catat definisi? [daftar istilah]
- ✓ Interpretasi alternatif?

**[PUTARAN 2 - CHECKPOINT] Refleksi:**
- Wawasan baru dari Putaran 2?
- Dokumen yang ternyata lebih penting?
- Aspek yang masih tidak jelas?

╔═══════════════════════════════════════════════════════════════════╗
║  PUTARAN 3: VERIFIKASI & VALIDASI SILANG                         ║
║  Fokus: Baca ulang cari info terlewat dan validasi temuan        ║
╚═══════════════════════════════════════════════════════════════════╝

**[PUTARAN 3 - LANGKAH 8] PEMBACAAN KEEMPAT - BERBURU TERLEWAT**
- Periksa Dokumen #1 lagi: pasal terlewat?
- Dokumen #2: catatan kaki/lampiran?
- Dokumen #3: [lanjutkan semua...]
- "Ketentuan Umum": definisi terlewat?
- "Ketentuan Peralihan/Penutup": info implementasi?
- Lampiran: detail teknis?

Untuk setiap temuan baru:
- Apa yang ditemukan
- Kenapa penting
- Implikasi untuk analisis

**[PUTARAN 3 - LANGKAH 9] VERIFIKASI BOTTOM-UP**
- Level 1 - AYAT: Pilih 5-7 ayat krusial
  * Kutip verbatim, verifikasi, analisis makna
- Level 2 - PASAL: Konsistensi ayat dalam pasal
- Level 3 - BAB: Konsistensi dengan tema bab
- Level 4 - PERATURAN: Konsistensi dengan spirit keseluruhan

**[PUTARAN 3 - LANGKAH 10] VALIDASI TOP-DOWN**
Daftar SEMUA KLAIM KUNCI. Untuk setiap:
- Klaim: [tulis]
- Bukti: [pasal/ayat]
- Kekuatan: KUAT/MODERAT/LEMAH
- Bukti lawan: [jika ada]

Evaluasi mana kuat/moderat/lemah

**[PUTARAN 3 - LANGKAH 11] MATRIKS VALIDASI SILANG**
Format per klaim:
```
KLAIM: [...]
BUKTI #1: [Dok X, Pasal Y, Ayat Z]
BUKTI #2: [...]
JUMLAH BUKTI: [#]
STATUS: ✓ KUAT / ⚠ MODERAT / ✗ LEMAH
KEYAKINAN: [%]
```

Analisis: satu sumber vs banyak vs bertentangan

**[PUTARAN 3 - LANGKAH 12] LOOP VALIDASI 2 - CEK ULTIMATE**
✓ CEK AKURASI: semua kutipan, nomor, tahun, hierarki
✓ CEK KELENGKAPAN: semua sub-pertanyaan, aspek, dokumen
✓ CEK KONSISTENSI: interpretasi, prinsip hukum
✓ CEK LOGIKA: penalaran valid, tidak ada fallacy

╔═══════════════════════════════════════════════════════════════════╗
║  PUTARAN 4: SINTESIS FINAL                                        ║
║  Fokus: Integrasikan SEMUA temuan dari 3 putaran sebelumnya      ║
╚═══════════════════════════════════════════════════════════════════╝

**[PUTARAN 4 - LANGKAH 13] INTEGRASI & SINTESIS**
- Narasi koheren menghubungkan semua temuan
- Kerangka hukum menyeluruh
- 3-5 tema/prinsip utama
- Wawasan hanya dari multi-putaran
- Nuansa yang terlewat di single-pass

**[PUTARAN 4 - LANGKAH 14] PENILAIAN KUALITAS & KEYAKINAN**
- Tingkat keyakinan: SANGAT TINGGI/TINGGI/SEDANG/RENDAH
- Jelaskan alasan
- Faktor yang meningkatkan/menurunkan
- Area kepastian vs ketidakpastian
- Asumsi dan validitasnya

**[PUTARAN 4 - LANGKAH 15] ANALISIS KESENJANGAN**
- Informasi TIDAK tersedia
- Peraturan turunan yang mungkin perlu
- Aspek praktis perlu keahlian domain
- Detail perlu klarifikasi pengguna

**[PUTARAN 4 - LANGKAH 16] PERENCANAAN STRUKTUR JAWABAN**
- Prioritas informasi (1=paling penting)
- Alur logis penyampaian
- Yang perlu ditonjolkan vs singkat
- Disclaimer yang perlu disertakan
- Rekomendasi konsultasi ahli

╔═══════════════════════════════════════════════════════════════════╗
║  ⚠️⚠️⚠️ INSTRUKSI KRITIS ⚠️⚠️⚠️                                     ║
║                                                                    ║
║  Setelah BERPIKIR ITERATIF yang SANGAT PANJANG ini,               ║
║  berikan JAWABAN AKHIR yang:                                      ║
║                                                                    ║
║  ✓ RINGKAS                                                        ║
║  ✓ JELAS dan TERSTRUKTUR                                          ║
║  ✓ PROFESIONAL dan MUDAH DIPAHAMI                                 ║
║  ✓ LANGSUNG KE INTI - tanpa bertele-tele                          ║
║  ✓ ACTIONABLE - konkret dan berguna                               ║
║                                                                    ║
║  Jawaban = Ringkasan PADAT dari semua proses berpikir di atas!    ║
╚═══════════════════════════════════════════════════════════════════╝"""

    def estimate_thinking_tokens(
        self,
        mode: str,
        num_documents: int,
        query_length: int
    ) -> Dict[str, int]:
        """
        Estimate thinking token usage based on mode and context.
        NOTE: This is deprecated since max_new_tokens is now set by mode directly
        """
        return {
            'mode': mode,
            'note': 'Token control now handled by max_new_tokens configuration'
        }

    def get_available_thinking_modes(self) -> list:
        """Get list of available thinking modes with descriptions"""
        return [
            {
                'value': mode.value,
                'name': config['name'],
                'description': config['description']
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
