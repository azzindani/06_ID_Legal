"""
Stress Test - Multi-Turn Conversation with Maximum Settings

This test runs a 7-turn complex legal conversation with ALL settings maxed out:
- ALL 5 search phases enabled (including expert_review)
- Maximum candidates per phase (800+)
- Maximum research team size (5 personas)
- Maximum final_top_k (20 documents)
- Maximum max_new_tokens (8192)
- Maximum conversation history (50 turns tracked)
- All validation features enabled

Conversation Flow (7 Turns):
- T1: Complex tax law question (establishes heavy context)
- T2: Follow-up with specific regulation reference
- T3: Cross-domain shift to labor law
- T4: Back-reference to T1 context
- T5: Complex procedural question spanning multiple domains
- T6: Clarification request building on all previous context
- T7: Summary request requiring full conversation memory

Purpose:
- Verify conversation memory under maximum context load
- Test context window handling with large documents
- Measure cumulative resource usage across turns
- Validate that maxed settings don't cause OOM or timeouts
- Test session management with heavy context

Run with:
    python tests/integration/test_stress_conversational.py

Options:
    --quick      Use moderate settings (5 turns, reduced candidates)
    --verbose    Show detailed output during processing
    --memory     Enable detailed memory profiling per turn
    --export     Export results to JSON
"""

import sys
import os
import time
import tracemalloc
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging
from utils.research_transparency import format_detailed_research_process, format_researcher_summary
from utils.conversation_audit import format_conversation_context, print_conversation_memory_summary


# Maximum stress test configuration for conversational
STRESS_CONFIG_CONV = {
    'final_top_k': 15,                     # High but manageable
    'max_rounds': 5,
    'research_team_size': 5,               # All 5 personas
    'max_new_tokens': 6144,                # High token limit
    'temperature': 0.7,
    'top_p': 1.0,
    'top_k': 80,
    'min_p': 0.05,
    'enable_cross_validation': True,
    'enable_devil_advocate': True,
    'consensus_threshold': 0.5,
    'parallel_research': True,
}

# Maximum search phases for conversation
STRESS_SEARCH_PHASES_CONV = {
    'initial_scan': {
        'candidates': 600,
        'semantic_threshold': 0.18,
        'keyword_threshold': 0.05,
        'description': 'Broad conversational scan',
        'time_limit': 90,
        'focus_areas': ['regulation_type', 'enacting_body'],
        'enabled': True
    },
    'focused_review': {
        'candidates': 300,
        'semantic_threshold': 0.28,
        'keyword_threshold': 0.10,
        'description': 'Focused conversational review',
        'time_limit': 90,
        'focus_areas': ['content', 'chapter', 'article'],
        'enabled': True
    },
    'deep_analysis': {
        'candidates': 150,
        'semantic_threshold': 0.38,
        'keyword_threshold': 0.14,
        'description': 'Deep conversational analysis',
        'time_limit': 90,
        'focus_areas': ['kg_entities', 'cross_references'],
        'enabled': True
    },
    'verification': {
        'candidates': 80,
        'semantic_threshold': 0.48,
        'keyword_threshold': 0.18,
        'description': 'Conversational verification',
        'time_limit': 60,
        'focus_areas': ['authority_score', 'temporal_score'],
        'enabled': True
    },
    'expert_review': {
        'candidates': 60,
        'semantic_threshold': 0.42,
        'keyword_threshold': 0.15,
        'description': 'Expert conversational review',
        'time_limit': 60,
        'focus_areas': ['legal_richness', 'completeness_score'],
        'enabled': True
    }
}

# Full stress conversation - 7 complex turns
STRESS_CONVERSATION_FULL = [
    {
        'turn': 1,
        'topic': 'PERPAJAKAN',
        'type': 'complex_initial',
        'query': """
        Saya ingin memahami secara komprehensif tentang prosedur keberatan pajak.
        Jelaskan secara detail tentang:
        1. Dasar hukum pengajuan keberatan menurut UU KUP
        2. Syarat-syarat formal dan materil yang harus dipenuhi
        3. Jangka waktu pengajuan dan konsekuensi keterlambatan
        4. Hak-hak wajib pajak selama proses keberatan
        """
    },
    {
        'turn': 2,
        'topic': 'PERPAJAKAN',
        'type': 'followup_specific',
        'query': """
        Terkait penjelasan sebelumnya, bagaimana jika keberatan ditolak?
        Jelaskan tentang mekanisme banding ke Pengadilan Pajak berdasarkan
        UU Pengadilan Pajak dan hubungannya dengan proses keberatan yang sudah dijelaskan.
        Sertakan juga tentang biaya yang diperlukan.
        """
    },
    {
        'turn': 3,
        'topic': 'KETENAGAKERJAAN',
        'type': 'topic_shift',
        'query': """
        Sekarang saya ingin bertanya tentang domain yang berbeda.
        Jelaskan tentang hak-hak pekerja yang di-PHK menurut UU Ketenagakerjaan
        dan UU Cipta Kerja. Bagaimana prosedur penyelesaian perselisihan PHK
        dan kompensasi apa saja yang berhak diterima pekerja?
        """
    },
    {
        'turn': 4,
        'topic': 'PERPAJAKAN',
        'type': 'back_reference',
        'query': """
        Kembali ke pembahasan pajak di awal, jika wajib pajak yang sedang
        mengajukan keberatan ternyata juga terlibat dalam perselisihan PHK
        sebagai pengusaha, apakah ada keterkaitan antara kewajiban pajak
        dengan pembayaran pesangon? Bagaimana perlakuan pajak atas pesangon?
        """
    },
    {
        'turn': 5,
        'topic': 'MULTI_DOMAIN',
        'type': 'complex_procedural',
        'query': """
        Jelaskan prosedur lengkap yang harus dilakukan pengusaha yang:
        1. Menghadapi sengketa keberatan pajak (seperti yang sudah dibahas)
        2. Sedang dalam proses PHK massal (seperti yang sudah dijelaskan)
        3. Ingin melakukan restrukturisasi perusahaan

        Bagaimana urutan prioritas penyelesaian dan lembaga mana saja yang
        harus dihubungi? Sertakan dasar hukumnya.
        """
    },
    {
        'turn': 6,
        'topic': 'CLARIFICATION',
        'type': 'clarification',
        'query': """
        Dari semua penjelasan sebelumnya, saya masih bingung tentang:
        1. Apakah pengusaha bisa menunda pembayaran pajak sambil menunggu keberatan?
        2. Bagaimana jika dana untuk pesangon digunakan untuk bayar pajak dulu?
        3. Apa sanksi jika tidak membayar pesangon tepat waktu karena masalah pajak?

        Tolong jelaskan dengan mengacu pada konteks yang sudah kita bahas.
        """
    },
    {
        'turn': 7,
        'topic': 'SUMMARY',
        'type': 'summary_request',
        'query': """
        Berdasarkan SELURUH pembahasan kita dari awal:
        1. Buatkan ringkasan poin-poin kunci tentang prosedur keberatan pajak
        2. Rangkum hak-hak pekerja yang di-PHK yang sudah dibahas
        3. Jelaskan keterkaitan antara kedua aspek tersebut
        4. Berikan rekomendasi langkah-langkah yang harus diambil pengusaha

        Pastikan mengacu pada semua peraturan yang sudah disebutkan sebelumnya.
        """
    }
]

# Quick mode conversation - 5 simpler turns
QUICK_CONVERSATION = [
    {
        'turn': 1,
        'topic': 'PERPAJAKAN',
        'type': 'initial',
        'query': "Jelaskan tentang prosedur keberatan pajak menurut UU KUP."
    },
    {
        'turn': 2,
        'topic': 'PERPAJAKAN',
        'type': 'followup',
        'query': "Bagaimana jika keberatan ditolak? Apa langkah selanjutnya?"
    },
    {
        'turn': 3,
        'topic': 'KETENAGAKERJAAN',
        'type': 'topic_shift',
        'query': "Sekarang jelaskan tentang hak pekerja yang di-PHK."
    },
    {
        'turn': 4,
        'topic': 'MULTI',
        'type': 'cross_reference',
        'query': "Bagaimana hubungan antara pajak dan pembayaran pesangon?"
    },
    {
        'turn': 5,
        'topic': 'SUMMARY',
        'type': 'summary',
        'query': "Rangkum pembahasan kita tentang pajak dan ketenagakerjaan."
    }
]

# Moderate config for quick mode
MODERATE_CONFIG_CONV = {
    'final_top_k': 8,
    'max_rounds': 3,
    'research_team_size': 3,
    'max_new_tokens': 2048,
    'temperature': 0.7,
    'enable_cross_validation': True,
    'enable_devil_advocate': False,
    'consensus_threshold': 0.6,
}


class ConversationalStressTester:
    """Multi-turn conversation stress test with maximum settings"""

    def __init__(self, quick_mode: bool = False, verbose: bool = False, memory_profile: bool = False):
        initialize_logging()
        self.logger = get_logger("ConvStressTest")
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.memory_profile = memory_profile

        # Select config based on mode
        self.config = MODERATE_CONFIG_CONV.copy() if quick_mode else STRESS_CONFIG_CONV.copy()
        if not quick_mode:
            self.config['search_phases'] = STRESS_SEARCH_PHASES_CONV

        # Select conversation
        self.conversation = QUICK_CONVERSATION if quick_mode else STRESS_CONVERSATION_FULL

        self.pipeline = None
        self.conversation_manager = None
        self.session_id = None

        # Results
        self.turn_results: List[Dict[str, Any]] = []
        self.total_results: Dict[str, Any] = {}

    def print_header(self):
        """Print test header"""
        mode = "QUICK MODE" if self.quick_mode else "MAXIMUM STRESS MODE"
        turns = len(self.conversation)

        print("\n" + "=" * 100)
        print(f"STRESS TEST - MULTI-TURN CONVERSATION - {mode}")
        print("=" * 100)
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Planned Turns: {turns}")
        print()

        print("Configuration:")
        print("-" * 50)
        for key, value in self.config.items():
            if key != 'search_phases':
                print(f"  {key}: {value}")

        if 'search_phases' in self.config:
            print("\nSearch Phases:")
            for phase, settings in self.config['search_phases'].items():
                status = "ENABLED" if settings.get('enabled', False) else "disabled"
                print(f"  {phase}: {settings.get('candidates', 0)} candidates ({status})")
        print()

    def initialize(self) -> bool:
        """Initialize pipeline and conversation manager"""
        self.logger.info("Initializing conversational stress test...")
        print("Initializing RAG Pipeline and Conversation Manager...")

        try:
            from pipeline import RAGPipeline
            from conversation import ConversationManager

            # Initialize pipeline
            self.pipeline = RAGPipeline(config=self.config)
            if not self.pipeline.initialize():
                print("Pipeline initialization FAILED")
                return False

            print("Pipeline initialized")

            # Initialize conversation manager with high limits
            self.conversation_manager = ConversationManager({
                'max_history_turns': 50,       # Track many turns
                'max_context_turns': 20,       # Use many for context
                'compression_threshold': 100   # High threshold to avoid early compression
            })

            self.session_id = self.conversation_manager.start_session()
            print(f"Conversation session started: {self.session_id}")

            self.logger.success("All components ready for conversational stress test")
            return True

        except Exception as e:
            print(f"Initialization error: {e}")
            self.logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation history for context"""
        if not self.conversation_manager or not self.session_id:
            return []

        history = self.conversation_manager.get_history(self.session_id, max_turns=10)
        context = []
        for turn in history:
            context.append({"role": "user", "content": turn['query']})
            # Truncate long answers to fit context
            answer = turn['answer'][:2000] if len(turn['answer']) > 2000 else turn['answer']
            context.append({"role": "assistant", "content": answer})
        return context

    def run_turn(self, turn_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single conversation turn"""
        turn_num = turn_data['turn']
        query = turn_data['query'].strip()
        topic = turn_data.get('topic', 'GENERAL')
        turn_type = turn_data.get('type', 'query')

        print(f"\n{'='*100}")
        print(f"TURN {turn_num}/{len(self.conversation)} - {topic} ({turn_type})")
        print("=" * 100)
        print(f"\nQuery ({len(query)} chars):")
        print("-" * 80)
        print(query[:500] + "..." if len(query) > 500 else query)
        print("-" * 80)

        # Start memory tracking for this turn
        if self.memory_profile:
            tracemalloc.start()

        # Get conversation context
        context = self._get_conversation_context()
        context_size = sum(len(m['content']) for m in context)
        print(f"Context size: {len(context)} messages, {context_size} chars")

        start_time = time.time()

        try:
            # Stream the response using query() with stream=True
            print("\nAnswer:")
            print("-" * 80)

            full_answer = ""
            chunk_count = 0
            result = None

            for chunk in self.pipeline.query(query, conversation_history=context, stream=True):
                if chunk.get('type') == 'token':
                    token = chunk.get('token', '')
                    print(token, end='', flush=True)
                    full_answer += token
                    chunk_count += 1
                elif chunk.get('type') == 'complete':
                    result = chunk
                    break

            turn_time = time.time() - start_time
            print(f"\n\n[Turn {turn_num}: {chunk_count} tokens in {turn_time:.2f}s]")

            # Add to conversation history
            self.conversation_manager.add_turn(
                self.session_id,
                query,
                full_answer,
                metadata={'turn': turn_num, 'topic': topic, 'type': turn_type}
            )

            # Memory stats
            memory_stats = {}
            if self.memory_profile:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_stats = {
                    'current_mb': current / (1024 * 1024),
                    'peak_mb': peak / (1024 * 1024)
                }
                print(f"Memory: Current={memory_stats['current_mb']:.2f}MB, Peak={memory_stats['peak_mb']:.2f}MB")

            turn_result = {
                'turn': turn_num,
                'topic': topic,
                'type': turn_type,
                'success': True,
                'query_length': len(query),
                'answer_length': len(full_answer),
                'chunk_count': chunk_count,
                'time': turn_time,
                'context_messages': len(context),
                'context_chars': context_size,
                'memory': memory_stats
            }

            if result:
                turn_result['sources_count'] = len(result.get('sources', []))
                turn_result['sources'] = result.get('sources', [])
                turn_result['citations'] = result.get('citations', [])
                turn_result['phase_metadata'] = result.get('phase_metadata', {})
                turn_result['research_log'] = result.get('research_log', {})
                turn_result['consensus_data'] = result.get('consensus_data', {})
                turn_result['research_data'] = result.get('research_data', {})

            return turn_result

        except Exception as e:
            turn_time = time.time() - start_time
            print(f"\n\nERROR in turn {turn_num}: {e}")
            self.logger.error(f"Turn {turn_num} failed: {e}")

            if self.memory_profile:
                tracemalloc.stop()

            return {
                'turn': turn_num,
                'topic': topic,
                'type': turn_type,
                'success': False,
                'error': str(e),
                'time': turn_time,
                'context_messages': len(context),
                'context_chars': context_size
            }

    def run_conversation(self) -> Dict[str, Any]:
        """Run the full stress conversation"""
        print("\n" + "=" * 100)
        print("STARTING CONVERSATIONAL STRESS TEST")
        print("=" * 100)

        total_start = time.time()

        # Run each turn
        for turn_data in self.conversation:
            turn_result = self.run_turn(turn_data)
            self.turn_results.append(turn_result)

            # Brief pause between turns (simulate real usage)
            time.sleep(0.5)

        total_time = time.time() - total_start

        # Compile overall results
        successful_turns = sum(1 for t in self.turn_results if t.get('success'))
        total_tokens = sum(t.get('chunk_count', 0) for t in self.turn_results)
        total_answer_chars = sum(t.get('answer_length', 0) for t in self.turn_results)

        self.total_results = {
            'success': successful_turns == len(self.conversation),
            'total_turns': len(self.conversation),
            'successful_turns': successful_turns,
            'failed_turns': len(self.conversation) - successful_turns,
            'total_time': total_time,
            'average_turn_time': total_time / len(self.conversation),
            'total_tokens': total_tokens,
            'total_answer_chars': total_answer_chars,
            'turn_details': self.turn_results,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'mode': 'quick' if self.quick_mode else 'maximum'
        }

        return self.total_results

    def print_results(self):
        """Print detailed results summary"""
        print("\n" + "=" * 100)
        print("CONVERSATIONAL STRESS TEST RESULTS")
        print("=" * 100)

        r = self.total_results
        success_rate = (r['successful_turns'] / r['total_turns']) * 100

        print(f"\nOverall Status: {'SUCCESS' if r['success'] else 'PARTIAL FAILURE'}")
        print(f"Turns: {r['successful_turns']}/{r['total_turns']} ({success_rate:.1f}% success)")
        print(f"Total Time: {r['total_time']:.2f}s")
        print(f"Average Turn Time: {r['average_turn_time']:.2f}s")
        print(f"Total Tokens Generated: {r['total_tokens']}")
        print(f"Total Answer Characters: {r['total_answer_chars']}")

        # Per-turn breakdown
        print("\n" + "-" * 80)
        print("TURN-BY-TURN BREAKDOWN")
        print("-" * 80)
        print(f"{'Turn':<6} {'Topic':<20} {'Status':<10} {'Time':>8} {'Tokens':>8} {'Context':>10}")
        print("-" * 80)

        for t in self.turn_results:
            status = "OK" if t.get('success') else "FAIL"
            time_str = f"{t.get('time', 0):.2f}s"
            tokens = t.get('chunk_count', 0)
            context = f"{t.get('context_messages', 0)} msgs"

            print(f"{t['turn']:<6} {t['topic']:<20} {status:<10} {time_str:>8} {tokens:>8} {context:>10}")

        # Memory summary if profiled
        if self.memory_profile:
            print("\n" + "-" * 50)
            print("MEMORY PROFILE")
            print("-" * 50)
            for t in self.turn_results:
                if t.get('memory'):
                    mem = t['memory']
                    print(f"Turn {t['turn']}: Current={mem['current_mb']:.2f}MB, Peak={mem['peak_mb']:.2f}MB")

        # Configuration summary
        print("\n" + "-" * 50)
        print("CONFIGURATION USED")
        print("-" * 50)
        mode = "Quick" if self.quick_mode else "Maximum"
        print(f"Mode: {mode}")
        print(f"final_top_k: {self.config.get('final_top_k', 'N/A')}")
        print(f"research_team_size: {self.config.get('research_team_size', 'N/A')}")
        print(f"max_new_tokens: {self.config.get('max_new_tokens', 'N/A')}")

        if 'search_phases' in self.config:
            enabled = sum(1 for p in self.config['search_phases'].values() if p.get('enabled'))
            print(f"Search Phases Enabled: {enabled}/5")

        # Print detailed output from last successful turn
        self.print_detailed_turn_output()

        # Print conversation context audit with FULL MEMORY CONTENT
        if self.conversation_manager and self.session_id:
            print("\n" + "=" * 100)
            print("CONVERSATION MEMORY & CONTEXT AUDIT")
            print("=" * 100)
            session_data = self.conversation_manager.get_session(self.session_id)
            if session_data:
                # Show full conversation content (not truncated)
                conversation_audit = format_conversation_context(
                    session_data,
                    show_full_content=True,  # ✅ Show FULL content
                    max_turns=None  # Show all turns
                )
                print(conversation_audit)

                # Also show the conversation history structure
                print("\n" + "=" * 100)
                print("CONVERSATION MEMORY STRUCTURE")
                print("=" * 100)
                print("This is the actual conversation history stored in memory:")
                print("")

                turns = session_data.get('turns', [])
                for idx, turn in enumerate(turns, 1):
                    print(f"Turn {idx}:")
                    print(f"  User Message: {turn.get('user_message', 'N/A')}")
                    print(f"  Assistant Message: {turn.get('assistant_message', 'N/A')[:300]}...")
                    print(f"  Metadata Keys: {list(turn.get('metadata', {}).keys())}")
                    print("")

                # Show what context is passed to pipeline at each turn
                print("\n" + "=" * 100)
                print("CONTEXT PASSED TO PIPELINE (Per Turn)")
                print("=" * 100)
                print("This shows what conversation history was sent to the pipeline at each turn:")
                print("")

                for idx, turn_result in enumerate(self.turn_results, 1):
                    context_msgs = turn_result.get('context_messages', 0)
                    print(f"Turn {idx}: {context_msgs} previous messages in context")

                    # Show the actual context if available
                    if idx > 1:  # Skip first turn (no context)
                        print(f"  Context includes:")
                        for prev_idx in range(1, idx):
                            if prev_idx - 1 < len(turns):
                                prev_turn = turns[prev_idx - 1]
                                user_msg = prev_turn.get('user_message', '')[:80]
                                asst_msg = prev_turn.get('assistant_message', '')[:80]
                                print(f"    - Turn {prev_idx}: User: {user_msg}...")
                                print(f"               Assistant: {asst_msg}...")
                    print("")

        # Print detailed research process from last turn
        if self.turn_results:
            last_turn = self.turn_results[-1]
            if last_turn.get('success'):
                print("\n" + "=" * 100)
                print("DETAILED RESEARCH PROCESS (Last Turn)")
                print("=" * 100)
                detailed_research = format_detailed_research_process(
                    last_turn,
                    top_n_per_researcher=10,
                    show_content=False
                )
                print(detailed_research)

        print("\n" + "=" * 100)

    def _extract_all_documents(self, turn_result: Dict[str, Any], max_docs: int = 50) -> list:
        """Extract all retrieved documents from turn metadata (limited to max_docs)"""
        all_docs = []
        seen_ids = set()

        # Try phase_metadata first
        phase_metadata = turn_result.get('phase_metadata', {})
        for phase_name, phase_data in phase_metadata.items():
            if isinstance(phase_data, dict):
                candidates = phase_data.get('candidates', phase_data.get('results', []))
                for doc in candidates:
                    record = doc.get('record', doc)
                    doc_id = record.get('global_id', str(hash(str(doc))))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc['_phase'] = phase_data.get('phase', phase_name)
                        doc['_researcher'] = phase_data.get('researcher_name', phase_data.get('researcher', ''))
                        all_docs.append(doc)
                        if len(all_docs) >= max_docs:
                            return all_docs

        # Try research_data
        if not all_docs:
            research_data = turn_result.get('research_data', {})
            all_results = research_data.get('all_results', [])
            for doc in all_results:
                record = doc.get('record', doc)
                doc_id = record.get('global_id', str(hash(str(doc))))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
                    if len(all_docs) >= max_docs:
                        return all_docs

        # Try sources as fallback
        if not all_docs:
            sources = turn_result.get('sources', turn_result.get('citations', []))
            for doc in sources:
                doc_id = doc.get('global_id', doc.get('regulation_number', str(hash(str(doc)))))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append({'record': doc, 'scores': {'final': doc.get('score', 0)}})
                    if len(all_docs) >= max_docs:
                        return all_docs

        return all_docs

    def print_detailed_turn_output(self):
        """Print detailed output from the last successful turn"""
        # Find last successful turn with metadata
        last_turn = None
        for t in reversed(self.turn_results):
            if t.get('success') and (t.get('sources') or t.get('phase_metadata')):
                last_turn = t
                break

        if not last_turn:
            print("\nNo detailed metadata available from turns")
            return

        turn_num = last_turn.get('turn', 0)
        print(f"\n" + "=" * 100)
        print(f"DETAILED OUTPUT FROM TURN {turn_num} (Last Successful Turn)")
        print("=" * 100)

        # LEGAL REFERENCES (Top K Documents Used in LLM Prompt)
        self._print_legal_references(last_turn)

        # RESEARCH PROCESS DETAILS
        self._print_research_process(last_turn)

        # ALL Retrieved Documents (Article-Level Details)
        self._print_all_documents(last_turn)

    def _print_legal_references(self, turn_result: Dict[str, Any]):
        """Print LEGAL REFERENCES (Top K Documents Used in LLM Prompt)"""
        print("\n" + "=" * 100)
        print("## LEGAL REFERENCES (Top K Documents Used in LLM Prompt)")
        print("=" * 100)
        print("These are the final selected documents sent to the LLM for answer generation.")
        print()

        sources = turn_result.get('sources', turn_result.get('citations', []))

        if sources:
            print(f"Documents Used in Prompt: {len(sources)}")
            print()

            for idx, source in enumerate(sources, 1):
                reg_type = source.get('regulation_type', 'N/A')
                reg_num = source.get('regulation_number', 'N/A')
                year = source.get('year', 'N/A')
                about = source.get('about', 'N/A')
                enacting_body = source.get('enacting_body', 'N/A')
                score = source.get('score', 0)
                content = source.get('content', '')

                print(f"### {idx}. {reg_type} No. {reg_num}/{year}")
                print(f"   About: {about}")
                print(f"   Enacting Body: {enacting_body}")
                print(f"   Final Score: {score:.4f}")

                if content:
                    content_preview = content[:800].replace('\n', ' ')
                    print(f"   Content Preview: {content_preview}...")
                print()
        else:
            print("No documents in prompt (check retrieval)")

    def _print_research_process(self, turn_result: Dict[str, Any]):
        """Print RESEARCH PROCESS DETAILS"""
        print("\n" + "=" * 100)
        print("## RESEARCH PROCESS DETAILS (ALL Retrieved Documents)")
        print("=" * 100)
        print("All documents retrieved during research process - for audit and verification.")
        print()

        research_log = turn_result.get('research_log', {})
        phase_metadata = turn_result.get('phase_metadata', {})

        # Team members
        team_members = research_log.get('team_members', [])
        if team_members:
            print("### Research Team")
            print(f"Team Size: {len(team_members)}")
            for member in team_members:
                if isinstance(member, dict):
                    print(f"   - {member.get('name', member.get('persona', 'Unknown'))}")
                else:
                    print(f"   - {member}")
        elif phase_metadata:
            unique_researchers = set()
            for phase_key, phase_data in phase_metadata.items():
                if isinstance(phase_data, dict):
                    researcher = phase_data.get('researcher_name', phase_data.get('researcher', ''))
                    if researcher:
                        unique_researchers.add(researcher)
            if unique_researchers:
                print("### Research Team")
                print(f"Team Size: {len(unique_researchers)}")
                for member in unique_researchers:
                    print(f"   - {member}")

        # Summary Statistics
        total_docs = research_log.get('total_documents_retrieved', 0)
        all_documents = self._extract_all_documents(turn_result, max_docs=50)
        if not total_docs:
            total_docs = len(all_documents)

        print(f"\n### Summary Statistics")
        print(f"Total Documents Retrieved: {total_docs}")
        print(f"Total Phases: {len(phase_metadata)}")

        # Phase breakdown
        if phase_metadata:
            print(f"\n### Phase Breakdown")
            print("-" * 80)
            for phase_key, phase_data in phase_metadata.items():
                if isinstance(phase_data, dict):
                    phase_name = phase_data.get('phase', phase_key)
                    researcher = phase_data.get('researcher_name', phase_data.get('researcher', 'Unknown'))
                    candidates = phase_data.get('candidates', phase_data.get('results', []))
                    confidence = phase_data.get('confidence', 1.0)

                    print(f"\n   Phase: {phase_name}")
                    print(f"   Researcher: {researcher}")
                    print(f"   Documents: {len(candidates)}")
                    print(f"   Confidence: {confidence:.2%}")

    def _print_all_documents(self, turn_result: Dict[str, Any]):
        """Print ALL Retrieved Documents (Article-Level Details) - Top 50"""
        print("\n" + "=" * 100)
        print("### ALL Retrieved Documents (Article-Level Details) - TOP 50")
        print("=" * 100)

        all_documents = self._extract_all_documents(turn_result, max_docs=50)

        if not all_documents:
            print("No documents retrieved")
            return

        print(f"Showing {len(all_documents)} documents")
        print()

        for i, doc in enumerate(all_documents, 1):
            record = doc.get('record', doc)
            scores = doc.get('scores', {})

            # Basic info
            reg_type = record.get('regulation_type', 'N/A')
            reg_num = record.get('regulation_number', 'N/A')
            year = record.get('year', 'N/A')
            about = record.get('about', 'N/A')
            enacting_body = record.get('enacting_body', 'N/A')
            global_id = record.get('global_id', 'N/A')

            # Scores
            final_score = scores.get('final', doc.get('final_score', doc.get('composite_score', record.get('score', 0))))
            semantic = scores.get('semantic', doc.get('semantic_score', 0))
            keyword = scores.get('keyword', doc.get('keyword_score', 0))
            kg = scores.get('kg', doc.get('kg_score', 0))
            authority = scores.get('authority', doc.get('authority_score', 0))
            temporal = scores.get('temporal', doc.get('temporal_score', 0))
            completeness = scores.get('completeness', doc.get('completeness_score', 0))

            # Article-level location
            chapter = record.get('chapter', record.get('bab', ''))
            article = record.get('article', record.get('pasal', ''))
            section = record.get('section', record.get('bagian', ''))
            paragraph = record.get('paragraph', record.get('ayat', ''))

            # KG metadata
            kg_domain = record.get('kg_primary_domain', record.get('primary_domain', ''))
            kg_hierarchy = record.get('kg_hierarchy_level', record.get('hierarchy_level', 0))
            kg_cross_refs = record.get('kg_cross_ref_count', record.get('cross_ref_count', 0))

            # Phase info
            phase = doc.get('_phase', '')
            researcher = doc.get('_researcher', '')

            print(f"[{i}] {reg_type} No. {reg_num}/{year}")
            print(f"    Global ID: {global_id}")
            print(f"    About: {about}")
            print(f"    Enacting Body: {enacting_body}")

            # Article-level location
            location_parts = []
            if chapter:
                location_parts.append(f"Bab {chapter}")
            if section:
                location_parts.append(f"Bagian {section}")
            if article:
                location_parts.append(f"Pasal {article}")
            if paragraph:
                location_parts.append(f"Ayat {paragraph}")

            if location_parts:
                print(f"    Location: {' > '.join(location_parts)}")
            else:
                print(f"    Location: (Full Document)")

            # All scores
            print(f"    Scores:")
            print(f"       Final: {final_score:.4f} | Semantic: {semantic:.4f} | Keyword: {keyword:.4f}")
            print(f"       KG: {kg:.4f} | Authority: {authority:.4f} | Temporal: {temporal:.4f} | Completeness: {completeness:.4f}")

            # KG metadata
            if kg_domain or kg_hierarchy:
                print(f"    Knowledge Graph: Domain={kg_domain or 'N/A'} | Hierarchy={kg_hierarchy} | CrossRefs={kg_cross_refs}")

            # Research info
            if phase or researcher:
                print(f"    Discovery: Phase={phase} | Researcher={researcher}")

            # Content (truncated - 300 chars)
            content = record.get('content', '')
            if content:
                content_truncated = content[:300].replace('\n', ' ').strip()
                print(f"    Content (truncated): {content_truncated}...")

            print("-" * 100)

    def export_results(self, filepath: str):
        """Export results to JSON"""
        import json
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.total_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Results exported to: {filepath}")
        except Exception as e:
            print(f"Export error: {e}")

    def shutdown(self):
        """Clean up resources"""
        if self.pipeline:
            try:
                self.pipeline.shutdown()
                print("Pipeline shutdown complete")
            except Exception as e:
                print(f"Shutdown warning: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Stress Test - Multi-Turn Conversation with Maximum Settings")
    parser.add_argument('--quick', action='store_true', help='Use moderate settings with 5 turns')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--memory', action='store_true', help='Enable memory profiling')
    parser.add_argument('--export', action='store_true', help='Export results to JSON')
    parser.add_argument('--output', type=str, help='Output file path for export')
    args = parser.parse_args()

    # Create tester
    tester = ConversationalStressTester(
        quick_mode=args.quick,
        verbose=args.verbose,
        memory_profile=args.memory
    )

    # Print header
    tester.print_header()

    # Initialize
    if not tester.initialize():
        print("\nFailed to initialize. Exiting.")
        sys.exit(1)

    try:
        # Run conversation
        results = tester.run_conversation()

        # Print results
        tester.print_results()

        # Export if requested
        if args.export:
            output_path = args.output or f"stress_conv_results_{int(time.time())}.json"
            tester.export_results(output_path)

        # Return appropriate exit code
        sys.exit(0 if results.get('success') else 1)

    finally:
        tester.shutdown()


if __name__ == "__main__":
    main()
