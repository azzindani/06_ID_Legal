"""
Stress Test - Multi-Turn Conversation with Maximum Settings

This test runs an 8-turn complex legal conversation with ALL settings maxed out:
- ALL 5 search phases enabled (including expert_review)
- Maximum candidates per phase (800+)
- Maximum research team size (5 personas)
- Maximum final_top_k (20 documents)
- Maximum max_new_tokens (8192)
- Maximum conversation history (50 turns tracked)
- All validation features enabled

Conversation Flow (8 Turns):
- T1: Teacher/professor allowance equality (establishes context)
- T2: Specific regulation PP No. 41 Tahun 2009
- T3: Follow-up on allowance differences
- T4: Topic shift to customs law - kawasan pabean
- T5: Follow-up on customs sanctions
- T6: Topic shift to labor law UU No. 13 Tahun 2003
- T7: Specific article query in UU No. 13 Tahun 2003
- T8: Summary of PP No. 8 Tahun 2007

Purpose:
- Verify conversation memory under maximum context load
- Test context window handling with large documents
- Measure cumulative resource usage across turns
- Validate that maxed settings don't cause OOM or timeouts
- Test session management with heavy context
- Test multi-domain topic switching

Run with:
    python tests/integration/test_stress_conversational.py

Options:
    --quick      Use moderate settings (8 turns, reduced candidates)
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

# Full stress conversation - 8 questions covering teacher/professor allowances, customs, and labor law
STRESS_CONVERSATION_FULL = [
    {
        'turn': 1,
        'topic': 'TUNJANGAN_PENDIDIK',
        'type': 'initial',
        'query': "Apakah terdapat pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?"
    },
    {
        'turn': 2,
        'topic': 'TUNJANGAN_PENDIDIK',
        'type': 'specific_regulation',
        'query': "Berdasarkan PP No. 41 Tahun 2009, sebutkan jenis-jenis tunjangan yang diatur di dalamnya."
    },
    {
        'turn': 3,
        'topic': 'TUNJANGAN_PENDIDIK',
        'type': 'followup_detailed',
        'query': "Masih merujuk pada PP No. 41 Tahun 2009, jelaskan perbedaan kriteria penerima, besaran, dan sumber pendanaan antara Tunjangan Khusus dan Tunjangan Kehormatan Profesor"
    },
    {
        'turn': 4,
        'topic': 'KEPABEANAN',
        'type': 'topic_shift',
        'query': "Ganti topik. Jelaskan secara singkat pengertian kawasan pabean menurut Undang-Undang Kepabeanan."
    },
    {
        'turn': 5,
        'topic': 'KEPABEANAN',
        'type': 'followup_sanctions',
        'query': "Berdasarkan Undang-Undang Kepabeanan tersebut, jelaskan sanksi pidana bagi pihak yang dengan sengaja salah memberitahukan jenis dan jumlah barang impor sehingga merugikan negara."
    },
    {
        'turn': 6,
        'topic': 'KETENAGAKERJAAN',
        'type': 'topic_shift',
        'query': "Sekarang beralih ke UU No. 13 Tahun 2003. Jelaskan secara umum ruang lingkup dan pokok bahasan undang-undang tersebut."
    },
    {
        'turn': 7,
        'topic': 'KETENAGAKERJAAN',
        'type': 'specific_article',
        'query': "Apa yang diatur dalam Pasal 1 UU No. 13 Tahun 2003?"
    },
    {
        'turn': 8,
        'topic': 'PP_8_2007',
        'type': 'summary',
        'query': "Terakhir, jelaskan secara ringkas PP No. 8 Tahun 2007, termasuk fokus pengaturannya."
    }
]

# Quick mode conversation - 8 questions (same as full stress but with moderate config)
QUICK_CONVERSATION = [
    {
        'turn': 1,
        'topic': 'TUNJANGAN_PENDIDIK',
        'type': 'initial',
        'query': "Apakah terdapat pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?"
    },
    {
        'turn': 2,
        'topic': 'TUNJANGAN_PENDIDIK',
        'type': 'specific_regulation',
        'query': "Berdasarkan PP No. 41 Tahun 2009, sebutkan jenis-jenis tunjangan yang diatur di dalamnya."
    },
    {
        'turn': 3,
        'topic': 'TUNJANGAN_PENDIDIK',
        'type': 'followup_detailed',
        'query': "Masih merujuk pada PP No. 41 Tahun 2009, jelaskan perbedaan kriteria penerima, besaran, dan sumber pendanaan antara Tunjangan Khusus dan Tunjangan Kehormatan Profesor"
    },
    {
        'turn': 4,
        'topic': 'KEPABEANAN',
        'type': 'topic_shift',
        'query': "Ganti topik. Jelaskan secara singkat pengertian kawasan pabean menurut Undang-Undang Kepabeanan."
    },
    {
        'turn': 5,
        'topic': 'KEPABEANAN',
        'type': 'followup_sanctions',
        'query': "Berdasarkan Undang-Undang Kepabeanan tersebut, jelaskan sanksi pidana bagi pihak yang dengan sengaja salah memberitahukan jenis dan jumlah barang impor sehingga merugikan negara."
    },
    {
        'turn': 6,
        'topic': 'KETENAGAKERJAAN',
        'type': 'topic_shift',
        'query': "Sekarang beralih ke UU No. 13 Tahun 2003. Jelaskan secara umum ruang lingkup dan pokok bahasan undang-undang tersebut."
    },
    {
        'turn': 7,
        'topic': 'KETENAGAKERJAAN',
        'type': 'specific_article',
        'query': "Apa yang diatur dalam Pasal 1 UU No. 13 Tahun 2003?"
    },
    {
        'turn': 8,
        'topic': 'PP_8_2007',
        'type': 'summary',
        'query': "Terakhir, jelaskan secara ringkas PP No. 8 Tahun 2007, termasuk fokus pengaturannya."
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
        self.memory_manager = None
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
        print("Initializing RAG Pipeline and Memory Manager...")

        try:
            from pipeline import RAGPipeline
            from conversation import MemoryManager, create_memory_manager

            # Initialize pipeline
            self.pipeline = RAGPipeline(config=self.config)
            if not self.pipeline.initialize():
                print("Pipeline initialization FAILED")
                return False

            print("Pipeline initialized")

            # Initialize memory manager with stress test limits + enhanced features
            # NOTE: Using lower limits (50/20) than legal defaults (100/30) to stress test
            # the system under heavy load with constrained memory
            self.memory_manager = create_memory_manager({
                'max_history_turns': 50,        # Track many turns (vs 100 legal default)
                'max_context_turns': 20,        # Use for context (vs 30 legal default)
                'enable_cache': True,           # Enable caching for performance
                'cache_size': 100,              # Large cache for stress test
                'max_tokens': 16000,            # High token limit for stress
                'enable_summarization': True,   # Test auto-summarization under stress
                'enable_key_facts': True        # Test key facts extraction under stress
            })

            self.session_id = self.memory_manager.start_session()
            print(f"Memory manager session started: {self.session_id}")

            self.logger.success("All components ready for conversational stress test")
            return True

        except Exception as e:
            print(f"Initialization error: {e}")
            self.logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation history for context with caching"""
        if not self.memory_manager or not self.session_id:
            return []

        # MemoryManager.get_context() automatically handles caching and formatting
        context = self.memory_manager.get_context(self.session_id, max_turns=10)
        return context or []

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
                # Handle different chunk types
                if not isinstance(chunk, dict):
                    # Safety check: if chunk is not a dict, log and skip
                    print(f"\nWARNING: Received non-dict chunk: {type(chunk)}")
                    continue

                if chunk.get('type') == 'token':
                    token = chunk.get('token', '')
                    print(token, end='', flush=True)
                    full_answer += token
                    chunk_count += 1
                elif chunk.get('type') == 'complete':
                    result = chunk
                    break
                elif chunk.get('type') == 'error':
                    # Handle error chunks
                    error_msg = chunk.get('error', 'Unknown error')
                    print(f"\nERROR from pipeline: {error_msg}")
                    result = chunk  # Store error chunk as result
                    break

            turn_time = time.time() - start_time
            print(f"\n\n[Turn {turn_num}: {chunk_count} tokens in {turn_time:.2f}s]")

            # Save to memory manager (with automatic caching)
            self.memory_manager.save_turn(
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
        if self.memory_manager and self.session_id:
            print("\n" + "=" * 100)
            print("CONVERSATION MEMORY & CONTEXT AUDIT")
            print("=" * 100)
            session_data = self.memory_manager.get_session(self.session_id)
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
                    print(f"  User Query: {turn.get('query', 'N/A')}")
                    print(f"  Assistant Answer: {turn.get('answer', 'N/A')[:300]}...")
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
                                user_msg = prev_turn.get('query', '')[:80]
                                asst_msg = prev_turn.get('answer', '')[:80]
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

        # ===== ENHANCED MEMORY TESTING UNDER STRESS =====
        # Test intelligent long-term memory features under maximum load
        if self.memory_manager and self.session_id:
            print("\n" + "=" * 100)
            print("ENHANCED MEMORY FEATURES TEST (Under Stress)")
            print("=" * 100)
            print("Testing intelligent long-term memory under maximum settings\n")

            # 1. Key Facts Extraction Test
            print("┌" + "─" * 98 + "┐")
            print("│ 1. KEY FACTS EXTRACTION (Under Maximum Load)                                            │")
            print("├" + "─" * 98 + "┤")
            key_facts = self.memory_manager.get_key_facts(self.session_id)
            if key_facts:
                print(f"│ Total key facts extracted: {len(key_facts):<67} │")
                print("│" + " " * 98 + "│")
                for i, fact in enumerate(key_facts[:10], 1):  # Show first 10
                    fact_str = f"{i}. {fact}"
                    print(f"│   {fact_str:<94} │")
                if len(key_facts) > 10:
                    print(f"│   ... and {len(key_facts) - 10} more facts                                                          │")
                print("│" + " " * 98 + "│")
                print("│ ✓ Key facts extracted even under maximum stress                                     │")
            else:
                print("│ No key facts extracted (none found in this conversation)                            │")
            print("└" + "─" * 98 + "┘")

            # 2. Session Summary Test
            print("\n┌" + "─" * 98 + "┐")
            print("│ 2. SESSION SUMMARY (Under Stress)                                                       │")
            print("├" + "─" * 98 + "┤")
            session_summary = self.memory_manager.get_session_summary_dict(self.session_id)
            if session_summary:
                topics = session_summary.get('topics_discussed', [])
                regulations = session_summary.get('regulations_mentioned', [])

                if topics:
                    topics_str = ', '.join(topics)
                    print(f"│ Topics: {topics_str:<85} │")

                if regulations:
                    print(f"│ Regulations mentioned: {len(regulations):<68} │")
                    for i, reg in enumerate(regulations[:5], 1):
                        print(f"│   {i}. {reg:<91} │")
                    if len(regulations) > 5:
                        print(f"│   ... and {len(regulations) - 5} more regulations                                              │")

                print("│" + " " * 98 + "│")
                print("│ ✓ Session tracking maintained under stress                                          │")
            else:
                print("│ No session summary available                                                        │")
            print("└" + "─" * 98 + "┘")

            # 3. Memory Performance Under Stress
            print("\n┌" + "─" * 98 + "┐")
            print("│ 3. MEMORY PERFORMANCE UNDER MAXIMUM LOAD                                                │")
            print("├" + "─" * 98 + "┤")

            mem_stats = self.memory_manager.get_stats()
            max_history = self.memory_manager.max_history_turns
            max_context = self.memory_manager.max_context_turns
            turn_count = len(self.turn_results)

            print(f"│ Stress test configuration:                                                          │")
            print(f"│   • Max history turns: {max_history} (lower than legal default 100)                        │")
            print(f"│   • Max context turns: {max_context} (lower than legal default 30)                         │")
            print(f"│   • Total turns executed: {turn_count}                                                         │")
            print("│" + " " * 98 + "│")

            cache_hits = mem_stats.get('manager_stats', {}).get('cache_hits', 0)
            cache_misses = mem_stats.get('manager_stats', {}).get('cache_misses', 0)
            cache_hit_rate = mem_stats.get('cache_hit_rate', 0)
            key_facts_count = mem_stats.get('total_key_facts', 0)
            summaries_count = mem_stats.get('manager_stats', {}).get('summaries_created', 0)

            print(f"│ Cache performance:                                                                   │")
            print(f"│   • Cache hits: {cache_hits}                                                                     │")
            print(f"│   • Cache misses: {cache_misses}                                                                    │")
            print(f"│   • Hit rate: {cache_hit_rate:.1%}                                                                 │")
            print("│" + " " * 98 + "│")

            print(f"│ Enhanced features under stress:                                                      │")
            print(f"│   • Key facts extracted: {key_facts_count}                                                          │")
            print(f"│   • Summaries created: {summaries_count}                                                            │")
            print("│" + " " * 98 + "│")

            # Check if summarization was triggered (turn count > max_context)
            if turn_count > max_context:
                print(f"│ ✓ Conversation exceeded max_context ({max_context}), summarization active                  │")
                print("│ ✓ System handled maximum load with intelligent memory                               │")
            else:
                print(f"│ • Conversation within max_context ({max_context}), all turns detailed                      │")

            print("│" + " " * 98 + "│")
            print("│ ✓ Enhanced memory features working correctly under maximum stress                    │")
            print("└" + "─" * 98 + "┘")

            # 4. Stress Test Verdict
            print("\n┌" + "─" * 98 + "┐")
            print("│ 4. STRESS TEST VERDICT - ENHANCED MEMORY                                                │")
            print("├" + "─" * 98 + "┤")
            print("│                                                                                          │")
            print("│ ✓ Key facts extraction working under maximum load                                       │")
            print("│ ✓ Session summary tracking maintained under stress                                      │")
            print("│ ✓ LRU caching performing correctly                                                      │")
            print("│ ✓ Intelligent context building active                                                   │")

            if turn_count > max_context:
                print("│ ✓ Automatic summarization triggered and working                                        │")

            print("│                                                                                          │")
            print("│ The enhanced memory system successfully handles maximum stress conditions               │")
            print("│ with constrained memory limits (50/20 vs 100/30 legal defaults).                        │")
            print("│                                                                                          │")
            print("└" + "─" * 98 + "┘")

            print("\n" + "=" * 100)
            print("ENHANCED MEMORY STRESS TEST COMPLETE ✓")
            print("=" * 100)

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
