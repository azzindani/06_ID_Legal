"""
Conversational RAG Test - Multi-Turn Conversation with Memory and Context Management

This test verifies the system's conversational capabilities using EXISTING MODULES:
- QueryDetector (core/search/query_detection.py) - Query analysis, follow-up detection
- KnowledgeGraphCore (core/knowledge_graph/kg_core.py) - Entity/regulation extraction
- ConversationManager (conversation/manager.py) - Session and history management
- RAGPipeline (pipeline/rag_pipeline.py) - Complete RAG processing

Features Tested:
1. CONVERSATION MEMORY - ConversationManager tracks context across turns
2. CONTEXT MANAGEMENT - QueryDetector detects follow-ups and topic changes
3. SPECIFIC REGULATION RECOGNITION - KnowledgeGraphCore extracts UU No. 13 Tahun 2003
4. ENTITY EXTRACTION - KnowledgeGraphCore identifies legal entities
5. TOPIC CHANGE DETECTION - QueryDetector analyzes query patterns
6. FOLLOW-UP QUESTION HANDLING - QueryDetector.is_followup detection

Conversation Flow (5 Questions):
────────────────────────────────────────────────────────────────────────────────
TOPIC 1: KETENAGAKERJAAN (Labor Law) - 3 Questions
  Q1: General question about worker rights (establishes context)
  Q2: Specific regulation reference - UU No. 13 Tahun 2003 on severance pay
  Q3: Follow-up question building on Q2 (tests memory)

TOPIC 2: LINGKUNGAN HIDUP (Environmental Law) - 1 Question
  Q4: Topic shift to environmental permits (tests context switching)

TOPIC 3: PERPAJAKAN (Tax Law) - 1 Question
  Q5: Different domain - tax objection mechanism (tests multi-domain)
────────────────────────────────────────────────────────────────────────────────

Run with:
    python tests/integration/test_conversational.py

Options:
    --export          Export results to JSON
    --output PATH     Custom output file path
    --verbose         Show detailed metadata for each turn
"""

import sys
import os
import time
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging
from pipeline import RAGPipeline
from conversation import ConversationManager, get_context_cache
from core.search.query_detection import QueryDetector
from core.knowledge_graph.kg_core import KnowledgeGraphCore


class ConversationalTester:
    """
    Tests multi-turn conversational capabilities using existing system modules.

    Uses:
    - QueryDetector for query analysis and follow-up detection
    - KnowledgeGraphCore for entity and regulation extraction
    - ConversationManager for session tracking
    - RAGPipeline for end-to-end processing
    """

    def __init__(self, verbose: bool = False):
        initialize_logging()
        self.logger = get_logger("ConversationalTest")
        self.verbose = verbose

        # Core components - use EXISTING modules
        self.pipeline: Optional[RAGPipeline] = None
        self.conversation_manager: Optional[ConversationManager] = None
        self.query_detector: Optional[QueryDetector] = None
        self.kg_core: Optional[KnowledgeGraphCore] = None
        self.session_id: Optional[str] = None

        # Results tracking
        self.turn_results: List[Dict[str, Any]] = []
        self.conversation_log: List[Dict[str, Any]] = []

        # Metrics for analysis
        self.metrics = {
            'topic_continuity_detected': 0,
            'topic_shifts_detected': 0,
            'specific_regulations_found': 0,
            'follow_up_context_used': 0,
            'entities_extracted': [],
            'regulation_references': [],
            'total_documents_retrieved': 0,
            'conversation_coherence_score': 0.0
        }

    def initialize(self) -> bool:
        """Initialize RAG pipeline and all required modules"""
        self.logger.info("=" * 100)
        self.logger.info("INITIALIZING CONVERSATIONAL RAG SYSTEM")
        self.logger.info("=" * 100)

        try:
            # Initialize RAG Pipeline
            self.pipeline = RAGPipeline()
            if not self.pipeline.initialize():
                self.logger.error("Pipeline initialization failed")
                return False

            # Initialize QueryDetector - EXISTING MODULE
            self.query_detector = QueryDetector()
            self.logger.info("QueryDetector initialized (core/search/query_detection.py)")

            # Initialize KnowledgeGraphCore - EXISTING MODULE
            self.kg_core = KnowledgeGraphCore()
            self.logger.info("KnowledgeGraphCore initialized (core/knowledge_graph/kg_core.py)")

            # Initialize Conversation Manager - EXISTING MODULE
            self.conversation_manager = ConversationManager({
                'max_history_turns': 50,
                'max_context_turns': 10
            })
            self.logger.info("ConversationManager initialized (conversation/manager.py)")

            # Start session
            self.session_id = self.conversation_manager.start_session()
            self.logger.info(f"Conversation session started: {self.session_id}")

            self.logger.success("All modules initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation history for context-aware queries"""
        if not self.conversation_manager or not self.session_id:
            return []

        history = self.conversation_manager.get_history(self.session_id, max_turns=5)
        context = []
        for turn in history:
            context.append({"role": "user", "content": turn['query']})
            context.append({"role": "assistant", "content": turn['answer'][:500]})
        return context

    def _analyze_query_with_modules(self, query: str, previous_queries: List[str]) -> Dict[str, Any]:
        """
        Analyze query using EXISTING modules:
        - QueryDetector.analyze_query() for query type, follow-up detection
        - KnowledgeGraphCore.extract_regulation_references_with_confidence() for regulations
        - KnowledgeGraphCore.extract_entities() for entity extraction
        """
        # Use QueryDetector for comprehensive analysis
        query_analysis = self.query_detector.analyze_query(
            query,
            conversation_history=[{'query': q} for q in previous_queries]
        )

        # Use KnowledgeGraphCore for regulation extraction
        regulation_refs = self.kg_core.extract_regulation_references_with_confidence(query)

        # Use KnowledgeGraphCore for entity extraction
        entities = self.kg_core.extract_entities(query)

        # Determine topic based on query_type from QueryDetector
        current_topic = query_analysis.get('query_type', 'general')

        # Detect topic shift by comparing with previous topics
        previous_topics = []
        for prev_q in previous_queries:
            prev_analysis = self.query_detector.analyze_query(prev_q)
            previous_topics.append(prev_analysis.get('query_type', 'general'))

        is_topic_shift = (
            len(previous_topics) > 0 and
            current_topic != previous_topics[-1] if previous_topics else False
        )

        return {
            'query_analysis': query_analysis,
            'query_type': query_analysis.get('query_type', 'general'),
            'is_followup': query_analysis.get('is_followup', False),
            'is_clarification': query_analysis.get('is_clarification', False),
            'has_regulation_ref': query_analysis.get('has_regulation_ref', False),
            'regulation_references': regulation_refs,
            'entities': entities,
            'current_topic': current_topic,
            'previous_topics': previous_topics,
            'is_topic_shift': is_topic_shift,
            'complexity_score': query_analysis.get('complexity_score', 0),
            'team_composition': query_analysis.get('team_composition', [])
        }

    def format_turn_output(
        self,
        turn_num: int,
        query: str,
        answer: str,
        metadata: Dict[str, Any],
        analysis: Dict[str, Any],
        streaming_stats: Dict[str, Any]
    ) -> str:
        """Format comprehensive output for a conversation turn"""
        lines = []

        lines.append("\n" + "█" * 100)
        lines.append(f"  TURN {turn_num}: {analysis.get('turn_type', 'QUERY')}")
        lines.append("█" * 100)

        # Question
        lines.append(f"\n┌{'─' * 98}┐")
        lines.append(f"│ QUESTION                                                                                     │")
        lines.append(f"├{'─' * 98}┤")
        q_display = query[:92] + '...' if len(query) > 92 else query
        lines.append(f"│ {q_display:<96} │")
        lines.append(f"└{'─' * 98}┘")

        # Analysis Section (using EXISTING module results)
        lines.append(f"\n┌{'─' * 98}┐")
        lines.append(f"│ ANALYSIS (from QueryDetector & KnowledgeGraphCore)                                           │")
        lines.append(f"├{'─' * 98}┤")

        query_type = analysis.get('query_type', 'N/A')
        is_followup = '✓' if analysis.get('is_followup') else '—'
        is_shift = '⚠️' if analysis.get('is_topic_shift') else '—'
        lines.append(f"│ Query Type: {query_type:<15} | Follow-up: {is_followup:<5} | Topic Shift: {is_shift:<5}                       │")

        # Regulation references from KnowledgeGraphCore
        reg_refs = analysis.get('regulation_references', [])
        if reg_refs:
            for ref in reg_refs[:2]:
                ref_str = f"{ref.get('type', 'N/A')} No. {ref.get('number', '?')}/{ref.get('year', '?')} (conf: {ref.get('confidence', 0):.1f})"
                lines.append(f"│ Regulation Found: {ref_str:<76} │")

        # Entities from KnowledgeGraphCore
        entities = analysis.get('entities', {})
        if entities:
            entity_summary = []
            for etype, evals in entities.items():
                if evals:
                    entity_summary.append(f"{etype}: {len(evals)}")
            if entity_summary:
                ent_str = ", ".join(entity_summary[:4])
                lines.append(f"│ Entities: {ent_str:<84} │")

        # Team composition from QueryDetector
        team = analysis.get('team_composition', [])
        if team:
            team_str = ", ".join(team[:3])
            lines.append(f"│ Research Team: {team_str:<79} │")

        lines.append(f"└{'─' * 98}┘")

        # Thinking Process (if available and verbose)
        thinking = metadata.get('thinking', '')
        if thinking and self.verbose:
            lines.append(f"\n┌{'─' * 98}┐")
            lines.append(f"│ THINKING PROCESS                                                                             │")
            lines.append(f"├{'─' * 98}┤")
            for i in range(0, min(len(thinking), 400), 94):
                chunk = thinking[i:i+94]
                lines.append(f"│ {chunk:<96} │")
            if len(thinking) > 400:
                lines.append(f"│ ... [truncated - {len(thinking)} chars total]                                                         │")
            lines.append(f"└{'─' * 98}┘")

        # Answer
        lines.append(f"\n┌{'─' * 98}┐")
        lines.append(f"│ ANSWER                                                                                       │")
        lines.append(f"├{'─' * 98}┤")
        answer_lines = answer.split('\n')
        for ans_line in answer_lines[:12]:
            for i in range(0, len(ans_line), 94):
                chunk = ans_line[i:i+94]
                lines.append(f"│ {chunk:<96} │")
        if len(answer_lines) > 12:
            lines.append(f"│ ... [truncated - {len(answer_lines)} lines total]                                                        │")
        lines.append(f"└{'─' * 98}┘")

        # Sources/Citations
        sources = metadata.get('sources', metadata.get('citations', []))
        if sources:
            lines.append(f"\n┌{'─' * 98}┐")
            lines.append(f"│ LEGAL SOURCES ({len(sources)} documents)                                                               │")
            lines.append(f"├{'─' * 98}┤")
            for idx, source in enumerate(sources[:5], 1):
                reg_type = source.get('regulation_type', 'N/A')
                reg_num = source.get('regulation_number', 'N/A')
                year = source.get('year', 'N/A')
                score = source.get('score', 0)
                src_str = f"{idx}. {reg_type} No. {reg_num}/{year} (Score: {score:.3f})"
                lines.append(f"│ {src_str:<96} │")
            if len(sources) > 5:
                lines.append(f"│ ... and {len(sources) - 5} more documents                                                             │")
            lines.append(f"└{'─' * 98}┘")

        # Context Memory (from ConversationManager)
        if turn_num > 1:
            lines.append(f"\n┌{'─' * 98}┐")
            lines.append(f"│ CONVERSATION MEMORY (from ConversationManager)                                                │")
            lines.append(f"├{'─' * 98}┤")
            context_used = len(analysis.get('previous_topics', []))
            lines.append(f"│ Previous turns tracked: {context_used:<71} │")
            if analysis.get('previous_topics'):
                prev_topics = ', '.join(analysis['previous_topics'][-3:])
                lines.append(f"│ Recent topics: {prev_topics:<79} │")
            lines.append(f"└{'─' * 98}┘")

        # Timing
        lines.append(f"\n┌{'─' * 98}┐")
        lines.append(f"│ PERFORMANCE                                                                                  │")
        lines.append(f"├{'─' * 98}┤")
        duration = streaming_stats.get('duration', 0)
        chunks = streaming_stats.get('chunk_count', 0)
        retrieval_time = metadata.get('retrieval_time', 0)
        perf_str = f"Streaming: {chunks} chunks in {duration:.2f}s | Retrieval: {retrieval_time:.2f}s"
        lines.append(f"│ {perf_str:<96} │")
        lines.append(f"└{'─' * 98}┘")

        return "\n".join(lines)

    def run_conversation_turn(
        self,
        query: str,
        turn_num: int,
        turn_type: str = "QUERY"
    ) -> Dict[str, Any]:
        """Run a single conversation turn using existing modules"""
        self.logger.info(f"\n{'#' * 100}")
        self.logger.info(f"TURN {turn_num}: {turn_type}")
        self.logger.info(f"{'#' * 100}")

        result = {
            'turn_num': turn_num,
            'turn_type': turn_type,
            'query': query,
            'success': False,
            'answer': '',
            'metadata': {},
            'analysis': {},
            'streaming_stats': {}
        }

        try:
            # Get previous queries from conversation log
            previous_queries = [t['query'] for t in self.conversation_log]

            # Analyze query using EXISTING MODULES
            analysis = self._analyze_query_with_modules(query, previous_queries)
            analysis['turn_type'] = turn_type

            # Update metrics based on module analysis
            if not analysis['is_topic_shift'] and turn_num > 1:
                self.metrics['topic_continuity_detected'] += 1
            if analysis['is_topic_shift']:
                self.metrics['topic_shifts_detected'] += 1
            if analysis['regulation_references']:
                self.metrics['specific_regulations_found'] += len(analysis['regulation_references'])
                for ref in analysis['regulation_references']:
                    ref_str = f"{ref.get('type', '')} {ref.get('number', '')}/{ref.get('year', '')}"
                    if ref_str not in self.metrics['regulation_references']:
                        self.metrics['regulation_references'].append(ref_str)
            if analysis['is_followup']:
                self.metrics['follow_up_context_used'] += 1

            print(f"\n{'=' * 100}")
            print(f"TURN {turn_num}: {query[:80]}{'...' if len(query) > 80 else ''}")
            print("=" * 100)
            print("\n[STREAMING ANSWER]")
            print("-" * 80)

            full_answer = ""
            chunk_count = 0
            start_time = time.time()
            final_metadata = {}

            # Stream the response using RAGPipeline
            for chunk in self.pipeline.query(query, stream=True):
                chunk_type = chunk.get('type', '')

                if chunk_type == 'token':
                    token = chunk.get('token', '')
                    print(token, end='', flush=True)
                    full_answer += token
                    chunk_count += 1

                elif chunk_type == 'complete':
                    full_answer = chunk.get('answer', full_answer)
                    final_metadata = chunk.get('metadata', {})
                    final_metadata['thinking'] = chunk.get('thinking', '')
                    final_metadata['sources'] = chunk.get('sources', [])
                    final_metadata['citations'] = chunk.get('citations', [])
                    final_metadata['query_type'] = chunk.get('query_type', 'general')
                    final_metadata['phase_metadata'] = chunk.get('phase_metadata', {})

                elif chunk_type == 'error':
                    error_msg = chunk.get('error', 'Unknown error')
                    self.logger.error(f"Streaming error: {error_msg}")
                    result['error'] = error_msg
                    return result

            duration = time.time() - start_time
            print(f"\n[Streamed {chunk_count} chunks in {duration:.2f}s]")
            print("-" * 80)

            streaming_stats = {
                'chunk_count': chunk_count,
                'duration': duration,
                'chars_per_second': len(full_answer) / duration if duration > 0 else 0
            }

            # Add turn to ConversationManager (EXISTING MODULE)
            if self.conversation_manager and self.session_id:
                self.conversation_manager.add_turn(
                    session_id=self.session_id,
                    query=query,
                    answer=full_answer,
                    metadata={
                        **final_metadata,
                        'analysis': analysis
                    }
                )

            # Format and display output
            formatted_output = self.format_turn_output(
                turn_num=turn_num,
                query=query,
                answer=full_answer,
                metadata=final_metadata,
                analysis=analysis,
                streaming_stats=streaming_stats
            )
            print(formatted_output)

            # Update conversation log
            self.conversation_log.append({
                'turn_num': turn_num,
                'query': query,
                'answer': full_answer,
                'topic': analysis['current_topic'],
                'timestamp': datetime.now().isoformat()
            })

            # Track documents and entities
            sources = final_metadata.get('sources', [])
            self.metrics['total_documents_retrieved'] += len(sources)

            for source in sources:
                entity = f"{source.get('regulation_type', '')} {source.get('regulation_number', '')}/{source.get('year', '')}"
                if entity not in self.metrics['entities_extracted']:
                    self.metrics['entities_extracted'].append(entity)

            result['success'] = True
            result['answer'] = full_answer
            result['metadata'] = final_metadata
            result['analysis'] = analysis
            result['streaming_stats'] = streaming_stats
            result['formatted_output'] = formatted_output

            self.logger.success(f"Turn {turn_num} completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Turn {turn_num} failed: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
            return result

    def run_full_conversation(self) -> bool:
        """Run the complete 5-question conversational test"""

        # Define the conversation flow
        conversation_script = [
            # TOPIC 1: Ketenagakerjaan (Labor Law) - 3 Questions
            {
                'query': "Apa saja hak-hak pekerja menurut peraturan ketenagakerjaan di Indonesia?",
                'type': "INITIAL QUERY",
                'description': "General question establishing labor law context"
            },
            {
                'query': "Jelaskan pasal-pasal dalam UU Nomor 13 Tahun 2003 yang mengatur tentang pesangon dan bagaimana cara menghitungnya",
                'type': "SPECIFIC REGULATION",
                'description': "Direct reference to UU No. 13 Tahun 2003 - tests KnowledgeGraphCore extraction"
            },
            {
                'query': "Bagaimana jika perusahaan tidak membayar pesangon tersebut? Apa upaya hukum yang dapat dilakukan pekerja?",
                'type': "FOLLOW-UP",
                'description': "Follow-up using context - tests QueryDetector.is_followup"
            },
            # TOPIC 2: Lingkungan Hidup (Environmental Law) - 1 Question
            {
                'query': "Sekarang saya ingin bertanya tentang izin lingkungan. Apa persyaratan untuk mendapatkan izin lingkungan berdasarkan peraturan yang berlaku?",
                'type': "TOPIC SHIFT",
                'description': "Topic change to environmental law - tests QueryDetector topic analysis"
            },
            # TOPIC 3: Perpajakan (Tax Law) - 1 Question
            {
                'query': "Pertanyaan terakhir mengenai perpajakan. Bagaimana mekanisme pengajuan keberatan pajak dan apa saja syaratnya?",
                'type': "NEW DOMAIN",
                'description': "Different domain (tax law) - tests multi-domain capability"
            }
        ]

        self.logger.info("\n" + "═" * 100)
        self.logger.info("  CONVERSATIONAL RAG TEST - Using Existing System Modules")
        self.logger.info("═" * 100)
        self.logger.info("\n  Modules Being Tested:")
        self.logger.info("    • QueryDetector (core/search/query_detection.py)")
        self.logger.info("    • KnowledgeGraphCore (core/knowledge_graph/kg_core.py)")
        self.logger.info("    • ConversationManager (conversation/manager.py)")
        self.logger.info("    • RAGPipeline (pipeline/rag_pipeline.py)")
        self.logger.info(f"\n  Session ID: {self.session_id}")
        self.logger.info(f"  Questions: {len(conversation_script)}")
        self.logger.info(f"  Topics: Ketenagakerjaan (3) → Lingkungan (1) → Perpajakan (1)")
        self.logger.info("═" * 100)

        # Initialize
        if not self.initialize():
            return False

        try:
            successful = 0

            for i, script_item in enumerate(conversation_script, 1):
                result = self.run_conversation_turn(
                    query=script_item['query'],
                    turn_num=i,
                    turn_type=script_item['type']
                )
                self.turn_results.append(result)

                if result['success']:
                    successful += 1

                # Pause between turns
                if i < len(conversation_script):
                    self.logger.info("\n⏳ Pause before next turn...")
                    time.sleep(3)

            # Calculate coherence score
            self.metrics['conversation_coherence_score'] = self._calculate_coherence_score()

            # Display Final Summary
            self._display_final_summary(successful, len(conversation_script))

            return successful == len(conversation_script)

        finally:
            if self.pipeline:
                self.logger.info("Shutting down pipeline...")
                self.pipeline.shutdown()

    def _calculate_coherence_score(self) -> float:
        """Calculate overall conversation coherence score"""
        if not self.turn_results:
            return 0.0

        scores = []

        # Score based on topic continuity (expected 2 continuations in Q2, Q3)
        continuity_score = min(1.0, self.metrics['topic_continuity_detected'] / 2.0)
        scores.append(continuity_score)

        # Score based on topic shifts handled (expected 2 shifts in Q4, Q5)
        shift_score = min(1.0, self.metrics['topic_shifts_detected'] / 2.0)
        scores.append(shift_score)

        # Score based on specific regulation recognition (Q2 has UU 13/2003)
        reg_score = 1.0 if self.metrics['specific_regulations_found'] >= 1 else 0.0
        scores.append(reg_score)

        # Score based on follow-up handling (Q3 is follow-up)
        followup_score = 1.0 if self.metrics['follow_up_context_used'] >= 1 else 0.0
        scores.append(followup_score)

        # Score based on successful turns
        success_rate = sum(1 for r in self.turn_results if r['success']) / len(self.turn_results)
        scores.append(success_rate)

        return sum(scores) / len(scores) if scores else 0.0

    def _display_final_summary(self, successful: int, total: int):
        """Display comprehensive final summary"""
        print("\n" + "═" * 100)
        print("  CONVERSATIONAL TEST SUMMARY")
        print("═" * 100)

        print(f"\n  RESULTS: {successful}/{total} turns successful")
        print(f"  Session ID: {self.session_id}")
        print(f"  Coherence Score: {self.metrics['conversation_coherence_score']:.1%}")

        print("\n  " + "─" * 96)
        print("  TURN-BY-TURN RESULTS")
        print("  " + "─" * 96)

        for result in self.turn_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            turn_type = result['turn_type']
            query_preview = result['query'][:50] + "..." if len(result['query']) > 50 else result['query']
            topic = result['analysis'].get('current_topic', 'N/A')

            print(f"  Turn {result['turn_num']}: [{status}] [{turn_type}]")
            print(f"         Query: {query_preview}")
            print(f"         Topic (from QueryDetector): {topic}")

            if result['success']:
                stats = result['streaming_stats']
                sources = result['metadata'].get('sources', [])
                print(f"         Chunks: {stats.get('chunk_count', 0)} | Duration: {stats.get('duration', 0):.2f}s | Sources: {len(sources)}")
            print()

        print("  " + "─" * 96)
        print("  MODULE VERIFICATION METRICS")
        print("  " + "─" * 96)

        print(f"\n  QueryDetector Results:")
        print(f"     • Follow-ups detected: {self.metrics['follow_up_context_used']}")
        print(f"     • Topic shifts detected: {self.metrics['topic_shifts_detected']}")
        print(f"     • Topic continuity: {self.metrics['topic_continuity_detected']}")

        print(f"\n  KnowledgeGraphCore Results:")
        print(f"     • Regulation references extracted: {self.metrics['specific_regulations_found']}")
        if self.metrics['regulation_references']:
            print(f"     • References: {', '.join(self.metrics['regulation_references'][:5])}")

        print(f"\n  ConversationManager Results:")
        print(f"     • Session tracked: {self.session_id}")
        print(f"     • Turns recorded: {len(self.conversation_log)}")

        print(f"\n  RAGPipeline Results:")
        print(f"     • Total documents retrieved: {self.metrics['total_documents_retrieved']}")
        print(f"     • Unique regulations cited: {len(self.metrics['entities_extracted'])}")

        print("\n" + "═" * 100)

        # Final verdict
        score = self.metrics['conversation_coherence_score']
        if score >= 0.8:
            print("  VERDICT: EXCELLENT - All modules working correctly together")
        elif score >= 0.6:
            print("  VERDICT: GOOD - Most module integrations working")
        elif score >= 0.4:
            print("  VERDICT: FAIR - Some module issues detected")
        else:
            print("  VERDICT: NEEDS WORK - Module integration problems found")

        print("═" * 100 + "\n")

    def export_results(self, output_path: Optional[str] = None) -> str:
        """Export all results to JSON"""
        if not output_path:
            output_path = f"conversational_test_results_{int(time.time())}.json"

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'modules_tested': [
                'QueryDetector (core/search/query_detection.py)',
                'KnowledgeGraphCore (core/knowledge_graph/kg_core.py)',
                'ConversationManager (conversation/manager.py)',
                'RAGPipeline (pipeline/rag_pipeline.py)'
            ],
            'total_turns': len(self.turn_results),
            'successful_turns': sum(1 for r in self.turn_results if r['success']),
            'metrics': {
                **self.metrics,
                'entities_extracted': self.metrics['entities_extracted'][:50]
            },
            'conversation_log': self.conversation_log,
            'turn_results': []
        }

        for result in self.turn_results:
            export_result = {
                'turn_num': result['turn_num'],
                'turn_type': result['turn_type'],
                'query': result['query'],
                'success': result['success'],
                'answer': result['answer'][:1000],  # Truncate for export
                'analysis': {
                    'query_type': result['analysis'].get('query_type'),
                    'is_followup': result['analysis'].get('is_followup'),
                    'is_topic_shift': result['analysis'].get('is_topic_shift'),
                    'regulation_references': result['analysis'].get('regulation_references', [])
                },
                'streaming_stats': result['streaming_stats'],
                'sources_count': len(result['metadata'].get('sources', []))
            }
            export_data['turn_results'].append(export_result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Results exported to: {output_path}")
        return output_path


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test conversational RAG using existing system modules"
    )
    parser.add_argument('--export', action='store_true', help='Export results to JSON')
    parser.add_argument('--output', type=str, help='Output file path for export')
    parser.add_argument('--verbose', action='store_true', help='Show detailed metadata')
    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                    ║
║   CONVERSATIONAL RAG TEST - Module Integration Verification                                        ║
║                                                                                                    ║
║   Testing: QueryDetector | KnowledgeGraphCore | ConversationManager | RAGPipeline                 ║
║                                                                                                    ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    tester = ConversationalTester(verbose=args.verbose)

    try:
        success = tester.run_full_conversation()

        if args.export:
            tester.export_results(args.output)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
