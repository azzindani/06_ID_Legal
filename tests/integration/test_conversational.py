"""
Conversational RAG Test - Multi-Turn Conversation with Memory and Context Management

This test demonstrates the system's ability to handle intelligent legal conversations:

1. CONVERSATION MEMORY - Maintains context across related questions
2. CONTEXT MANAGEMENT - Tracks topic continuity and detects topic shifts
3. SPECIFIC REGULATION RECOGNITION - Handles direct references (e.g., UU No. 13 Tahun 2003)
4. ENTITY EXTRACTION - Identifies legal entities, articles, and references
5. TOPIC CHANGE DETECTION - Gracefully handles conversation pivots
6. DYNAMIC COMMUNITY DETECTION - Groups related regulations thematically
7. FOLLOW-UP QUESTION HANDLING - Understands implicit references from prior turns

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

Author: Legal RAG System Test Suite
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


class ConversationalTester:
    """
    Tests multi-turn conversational capabilities with memory and context management.

    This class simulates a high-intelligence legal assistant handling:
    - Continuous conversations with context memory
    - Topic continuity and topic shifts
    - Specific regulation lookups
    - Follow-up questions with implicit references
    - Multi-domain legal knowledge
    """

    def __init__(self, verbose: bool = False):
        initialize_logging()
        self.logger = get_logger("ConversationalTest")
        self.verbose = verbose

        # Core components
        self.pipeline: Optional[RAGPipeline] = None
        self.conversation_manager: Optional[ConversationManager] = None
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
            'communities_discovered': [],
            'total_documents_retrieved': 0,
            'conversation_coherence_score': 0.0
        }

    def initialize(self) -> bool:
        """Initialize RAG pipeline and conversation manager"""
        self.logger.info("=" * 100)
        self.logger.info("INITIALIZING CONVERSATIONAL RAG SYSTEM")
        self.logger.info("=" * 100)

        try:
            # Initialize RAG Pipeline
            self.pipeline = RAGPipeline()
            if not self.pipeline.initialize():
                self.logger.error("Pipeline initialization failed")
                return False

            # Initialize Conversation Manager
            self.conversation_manager = ConversationManager({
                'max_history_turns': 50,
                'max_context_turns': 10  # Keep more context for follow-ups
            })

            # Start session
            self.session_id = self.conversation_manager.start_session()
            self.logger.info(f"Conversation session started: {self.session_id}")

            self.logger.success("System initialized successfully")
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
            context.append({"role": "assistant", "content": turn['answer'][:500]})  # Truncate for context
        return context

    def _analyze_topic_continuity(
        self,
        current_query: str,
        previous_queries: List[str]
    ) -> Dict[str, Any]:
        """Analyze if current query continues previous topic or shifts"""
        # Keywords for topic detection
        topic_keywords = {
            'ketenagakerjaan': ['pekerja', 'buruh', 'ketenagakerjaan', 'pesangon', 'phk', 'upah', 'tenaga kerja'],
            'lingkungan': ['lingkungan', 'izin lingkungan', 'amdal', 'pencemaran', 'limbah'],
            'perpajakan': ['pajak', 'perpajakan', 'wajib pajak', 'keberatan', 'banding'],
            'perusahaan': ['perseroan', 'pt', 'perusahaan', 'direksi', 'komisaris'],
            'konsumen': ['konsumen', 'perlindungan konsumen', 'produk', 'pengaduan']
        }

        def detect_topic(query: str) -> str:
            query_lower = query.lower()
            for topic, keywords in topic_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    return topic
            return 'general'

        current_topic = detect_topic(current_query)
        previous_topics = [detect_topic(q) for q in previous_queries] if previous_queries else []

        is_continuation = current_topic in previous_topics if previous_topics else True
        is_topic_shift = not is_continuation and len(previous_topics) > 0

        return {
            'current_topic': current_topic,
            'previous_topics': previous_topics,
            'is_continuation': is_continuation,
            'is_topic_shift': is_topic_shift
        }

    def _extract_regulation_reference(self, query: str) -> Optional[Dict[str, str]]:
        """Extract specific regulation reference from query"""
        import re

        # Pattern for Indonesian regulations: UU No. 13 Tahun 2003, PP 35/2021, etc.
        patterns = [
            r'UU\s*(?:No\.?|Nomor)\s*(\d+)\s*(?:Tahun|Thn|\/)\s*(\d{4})',
            r'PP\s*(?:No\.?|Nomor)\s*(\d+)\s*(?:Tahun|Thn|\/)\s*(\d{4})',
            r'Perpres\s*(?:No\.?|Nomor)\s*(\d+)\s*(?:Tahun|Thn|\/)\s*(\d{4})',
            r'Perda\s*(?:No\.?|Nomor)\s*(\d+)\s*(?:Tahun|Thn|\/)\s*(\d{4})',
            r'Permen\w*\s*(?:No\.?|Nomor)\s*(\d+)\s*(?:Tahun|Thn|\/)\s*(\d{4})'
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                reg_type = pattern.split(r'\s')[0].replace('\\', '').upper()
                return {
                    'type': reg_type,
                    'number': match.group(1),
                    'year': match.group(2),
                    'full_reference': match.group(0)
                }
        return None

    def _is_follow_up_question(self, query: str) -> bool:
        """Detect if query is a follow-up question"""
        follow_up_indicators = [
            'bagaimana jika', 'lalu bagaimana', 'selanjutnya',
            'apakah juga', 'terkait hal tersebut', 'mengenai hal itu',
            'tentang itu', 'tersebut', 'nya', 'yang tadi',
            'jika demikian', 'dalam hal ini', 'untuk kasus ini'
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in follow_up_indicators)

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
        lines.append(f"│ 📝 QUESTION                                                                              │")
        lines.append(f"├{'─' * 98}┤")
        lines.append(f"│ {query[:95]}{'...' if len(query) > 95 else ''} │")
        lines.append(f"└{'─' * 98}┘")

        # Analysis Section
        lines.append(f"\n┌{'─' * 98}┐")
        lines.append(f"│ 🔍 ANALYSIS                                                                              │")
        lines.append(f"├{'─' * 98}┤")

        topic_info = analysis.get('topic_analysis', {})
        lines.append(f"│ Topic: {topic_info.get('current_topic', 'N/A'):<20} | Continuation: {'✓' if topic_info.get('is_continuation') else '✗':<5} | Topic Shift: {'⚠️' if topic_info.get('is_topic_shift') else '—':<5} │")

        reg_ref = analysis.get('regulation_reference')
        if reg_ref:
            lines.append(f"│ 📜 Specific Regulation: {reg_ref.get('full_reference', 'N/A'):<67} │")

        if analysis.get('is_follow_up'):
            lines.append(f"│ 💭 Follow-up Question: Using context from previous turns                                 │")

        query_type = metadata.get('query_type', 'general')
        lines.append(f"│ Query Type: {query_type:<83} │")
        lines.append(f"└{'─' * 98}┘")

        # Thinking Process (if available)
        thinking = metadata.get('thinking', '')
        if thinking and self.verbose:
            lines.append(f"\n┌{'─' * 98}┐")
            lines.append(f"│ 🧠 THINKING PROCESS                                                                      │")
            lines.append(f"├{'─' * 98}┤")
            # Wrap thinking content
            for i in range(0, min(len(thinking), 500), 95):
                chunk = thinking[i:i+95]
                lines.append(f"│ {chunk:<95} │")
            if len(thinking) > 500:
                lines.append(f"│ ... [truncated - {len(thinking)} chars total]                                                  │")
            lines.append(f"└{'─' * 98}┘")

        # Answer
        lines.append(f"\n┌{'─' * 98}┐")
        lines.append(f"│ ✅ ANSWER                                                                                │")
        lines.append(f"├{'─' * 98}┤")
        # Wrap answer
        answer_lines = answer.split('\n')
        for ans_line in answer_lines[:15]:  # Show first 15 lines
            for i in range(0, len(ans_line), 95):
                chunk = ans_line[i:i+95]
                lines.append(f"│ {chunk:<95} │")
        if len(answer_lines) > 15:
            lines.append(f"│ ... [truncated - {len(answer_lines)} lines total]                                                 │")
        lines.append(f"└{'─' * 98}┘")

        # Sources/Citations
        sources = metadata.get('sources', metadata.get('citations', []))
        if sources:
            lines.append(f"\n┌{'─' * 98}┐")
            lines.append(f"│ 📚 LEGAL SOURCES ({len(sources)} documents)                                                      │")
            lines.append(f"├{'─' * 98}┤")
            for idx, source in enumerate(sources[:5], 1):
                reg_type = source.get('regulation_type', 'N/A')
                reg_num = source.get('regulation_number', 'N/A')
                year = source.get('year', 'N/A')
                score = source.get('score', 0)
                about = source.get('about', '')[:60]
                lines.append(f"│ {idx}. {reg_type} No. {reg_num}/{year} (Score: {score:.3f})                                      │")
                lines.append(f"│    {about}...                                                  │")
            if len(sources) > 5:
                lines.append(f"│ ... and {len(sources) - 5} more documents                                                      │")
            lines.append(f"└{'─' * 98}┘")

        # Context Memory Indicators
        if turn_num > 1:
            lines.append(f"\n┌{'─' * 98}┐")
            lines.append(f"│ 💾 CONVERSATION MEMORY                                                                   │")
            lines.append(f"├{'─' * 98}┤")
            context_used = analysis.get('context_turns_used', 0)
            lines.append(f"│ Previous turns in context: {context_used:<69} │")
            if topic_info.get('previous_topics'):
                prev_topics = ', '.join(set(topic_info['previous_topics']))
                lines.append(f"│ Previous topics: {prev_topics:<77} │")
            lines.append(f"└{'─' * 98}┘")

        # Timing
        lines.append(f"\n┌{'─' * 98}┐")
        lines.append(f"│ ⏱️  PERFORMANCE                                                                           │")
        lines.append(f"├{'─' * 98}┤")
        duration = streaming_stats.get('duration', 0)
        chunks = streaming_stats.get('chunk_count', 0)
        retrieval_time = metadata.get('retrieval_time', 0)
        lines.append(f"│ Streaming: {chunks} chunks in {duration:.2f}s | Retrieval: {retrieval_time:.2f}s                                      │")
        lines.append(f"└{'─' * 98}┘")

        return "\n".join(lines)

    def run_conversation_turn(
        self,
        query: str,
        turn_num: int,
        turn_type: str = "QUERY"
    ) -> Dict[str, Any]:
        """Run a single conversation turn with full analysis"""
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
            # Pre-query analysis
            previous_queries = [t['query'] for t in self.conversation_log]

            topic_analysis = self._analyze_topic_continuity(query, previous_queries)
            regulation_ref = self._extract_regulation_reference(query)
            is_follow_up = self._is_follow_up_question(query)

            analysis = {
                'turn_type': turn_type,
                'topic_analysis': topic_analysis,
                'regulation_reference': regulation_ref,
                'is_follow_up': is_follow_up,
                'context_turns_used': min(len(previous_queries), 5)
            }

            # Update metrics
            if topic_analysis['is_continuation'] and turn_num > 1:
                self.metrics['topic_continuity_detected'] += 1
            if topic_analysis['is_topic_shift']:
                self.metrics['topic_shifts_detected'] += 1
            if regulation_ref:
                self.metrics['specific_regulations_found'] += 1
            if is_follow_up:
                self.metrics['follow_up_context_used'] += 1

            # Get conversation context
            conversation_context = self._get_conversation_context()

            print(f"\n{'=' * 100}")
            print(f"TURN {turn_num}: {query[:80]}{'...' if len(query) > 80 else ''}")
            print("=" * 100)
            print("\n[STREAMING ANSWER]")
            print("-" * 80)

            full_answer = ""
            chunk_count = 0
            start_time = time.time()
            final_metadata = {}

            # Prepare query with context for follow-ups
            enriched_query = query
            if is_follow_up and conversation_context:
                # Add implicit context for better understanding
                enriched_query = f"[Melanjutkan pembahasan sebelumnya] {query}"

            # Stream the response
            for chunk in self.pipeline.query(enriched_query, stream=True):
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

            # Add turn to conversation manager
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
                'topic': topic_analysis['current_topic'],
                'timestamp': datetime.now().isoformat()
            })

            # Track documents retrieved
            sources = final_metadata.get('sources', [])
            self.metrics['total_documents_retrieved'] += len(sources)

            # Extract entities from sources
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
                'description': "Direct reference to UU No. 13 Tahun 2003 - tests specific regulation recognition"
            },
            {
                'query': "Bagaimana jika perusahaan tidak membayar pesangon tersebut? Apa upaya hukum yang dapat dilakukan pekerja?",
                'type': "FOLLOW-UP",
                'description': "Follow-up using context from Q2 - tests conversation memory"
            },
            # TOPIC 2: Lingkungan Hidup (Environmental Law) - 1 Question
            {
                'query': "Sekarang saya ingin bertanya tentang izin lingkungan. Apa persyaratan untuk mendapatkan izin lingkungan berdasarkan peraturan yang berlaku?",
                'type': "TOPIC SHIFT",
                'description': "Topic change to environmental law - tests context switching"
            },
            # TOPIC 3: Perpajakan (Tax Law) - 1 Question
            {
                'query': "Pertanyaan terakhir mengenai perpajakan. Bagaimana mekanisme pengajuan keberatan pajak dan apa saja syaratnya?",
                'type': "NEW DOMAIN",
                'description': "Different domain (tax law) - tests multi-domain capability"
            }
        ]

        self.logger.info("\n" + "═" * 100)
        self.logger.info("  CONVERSATIONAL RAG TEST - MULTI-TURN INTELLIGENT LEGAL ASSISTANT")
        self.logger.info("═" * 100)
        self.logger.info(f"\n  Session ID: {self.session_id}")
        self.logger.info(f"  Questions: {len(conversation_script)}")
        self.logger.info(f"  Topics: Ketenagakerjaan (3) → Lingkungan (1) → Perpajakan (1)")
        self.logger.info("\n  Testing:")
        self.logger.info("    ✓ Conversation Memory")
        self.logger.info("    ✓ Context Management")
        self.logger.info("    ✓ Specific Regulation Recognition (UU 13/2003)")
        self.logger.info("    ✓ Follow-up Question Handling")
        self.logger.info("    ✓ Topic Shift Detection")
        self.logger.info("    ✓ Multi-Domain Knowledge")
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

                # Pause between turns to simulate real conversation
                if i < len(conversation_script):
                    self.logger.info("\n⏳ Simulating conversation pause...")
                    time.sleep(3)

            # Calculate conversation coherence score
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

        # Score based on topic continuity
        if self.metrics['topic_continuity_detected'] >= 2:  # Expected 2 continuations
            scores.append(1.0)
        else:
            scores.append(self.metrics['topic_continuity_detected'] / 2.0)

        # Score based on topic shifts handled
        if self.metrics['topic_shifts_detected'] >= 2:  # Expected 2 shifts
            scores.append(1.0)
        else:
            scores.append(self.metrics['topic_shifts_detected'] / 2.0)

        # Score based on specific regulation recognition
        if self.metrics['specific_regulations_found'] >= 1:
            scores.append(1.0)
        else:
            scores.append(0.0)

        # Score based on follow-up handling
        if self.metrics['follow_up_context_used'] >= 1:
            scores.append(1.0)
        else:
            scores.append(0.0)

        # Score based on successful turns
        success_rate = sum(1 for r in self.turn_results if r['success']) / len(self.turn_results)
        scores.append(success_rate)

        return sum(scores) / len(scores) if scores else 0.0

    def _display_final_summary(self, successful: int, total: int):
        """Display comprehensive final summary"""
        print("\n" + "═" * 100)
        print("  CONVERSATIONAL TEST SUMMARY")
        print("═" * 100)

        print(f"\n  📊 RESULTS: {successful}/{total} turns successful")
        print(f"  🆔 Session ID: {self.session_id}")
        print(f"  🎯 Coherence Score: {self.metrics['conversation_coherence_score']:.1%}")

        print("\n  " + "─" * 96)
        print("  TURN-BY-TURN RESULTS")
        print("  " + "─" * 96)

        for result in self.turn_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            turn_type = result['turn_type']
            query_preview = result['query'][:50] + "..." if len(result['query']) > 50 else result['query']
            topic = result['analysis'].get('topic_analysis', {}).get('current_topic', 'N/A')

            print(f"  Turn {result['turn_num']}: [{status}] [{turn_type}]")
            print(f"         Query: {query_preview}")
            print(f"         Topic: {topic}")

            if result['success']:
                stats = result['streaming_stats']
                sources = result['metadata'].get('sources', [])
                print(f"         Chunks: {stats.get('chunk_count', 0)} | Duration: {stats.get('duration', 0):.2f}s | Sources: {len(sources)}")
            print()

        print("  " + "─" * 96)
        print("  CONVERSATION INTELLIGENCE METRICS")
        print("  " + "─" * 96)

        print(f"\n  💾 Memory & Context:")
        print(f"     • Topic continuity detected: {self.metrics['topic_continuity_detected']} times")
        print(f"     • Topic shifts handled: {self.metrics['topic_shifts_detected']} times")
        print(f"     • Follow-up context used: {self.metrics['follow_up_context_used']} times")

        print(f"\n  📜 Regulation Recognition:")
        print(f"     • Specific regulations identified: {self.metrics['specific_regulations_found']}")
        print(f"     • Total entities extracted: {len(self.metrics['entities_extracted'])}")

        if self.metrics['entities_extracted'][:10]:
            print(f"     • Sample entities: {', '.join(self.metrics['entities_extracted'][:5])}")

        print(f"\n  📚 Document Retrieval:")
        print(f"     • Total documents retrieved: {self.metrics['total_documents_retrieved']}")
        print(f"     • Average per turn: {self.metrics['total_documents_retrieved'] / total:.1f}")

        # Coherence breakdown
        print(f"\n  🎯 Coherence Score Breakdown:")
        print(f"     • Topic Continuity: {'✓' if self.metrics['topic_continuity_detected'] >= 2 else '○'}")
        print(f"     • Topic Shift Handling: {'✓' if self.metrics['topic_shifts_detected'] >= 2 else '○'}")
        print(f"     • Regulation Recognition: {'✓' if self.metrics['specific_regulations_found'] >= 1 else '○'}")
        print(f"     • Follow-up Handling: {'✓' if self.metrics['follow_up_context_used'] >= 1 else '○'}")
        print(f"     • Overall Success Rate: {successful}/{total}")

        print("\n" + "═" * 100)

        # Final verdict
        if self.metrics['conversation_coherence_score'] >= 0.8:
            print("  🏆 VERDICT: EXCELLENT - System demonstrates high conversational intelligence")
        elif self.metrics['conversation_coherence_score'] >= 0.6:
            print("  ✅ VERDICT: GOOD - System handles most conversational scenarios")
        elif self.metrics['conversation_coherence_score'] >= 0.4:
            print("  ⚠️  VERDICT: FAIR - Some conversational features need improvement")
        else:
            print("  ❌ VERDICT: NEEDS WORK - Significant conversational improvements needed")

        print("═" * 100 + "\n")

    def export_results(self, output_path: Optional[str] = None) -> str:
        """Export all results to JSON"""
        if not output_path:
            output_path = f"conversational_test_results_{int(time.time())}.json"

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'total_turns': len(self.turn_results),
            'successful_turns': sum(1 for r in self.turn_results if r['success']),
            'metrics': {
                **self.metrics,
                'entities_extracted': self.metrics['entities_extracted'][:50]  # Limit for export
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
                'answer': result['answer'],
                'analysis': result['analysis'],
                'streaming_stats': result['streaming_stats'],
                # Exclude very large fields
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
        description="Test conversational RAG with memory and context management"
    )
    parser.add_argument('--export', action='store_true', help='Export results to JSON')
    parser.add_argument('--output', type=str, help='Output file path for export')
    parser.add_argument('--verbose', action='store_true', help='Show detailed metadata')
    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                    ║
║   ██╗     ███████╗ ██████╗  █████╗ ██╗         ██████╗  █████╗  ██████╗                           ║
║   ██║     ██╔════╝██╔════╝ ██╔══██╗██║         ██╔══██╗██╔══██╗██╔════╝                           ║
║   ██║     █████╗  ██║  ███╗███████║██║         ██████╔╝███████║██║  ███╗                          ║
║   ██║     ██╔══╝  ██║   ██║██╔══██║██║         ██╔══██╗██╔══██║██║   ██║                          ║
║   ███████╗███████╗╚██████╔╝██║  ██║███████╗    ██║  ██║██║  ██║╚██████╔╝                          ║
║   ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝                           ║
║                                                                                                    ║
║   CONVERSATIONAL INTELLIGENCE TEST                                                                 ║
║   Testing: Memory | Context | Topic Shifts | Regulation Recognition | Follow-ups                  ║
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
