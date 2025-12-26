"""
Complete RAG Output Test with Streaming
Tests comprehensive output including all metadata for auditing

This test demonstrates:
1. Streaming LLM responses using TextIteratorStreamer
2. Complete metadata for ALL retrieved regulations (not just cited)
3. Research process transparency with phase-by-phase breakdown
4. Full scoring information for audit purposes

Run with:
    python tests/integration/test_complete_output.py

You'll see complete RAG output with real-time streaming!
"""

import sys
import os
import time
import json
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import LOG_DIR, ENABLE_FILE_LOGGING, LOG_VERBOSITY
from utils.logger_utils import get_logger, initialize_logging
from pipeline import RAGPipeline
from utils.formatting import _extract_all_documents_from_metadata
from utils.research_transparency import format_detailed_research_process


class CompleteOutputTester:
    """Tests comprehensive RAG output with streaming and full metadata"""

    def __init__(self, thinking_mode: str = 'low'):
        initialize_logging(
        enable_file_logging=ENABLE_FILE_LOGGING,
        log_dir=LOG_DIR,
        verbosity_mode=LOG_VERBOSITY
    )
        self.logger = get_logger("CompleteOutputTest")
        self.pipeline: Optional[RAGPipeline] = None
        self.results: List[Dict[str, Any]] = []
        self.thinking_mode = thinking_mode

    def initialize(self) -> bool:
        """Initialize RAG pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING RAG PIPELINE")
        self.logger.info("=" * 80)

        try:
            self.pipeline = RAGPipeline()
            if not self.pipeline.initialize():
                self.logger.error("Pipeline initialization failed")
                return False

            self.logger.success("Pipeline initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def format_complete_output(
        self,
        query: str,
        answer: str,
        metadata: Dict[str, Any],
        streaming_stats: Dict[str, Any]
    ) -> str:
        """Format complete output with all metadata for auditing"""
        lines = []

        lines.append("\n" + "=" * 100)
        lines.append("COMPLETE RAG OUTPUT")
        lines.append("=" * 100)

        # Question
        lines.append(f"\n## QUESTION")
        lines.append("-" * 80)
        lines.append(query)

        # Query Type
        query_type = metadata.get('query_type', 'general')
        lines.append(f"\n## QUERY TYPE: {query_type}")

        # Thinking Process
        thinking = metadata.get('thinking', '')
        if thinking:
            lines.append(f"\n## THINKING PROCESS")
            lines.append("-" * 80)
            lines.append(thinking)

        # Complete Prompt - FULL TRANSPARENCY
        complete_prompt = metadata.get('complete_prompt', '')
        if complete_prompt:
            lines.append(f"\n## COMPLETE LLM INPUT PROMPT (FULL TRANSPARENCY)")
            lines.append("=" * 100)
            lines.append(f"Character Count: {len(complete_prompt):,}")
            lines.append("-" * 80)
            lines.append(complete_prompt)
            lines.append("-" * 80)

        # Answer (already streamed, show final)
        lines.append(f"\n## ANSWER")
        lines.append("-" * 80)
        lines.append(answer)
        lines.append(f"\n[Streamed: {streaming_stats.get('chunk_count', 0)} chunks in {streaming_stats.get('duration', 0):.2f}s]")

        # Legal References - TOP K documents used as LLM prompt input
        lines.append(f"\n## LEGAL REFERENCES (Top K Documents Used in LLM Prompt)")
        lines.append("-" * 80)
        lines.append("These are the final selected documents sent to the LLM for answer generation.")
        lines.append("")

        # Get sources/citations - these are the top k final results
        sources = metadata.get('sources', metadata.get('citations', []))

        if sources:
            lines.append(f"Documents Used in Prompt: {len(sources)}")
            lines.append("")

            for idx, source in enumerate(sources, 1):
                reg_type = source.get('regulation_type', 'N/A')
                reg_num = source.get('regulation_number', 'N/A')
                year = source.get('year', 'N/A')
                about = source.get('about', 'N/A')
                enacting_body = source.get('enacting_body', 'N/A')
                score = source.get('score', 0)
                content = source.get('content', '')

                lines.append(f"### {idx}. {reg_type} No. {reg_num}/{year}")
                lines.append(f"   About: {about}")
                lines.append(f"   Enacting Body: {enacting_body}")
                lines.append(f"   Final Score: {score:.4f}")

                # Content preview (truncated for readability)
                if content:
                    content_preview = content[:800].replace('\n', ' ')
                    lines.append(f"   Content Preview: {content_preview}...")

                lines.append("")
        else:
            lines.append("No documents in prompt (check retrieval)")

        # Use new detailed research process formatter
        detailed_research = format_detailed_research_process(
            metadata,
            top_n_per_researcher=20,
            show_content=True
        )
        lines.append(detailed_research)
        lines.append("")

        # Also keep the old format for compatibility
        lines.append(f"\n## LEGACY FORMAT: ALL Retrieved Documents")
        lines.append("-" * 80)
        lines.append("All documents retrieved during research process - for audit and verification.")
        lines.append("Content is truncated for readability. Use export for full content.")
        lines.append("")

        # Try multiple sources for research data
        research_log = metadata.get('research_log', {})
        phase_metadata = metadata.get('phase_metadata', metadata.get('all_retrieved_metadata', {}))

        # Also extract all documents for the detailed listing
        all_documents = _extract_all_documents_from_metadata(metadata)

        has_research_info = False

        if research_log or phase_metadata or all_documents:
            has_research_info = True

            # Team members from research_log
            team_members = research_log.get('team_members', [])
            if team_members:
                lines.append(f"### Research Team")
                lines.append(f"Team Size: {len(team_members)}")
                for member in team_members:
                    if isinstance(member, dict):
                        lines.append(f"   - {member.get('name', member.get('persona', 'Unknown'))}")
                    else:
                        lines.append(f"   - {member}")
            else:
                # Extract team members from phase_metadata
                unique_researchers = set()
                for phase_key, phase_data in phase_metadata.items():
                    if isinstance(phase_data, dict):
                        researcher = phase_data.get('researcher_name', phase_data.get('researcher', ''))
                        if researcher:
                            unique_researchers.add(researcher)
                if unique_researchers:
                    lines.append(f"### Research Team")
                    lines.append(f"Team Size: {len(unique_researchers)}")
                    for member in unique_researchers:
                        lines.append(f"   - {member}")

            # Summary Statistics
            total_docs = research_log.get('total_documents_retrieved', 0)
            if not total_docs:
                total_docs = len(all_documents)
            if not total_docs and phase_metadata:
                total_docs = sum(
                    len(pm.get('candidates', pm.get('results', [])))
                    for pm in phase_metadata.values() if isinstance(pm, dict)
                )

            lines.append(f"\n### Summary Statistics")
            lines.append(f"Total Documents Retrieved: {total_docs}")
            lines.append(f"Total Phases: {len(phase_metadata)}")
            lines.append("")

            # ALL DOCUMENTS with full article-level metadata
            lines.append(f"### ALL Retrieved Documents (Article-Level Details)")
            lines.append("=" * 100)

            if all_documents:
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
                    final_score = scores.get('final', doc.get('final_score', doc.get('composite_score', 0)))
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

                    # Team consensus
                    team_consensus = doc.get('team_consensus', False)
                    researcher_agreement = doc.get('researcher_agreement', 0)

                    lines.append(f"\n### [{i}] {reg_type} No. {reg_num}/{year}")
                    lines.append(f"    Global ID: {global_id}")
                    lines.append(f"    About: {about}")
                    lines.append(f"    Enacting Body: {enacting_body}")

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
                        lines.append(f"    Location: {' > '.join(location_parts)}")
                    else:
                        lines.append(f"    Location: (Full Document)")

                    # All scores
                    lines.append(f"    Scores:")
                    lines.append(f"       Final: {final_score:.4f} | Semantic: {semantic:.4f} | Keyword: {keyword:.4f}")
                    lines.append(f"       KG: {kg:.4f} | Authority: {authority:.4f} | Temporal: {temporal:.4f} | Completeness: {completeness:.4f}")

                    # KG metadata
                    if kg_domain or kg_hierarchy:
                        lines.append(f"    Knowledge Graph: Domain={kg_domain or 'N/A'} | Hierarchy={kg_hierarchy} | CrossRefs={kg_cross_refs}")

                    # Research info
                    if phase or researcher:
                        lines.append(f"    Discovery: Phase={phase} | Researcher={researcher}")

                    # Team consensus
                    lines.append(f"    Team Consensus: {'Yes' if team_consensus else 'No'} | Agreement: {researcher_agreement}")

                    # Content (TRUNCATED - 300 chars)
                    content = record.get('content', '')
                    if content:
                        content_truncated = content[:300].replace('\n', ' ').strip()
                        lines.append(f"    Content (truncated): {content_truncated}...")

                    lines.append("")
                    lines.append("-" * 100)

            # Phase breakdown summary
            if phase_metadata:
                lines.append(f"\n### Phase Breakdown Summary")
                lines.append("=" * 80)

                for phase_key, phase_data in phase_metadata.items():
                    if isinstance(phase_data, dict):
                        phase_name = phase_data.get('phase', phase_key)
                        researcher = phase_data.get('researcher_name', phase_data.get('researcher', 'Unknown'))
                        candidates = phase_data.get('candidates', phase_data.get('results', []))
                        confidence = phase_data.get('confidence', 1.0)

                        lines.append(f"\n   Phase: {phase_name}")
                        lines.append(f"   Researcher: {researcher}")
                        lines.append(f"   Documents: {len(candidates)}")
                        lines.append(f"   Confidence: {confidence:.2%}")

                        # List document IDs in this phase
                        if candidates:
                            doc_ids = []
                            for c in candidates[:10]:  # Show first 10
                                r = c.get('record', c)
                                doc_ids.append(f"{r.get('regulation_type', 'N/A')} {r.get('regulation_number', '')}/{r.get('year', '')}")
                            lines.append(f"   Documents: {', '.join(doc_ids)}")
                            if len(candidates) > 10:
                                lines.append(f"   ... and {len(candidates) - 10} more")

        if not has_research_info:
            lines.append("Research process details not available")

        # Timing Information
        lines.append(f"\n## TIMING")
        lines.append("-" * 80)
        total_time = metadata.get('total_time', 0)
        retrieval_time = metadata.get('retrieval_time', 0)
        generation_time = metadata.get('generation_time', streaming_stats.get('duration', 0))

        lines.append(f"Total Time: {total_time:.3f}s")
        lines.append(f"Retrieval Time: {retrieval_time:.3f}s")
        lines.append(f"Generation Time: {generation_time:.3f}s")
        lines.append(f"Tokens Generated: {metadata.get('tokens_generated', streaming_stats.get('chunk_count', 0))}")

        lines.append("\n" + "=" * 100)

        return "\n".join(lines)

    # _extract_all_documents is now imported from utils.formatting as _extract_all_documents_from_metadata

    def run_query_with_streaming(self, query: str, query_num: int) -> Dict[str, Any]:
        """Run a single query with streaming output"""
        self.logger.info(f"\n{'#' * 100}")
        self.logger.info(f"QUERY {query_num}/5")
        self.logger.info(f"{'#' * 100}")

        result = {
            'query': query,
            'query_num': query_num,
            'success': False,
            'answer': '',
            'metadata': {},
            'streaming_stats': {}
        }

        try:
            print(f"\n{'=' * 100}")
            print(f"QUERY {query_num}: {query}")
            print("=" * 100)
            print("\n[STREAMING ANSWER]")
            print("-" * 80)

            full_answer = ""
            chunk_count = 0
            start_time = time.time()
            final_metadata = {}

            # Stream the response
            thinking_content = ""
            phase_metadata = {}
            research_log = {}
            sources = []
            citations = []

            for chunk in self.pipeline.query(query, stream=True, thinking_mode=self.thinking_mode):
                chunk_type = chunk.get('type', '')

                if chunk_type == 'token':
                    token = chunk.get('token', '')
                    print(token, end='', flush=True)
                    full_answer += token
                    chunk_count += 1

                elif chunk_type == 'thinking':
                    # Thinking chunk - print CoT content
                    token = chunk.get('token', '')
                    print(token, end='', flush=True)
                    chunk_count += 1

                elif chunk_type == 'complete':
                    full_answer = chunk.get('answer', full_answer)
                    final_metadata = chunk.get('metadata', {})
                    # Extract ALL metadata from the complete chunk
                    thinking_content = chunk.get('thinking', '')
                    phase_metadata = chunk.get('phase_metadata', chunk.get('all_retrieved_metadata', {}))
                    research_log = chunk.get('research_log', {})
                    sources = chunk.get('sources', [])
                    citations = chunk.get('citations', [])
                    # Store complete prompt for transparency
                    complete_prompt = chunk.get('complete_prompt', '')
                    # Also include consensus and research data
                    final_metadata['consensus_data'] = chunk.get('consensus_data', {})
                    final_metadata['research_data'] = chunk.get('research_data', {})
                    final_metadata['complete_prompt'] = complete_prompt

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

            # Merge all metadata for formatting
            full_metadata = {
                **final_metadata,
                'thinking': thinking_content,
                'phase_metadata': phase_metadata,
                'research_log': research_log,
                'sources': sources,
                'citations': citations
            }

            # Display complete formatted output
            complete_output = self.format_complete_output(
                query=query,
                answer=full_answer,
                metadata=full_metadata,
                streaming_stats=streaming_stats
            )
            print(complete_output)

            result['success'] = True
            result['answer'] = full_answer
            result['metadata'] = full_metadata  # Use full metadata with all research info
            result['streaming_stats'] = streaming_stats
            result['formatted_output'] = complete_output

            self.logger.success(f"Query {query_num} completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Query {query_num} failed: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
            return result

    def run_all_queries(self) -> bool:
        """Run 5 different queries with streaming and complete metadata"""

        # 5 diverse legal queries in Indonesian
        queries = [
            "Apa saja hak-hak pekerja menurut UU Ketenagakerjaan?",
            "Jelaskan sanksi pidana dalam UU Perlindungan Data Pribadi",
            "Bagaimana prosedur pendirian perseroan terbatas menurut UU Perseroan Terbatas?",
            "Apa yang dimaksud dengan perlindungan konsumen dan bagaimana mekanisme pengaduannya?",
            "Jelaskan tentang hak cipta dan perlindungan kekayaan intelektual di Indonesia"
        ]

        self.logger.info("\n" + "=" * 100)
        self.logger.info("COMPREHENSIVE RAG OUTPUT TEST - 5 QUERIES WITH STREAMING")
        self.logger.info("=" * 100)
        self.logger.info(f"Running {len(queries)} queries with full metadata output...")

        # Initialize pipeline
        if not self.initialize():
            return False

        try:
            successful = 0

            for i, query in enumerate(queries, 1):
                result = self.run_query_with_streaming(query, i)
                self.results.append(result)

                if result['success']:
                    successful += 1

                # Brief pause between queries
                if i < len(queries):
                    self.logger.info("\nPausing before next query...")
                    time.sleep(2)

            # Final Summary
            self.logger.info("\n" + "=" * 100)
            self.logger.info("TEST SUMMARY")
            self.logger.info("=" * 100)

            for result in self.results:
                status = "PASS" if result['success'] else "FAIL"
                query_preview = result['query'][:50] + "..." if len(result['query']) > 50 else result['query']
                self.logger.info(f"Query {result['query_num']}: [{status}] {query_preview}")

                if result['success']:
                    stats = result['streaming_stats']
                    self.logger.info(f"   - Chunks: {stats.get('chunk_count', 0)}, Duration: {stats.get('duration', 0):.2f}s")

                    # Document count
                    all_docs = _extract_all_documents_from_metadata(result['metadata'])
                    self.logger.info(f"   - Documents Retrieved: {len(all_docs)}")

            self.logger.info("-" * 100)
            self.logger.info(f"RESULT: {successful}/{len(queries)} queries successful")
            self.logger.info("=" * 100)

            return successful == len(queries)

        finally:
            if self.pipeline:
                self.logger.info("Shutting down pipeline...")
                self.pipeline.shutdown()

    def export_results(self, output_path: Optional[str] = None) -> str:
        """Export all results to JSON for further processing"""
        if not output_path:
            output_path = f"complete_output_results_{int(time.time())}.json"

        export_data = {
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_queries': len(self.results),
            'successful_queries': sum(1 for r in self.results if r['success']),
            'results': []
        }

        for result in self.results:
            export_result = {
                'query_num': result['query_num'],
                'query': result['query'],
                'success': result['success'],
                'answer': result['answer'],
                'streaming_stats': result['streaming_stats'],
                'metadata': result['metadata'],
                'formatted_output': result.get('formatted_output', '')
            }
            export_data['results'].append(export_result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Results exported to: {output_path}")
        return output_path


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Test complete RAG output with streaming")
    parser.add_argument('--export', action='store_true', help='Export results to JSON')
    parser.add_argument('--output', type=str, help='Output file path for export')

    # Add thinking mode arguments (mutually exclusive)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument('--low', action='store_const', const='low', dest='thinking_mode',
                               help='Low thinking mode (2048-4096 tokens, basic analysis)')
    thinking_group.add_argument('--medium', action='store_const', const='medium', dest='thinking_mode',
                               help='Medium thinking mode (4096-8192 tokens, deep thinking)')
    thinking_group.add_argument('--high', action='store_const', const='high', dest='thinking_mode',
                               help='High thinking mode (8192-16384 tokens, iterative & recursive)')
    parser.set_defaults(thinking_mode='low')

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"THINKING MODE: {args.thinking_mode.upper()}")
    print(f"{'=' * 80}\n")

    tester = CompleteOutputTester(thinking_mode=args.thinking_mode)

    try:
        success = tester.run_all_queries()

        if args.export:
            tester.export_results(args.output)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
