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

from logger_utils import get_logger, initialize_logging
from pipeline import RAGPipeline


class CompleteOutputTester:
    """Tests comprehensive RAG output with streaming and full metadata"""

    def __init__(self):
        initialize_logging()
        self.logger = get_logger("CompleteOutputTest")
        self.pipeline: Optional[RAGPipeline] = None
        self.results: List[Dict[str, Any]] = []

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

        # Answer (already streamed, show final)
        lines.append(f"\n## ANSWER")
        lines.append("-" * 80)
        lines.append(answer)
        lines.append(f"\n[Streamed: {streaming_stats.get('chunk_count', 0)} chunks in {streaming_stats.get('duration', 0):.2f}s]")

        # Legal References - ALL retrieved documents with FULL details
        lines.append(f"\n## LEGAL REFERENCES (All Retrieved Documents - FULL DETAILS)")
        lines.append("-" * 80)

        # Get all retrieved documents from phase_metadata or research_data
        all_documents = self._extract_all_documents(metadata)

        if all_documents:
            lines.append(f"Total Documents Retrieved: {len(all_documents)}")
            lines.append("")

            for idx, doc in enumerate(all_documents, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})

                # Basic info
                reg_type = record.get('regulation_type', 'N/A')
                reg_num = record.get('regulation_number', 'N/A')
                year = record.get('year', 'N/A')
                about = record.get('about', 'N/A')
                enacting_body = record.get('enacting_body', 'N/A')
                global_id = record.get('global_id', 'N/A')

                lines.append(f"### {idx}. {reg_type} No. {reg_num}/{year}")
                lines.append(f"   Global ID: {global_id}")
                lines.append(f"   About: {about}")  # Full about, not truncated
                lines.append(f"   Enacting Body: {enacting_body}")

                # Article/Chapter/Section details if available
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', ''))
                section = record.get('section', record.get('bagian', ''))
                paragraph = record.get('paragraph', record.get('ayat', ''))

                if chapter or article or section or paragraph:
                    lines.append(f"   Location in Document:")
                    if chapter:
                        lines.append(f"      Chapter/Bab: {chapter}")
                    if section:
                        lines.append(f"      Section/Bagian: {section}")
                    if article:
                        lines.append(f"      Article/Pasal: {article}")
                    if paragraph:
                        lines.append(f"      Paragraph/Ayat: {paragraph}")

                # Scores - FULL breakdown
                final_score = scores.get('final', doc.get('final_score', doc.get('composite_score', 0)))
                semantic_score = scores.get('semantic', doc.get('semantic_score', 0))
                keyword_score = scores.get('keyword', doc.get('keyword_score', 0))
                kg_score = scores.get('kg', doc.get('kg_score', 0))
                authority_score = scores.get('authority', doc.get('authority_score', 0))
                temporal_score = scores.get('temporal', doc.get('temporal_score', 0))
                completeness_score = scores.get('completeness', doc.get('completeness_score', 0))

                lines.append(f"   Relevance Scores:")
                lines.append(f"      Final Score: {final_score:.4f}")
                lines.append(f"      Semantic: {semantic_score:.4f}")
                lines.append(f"      Keyword: {keyword_score:.4f}")
                lines.append(f"      KG Score: {kg_score:.4f}")
                lines.append(f"      Authority: {authority_score:.4f}")
                lines.append(f"      Temporal: {temporal_score:.4f}")
                lines.append(f"      Completeness: {completeness_score:.4f}")

                # KG Metadata - FULL details
                kg_domain = record.get('kg_primary_domain', record.get('primary_domain', ''))
                kg_hierarchy = record.get('kg_hierarchy_level', record.get('hierarchy_level', 0))
                kg_cross_refs = record.get('kg_cross_ref_count', record.get('cross_ref_count', 0))
                kg_communities = record.get('kg_communities', record.get('communities', []))

                if kg_domain or kg_hierarchy or kg_cross_refs:
                    lines.append(f"   Knowledge Graph Metadata:")
                    lines.append(f"      Domain: {kg_domain or 'N/A'}")
                    lines.append(f"      Hierarchy Level: {kg_hierarchy}")
                    lines.append(f"      Cross References: {kg_cross_refs}")
                    if kg_communities:
                        lines.append(f"      Communities: {kg_communities[:5]}")

                # Team Consensus - FULL details
                team_consensus = doc.get('team_consensus', False)
                researcher_agreement = doc.get('researcher_agreement', 0)
                personas_agreed = doc.get('personas_agreed', [])

                lines.append(f"   Research Team Analysis:")
                lines.append(f"      Team Consensus: {'Yes' if team_consensus else 'No'}")
                lines.append(f"      Researcher Agreement: {researcher_agreement}")
                if personas_agreed:
                    lines.append(f"      Personas Agreed: {', '.join(str(p) for p in personas_agreed)}")

                # Phase info if available
                phase = doc.get('_phase', '')
                researcher = doc.get('_researcher', '')
                if phase or researcher:
                    lines.append(f"   Discovery Info:")
                    if phase:
                        lines.append(f"      Phase: {phase}")
                    if researcher:
                        lines.append(f"      Discovered By: {researcher}")

                # FULL Content - not truncated
                content = record.get('content', '')
                if content:
                    lines.append(f"   Full Content ({len(content)} chars):")
                    lines.append(f"   " + "-" * 60)
                    # Indent content for readability
                    for content_line in content.split('\n'):
                        lines.append(f"      {content_line}")
                    lines.append(f"   " + "-" * 60)

                lines.append("")  # Blank line between documents
                lines.append("~" * 80)  # Separator
                lines.append("")
        else:
            # Fall back to sources/citations if no detailed metadata
            sources = metadata.get('sources', metadata.get('citations', []))
            if sources:
                lines.append(f"Cited Sources: {len(sources)}")
                for idx, source in enumerate(sources, 1):
                    reg_type = source.get('regulation_type', 'N/A')
                    reg_num = source.get('regulation_number', 'N/A')
                    year = source.get('year', 'N/A')
                    about = source.get('about', 'N/A')
                    content = source.get('content', '')

                    lines.append(f"\n### {idx}. {reg_type} No. {reg_num}/{year}")
                    lines.append(f"   About: {about}")
                    if content:
                        lines.append(f"   Content ({len(content)} chars):")
                        for content_line in content.split('\n')[:20]:  # First 20 lines
                            lines.append(f"      {content_line}")
                        if len(content.split('\n')) > 20:
                            lines.append(f"      ... (truncated)")
            else:
                lines.append("No documents retrieved")

        # Research Process Details - FULL DETAILS for each phase
        lines.append(f"\n## RESEARCH PROCESS DETAILS (FULL)")
        lines.append("-" * 80)

        # Try multiple sources for research data
        research_log = metadata.get('research_log', {})
        phase_metadata = metadata.get('phase_metadata', metadata.get('all_retrieved_metadata', {}))

        has_research_info = False

        if research_log or phase_metadata:
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

            # Total documents
            total_docs = research_log.get('total_documents_retrieved', 0)
            if not total_docs and phase_metadata:
                total_docs = sum(
                    len(pm.get('candidates', pm.get('results', [])))
                    for pm in phase_metadata.values() if isinstance(pm, dict)
                )
            lines.append(f"\n### Summary Statistics")
            lines.append(f"Total Documents Retrieved: {total_docs}")
            lines.append(f"Total Phases: {len(phase_metadata)}")

            # Phase results from phase_metadata - FULL DETAILS
            if phase_metadata:
                lines.append(f"\n### Phase-by-Phase Breakdown (ALL Documents)")
                lines.append("=" * 80)

                for phase_key, phase_data in phase_metadata.items():
                    if isinstance(phase_data, dict):
                        phase_name = phase_data.get('phase', phase_key)
                        researcher = phase_data.get('researcher_name', phase_data.get('researcher', 'Unknown'))
                        candidates = phase_data.get('candidates', phase_data.get('results', []))
                        confidence = phase_data.get('confidence', 1.0)

                        lines.append(f"\n#### PHASE: {phase_name}")
                        lines.append(f"   Researcher: {researcher}")
                        lines.append(f"   Documents Found: {len(candidates)}")
                        lines.append(f"   Confidence: {confidence:.2%}")
                        lines.append("")

                        # Show ALL documents in this phase with FULL details
                        if candidates:
                            lines.append(f"   Documents Retrieved in This Phase:")
                            lines.append(f"   " + "-" * 70)

                            for i, doc in enumerate(candidates, 1):  # Show ALL documents, not just first 5
                                record = doc.get('record', doc)
                                scores = doc.get('scores', {})

                                reg_type = record.get('regulation_type', 'N/A')
                                reg_num = record.get('regulation_number', 'N/A')
                                year = record.get('year', 'N/A')
                                about = record.get('about', 'N/A')
                                global_id = record.get('global_id', 'N/A')

                                # Scores
                                final_score = scores.get('final', doc.get('composite_score', 0))
                                semantic = scores.get('semantic', doc.get('semantic_score', 0))
                                keyword = scores.get('keyword', doc.get('keyword_score', 0))
                                kg = scores.get('kg', doc.get('kg_score', 0))
                                authority = scores.get('authority', doc.get('authority_score', 0))
                                temporal = scores.get('temporal', doc.get('temporal_score', 0))

                                # Article-level details
                                chapter = record.get('chapter', record.get('bab', ''))
                                article = record.get('article', record.get('pasal', ''))
                                section = record.get('section', record.get('bagian', ''))
                                paragraph = record.get('paragraph', record.get('ayat', ''))

                                lines.append(f"\n   [{i}] {reg_type} No. {reg_num}/{year}")
                                lines.append(f"       ID: {global_id}")
                                lines.append(f"       About: {about}")

                                # Show article/chapter if available
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
                                    lines.append(f"       Location: {' > '.join(location_parts)}")

                                lines.append(f"       Scores: Final={final_score:.4f} | Semantic={semantic:.4f} | "
                                           f"Keyword={keyword:.4f} | KG={kg:.4f} | Authority={authority:.4f} | "
                                           f"Temporal={temporal:.4f}")

                                # Show content preview (first 500 chars)
                                content = record.get('content', '')
                                if content:
                                    content_preview = content[:500].replace('\n', ' ')
                                    lines.append(f"       Content Preview: {content_preview}...")

                            lines.append(f"   " + "-" * 70)

                        lines.append("")
                        lines.append("~" * 80)

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

    def _extract_all_documents(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all retrieved documents from various metadata locations"""
        all_docs = []
        seen_ids = set()

        # Try phase_metadata first (most complete)
        phase_metadata = metadata.get('phase_metadata', {})
        for phase_name, phase_data in phase_metadata.items():
            if isinstance(phase_data, dict):
                candidates = phase_data.get('candidates', phase_data.get('results', []))
                for doc in candidates:
                    doc_id = doc.get('record', doc).get('global_id', str(hash(str(doc))))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)

        # Try research_data
        if not all_docs:
            research_data = metadata.get('research_data', {})
            all_results = research_data.get('all_results', [])
            for doc in all_results:
                doc_id = doc.get('record', doc).get('global_id', str(hash(str(doc))))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        # Try research_log
        if not all_docs:
            research_log = metadata.get('research_log', {})
            phase_results = research_log.get('phase_results', {})
            for phase_name, phase_data in phase_results.items():
                if isinstance(phase_data, dict):
                    candidates = phase_data.get('candidates', phase_data.get('results', []))
                    for doc in candidates:
                        doc_id = doc.get('record', doc).get('global_id', str(hash(str(doc))))
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            all_docs.append(doc)

        # Try consensus_data
        if not all_docs:
            consensus_data = metadata.get('consensus_data', {})
            final_results = consensus_data.get('final_results', [])
            for doc in final_results:
                doc_id = doc.get('record', doc).get('global_id', str(hash(str(doc))))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        # Try sources/citations at top level (fallback)
        if not all_docs:
            sources = metadata.get('sources', metadata.get('citations', []))
            for doc in sources:
                doc_id = doc.get('global_id', doc.get('regulation_number', str(hash(str(doc)))))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    # Convert source format to document format
                    all_docs.append({
                        'record': doc,
                        'scores': {
                            'final': doc.get('score', 0),
                            'semantic': doc.get('semantic_score', 0),
                            'keyword': doc.get('keyword_score', 0),
                            'kg': doc.get('kg_score', 0),
                            'authority': doc.get('authority_score', 0),
                            'temporal': doc.get('temporal_score', 0)
                        }
                    })

        return all_docs

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
                    # Extract ALL metadata from the complete chunk
                    thinking_content = chunk.get('thinking', '')
                    phase_metadata = chunk.get('phase_metadata', chunk.get('all_retrieved_metadata', {}))
                    research_log = chunk.get('research_log', {})
                    sources = chunk.get('sources', [])
                    citations = chunk.get('citations', [])
                    # Also include consensus and research data
                    final_metadata['consensus_data'] = chunk.get('consensus_data', {})
                    final_metadata['research_data'] = chunk.get('research_data', {})

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
                    all_docs = self._extract_all_documents(result['metadata'])
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
    args = parser.parse_args()

    tester = CompleteOutputTester()

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
