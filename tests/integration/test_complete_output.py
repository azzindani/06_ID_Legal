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

        # Legal References - ALL retrieved documents
        lines.append(f"\n## LEGAL REFERENCES (All Retrieved Documents)")
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

                lines.append(f"### {idx}. {reg_type} No. {reg_num}/{year}")
                lines.append(f"   About: {about[:150]}..." if len(about) > 150 else f"   About: {about}")

                # Scores
                final_score = scores.get('final', doc.get('final_score', doc.get('composite_score', 0)))
                semantic_score = scores.get('semantic', doc.get('semantic_score', 0))
                keyword_score = scores.get('keyword', doc.get('keyword_score', 0))
                kg_score = scores.get('kg', doc.get('kg_score', 0))
                authority_score = scores.get('authority', doc.get('authority_score', 0))
                temporal_score = scores.get('temporal', doc.get('temporal_score', 0))

                lines.append(f"   Score: {final_score:.4f}")
                lines.append(f"   - Semantic: {semantic_score:.4f}")
                lines.append(f"   - Keyword: {keyword_score:.4f}")
                lines.append(f"   - KG Score: {kg_score:.4f}")
                lines.append(f"   - Authority: {authority_score:.4f}")
                lines.append(f"   - Temporal: {temporal_score:.4f}")

                # KG Metadata
                kg_domain = record.get('kg_primary_domain', record.get('primary_domain', ''))
                kg_hierarchy = record.get('kg_hierarchy_level', record.get('hierarchy_level', 0))
                if kg_domain or kg_hierarchy:
                    lines.append(f"   Domain: {kg_domain} | Hierarchy Level: {kg_hierarchy}")

                # Team Consensus
                if doc.get('team_consensus'):
                    agreement = doc.get('researcher_agreement', 0)
                    personas = doc.get('personas_agreed', [])
                    lines.append(f"   Team Consensus: Yes ({agreement} researchers agreed)")
                    if personas:
                        lines.append(f"   Personas: {', '.join(personas[:3])}")

                # Content snippet
                content = record.get('content', '')
                if content:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    lines.append(f"   Content: {snippet}")

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
                    lines.append(f"   {idx}. {reg_type} No. {reg_num}/{year}: {about[:80]}...")
            else:
                lines.append("No documents retrieved")

        # Research Process Details
        lines.append(f"\n## RESEARCH PROCESS DETAILS")
        lines.append("-" * 80)

        research_log = metadata.get('research_log', metadata.get('research_data', {}))

        if research_log:
            # Team members
            team_members = research_log.get('team_members', [])
            lines.append(f"Team Members: {len(team_members)}")
            for member in team_members:
                if isinstance(member, dict):
                    lines.append(f"   - {member.get('name', member.get('persona', 'Unknown'))}")
                else:
                    lines.append(f"   - {member}")

            # Total documents
            total_docs = research_log.get('total_documents_retrieved', 0)
            lines.append(f"Total Documents Retrieved: {total_docs}")

            # Phase results
            phase_results = research_log.get('phase_results', research_log.get('phases', {}))
            if phase_results:
                lines.append(f"\nPhases Executed: {len(phase_results)}")

                for phase_name, phase_data in phase_results.items():
                    if isinstance(phase_data, dict):
                        researcher = phase_data.get('researcher_name', phase_data.get('researcher', 'Unknown'))
                        candidates = phase_data.get('candidates', phase_data.get('results', []))
                        confidence = phase_data.get('confidence', 0)

                        lines.append(f"\n   Phase: {phase_name}")
                        lines.append(f"   Researcher: {researcher}")
                        lines.append(f"   Documents: {len(candidates)}")
                        lines.append(f"   Confidence: {confidence:.2%}")
        else:
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

            # Display complete formatted output
            complete_output = self.format_complete_output(
                query=query,
                answer=full_answer,
                metadata=final_metadata,
                streaming_stats=streaming_stats
            )
            print(complete_output)

            result['success'] = True
            result['answer'] = full_answer
            result['metadata'] = final_metadata
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
