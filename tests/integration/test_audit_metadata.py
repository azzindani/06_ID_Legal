"""
Comprehensive Audit & Metadata Test
Shows ALL internal details, scores, and calculations for transparency

This test is designed for:
1. Auditing the RAG system behavior
2. Verifying scoring calculations
3. Understanding ranking decisions
4. UI development reference
5. Debugging and transparency

Run with:
    python tests/integration/test_audit_metadata.py

Output includes:
- All search scores (semantic, keyword, KG, authority, temporal, completeness)
- Weight calculations and final scoring
- Complete metadata for every document
- Phase-by-phase research results
- Persona contributions
- Timing breakdowns
- Citation details
- Everything needed for audit and verification
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
from pipeline.rag_pipeline import RAGPipeline


class AuditTester:
    """Comprehensive audit test showing all internal details"""

    def __init__(self):
        initialize_logging()
        self.logger = get_logger("AuditTest")
        self.pipeline: Optional[RAGPipeline] = None

    def print_section(self, title: str, char: str = "="):
        """Print formatted section header"""
        width = 100
        print("\n" + char * width)
        print(f" {title} ".center(width, char))
        print(char * width)

    def print_subsection(self, title: str):
        """Print formatted subsection"""
        print(f"\n{'─' * 100}")
        print(f"  {title}")
        print('─' * 100)

    def format_score(self, score: float) -> str:
        """Format score with visual bar"""
        percentage = int(score * 100)
        bar_length = 20
        filled = int(bar_length * score)
        bar = '█' * filled + '░' * (bar_length - filled)
        return f"{score:.4f} ({percentage:3d}%) {bar}"

    def display_scoring_breakdown(self, result: Dict[str, Any]):
        """Display detailed scoring breakdown for a single result"""
        scores = result.get('scores', {})

        print(f"\n    {'Score Type':<25} {'Value':<40}")
        print(f"    {'-' * 70}")

        # Individual scores
        print(f"    {'Semantic Match':<25} {self.format_score(scores.get('semantic', 0.0))}")
        print(f"    {'Keyword Precision':<25} {self.format_score(scores.get('keyword', 0.0))}")
        print(f"    {'Knowledge Graph':<25} {self.format_score(scores.get('kg', 0.0))}")
        print(f"    {'Authority Hierarchy':<25} {self.format_score(scores.get('authority', 0.0))}")
        print(f"    {'Temporal Relevance':<25} {self.format_score(scores.get('temporal', 0.0))}")
        print(f"    {'Legal Completeness':<25} {self.format_score(scores.get('completeness', 0.0))}")
        print(f"    {'-' * 70}")
        print(f"    {'FINAL SCORE':<25} {self.format_score(scores.get('final', 0.0))}")

    def display_document_metadata(self, record: Dict[str, Any]):
        """Display complete document metadata"""
        print(f"\n    {'Field':<30} {'Value'}")
        print(f"    {'-' * 90}")

        # Core identification
        print(f"    {'Global ID':<30} {record.get('global_id', 'N/A')}")
        print(f"    {'Regulation Type':<30} {record.get('regulation_type', 'N/A')}")
        print(f"    {'Regulation Number':<30} {record.get('regulation_number', 'N/A')}")
        print(f"    {'Year':<30} {record.get('year', 'N/A')}")
        print(f"    {'About':<30} {record.get('about', 'N/A')[:50]}...")

        # Legal classification
        print(f"\n    {'Legal Domain':<30} {record.get('legal_domain', 'N/A')}")
        print(f"    {'Sub-domain':<30} {record.get('sub_domain', 'N/A')}")
        print(f"    {'Hierarchy Level':<30} {record.get('hierarchy_level', 'N/A')}")

        # Knowledge graph scores
        print(f"\n    {'KG Authority Score':<30} {record.get('kg_authority_score', 0.0):.4f}")
        print(f"    {'KG Temporal Score':<30} {record.get('kg_temporal_score', 0.0):.4f}")
        print(f"    {'KG Completeness Score':<30} {record.get('kg_completeness_score', 0.0):.4f}")

        # Entities and relationships
        entities = record.get('kg_entities', [])
        if entities:
            print(f"\n    {'Entities Extracted':<30} {len(entities)}")
            for i, entity in enumerate(entities[:5], 1):  # Show first 5
                print(f"      {i}. {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')})")
            if len(entities) > 5:
                print(f"      ... and {len(entities) - 5} more")

        relationships = record.get('kg_relationships', [])
        if relationships:
            print(f"\n    {'Relationships Found':<30} {len(relationships)}")
            for i, rel in enumerate(relationships[:3], 1):  # Show first 3
                print(f"      {i}. {rel.get('source', 'N/A')} -> "
                      f"{rel.get('type', 'N/A')} -> {rel.get('target', 'N/A')}")
            if len(relationships) > 3:
                print(f"      ... and {len(relationships) - 3} more")

        # Document references
        references = record.get('references', [])
        if references:
            print(f"\n    {'Document References':<30} {len(references)}")
            for i, ref in enumerate(references[:3], 1):  # Show first 3
                print(f"      {i}. {ref}")
            if len(references) > 3:
                print(f"      ... and {len(references) - 3} more")

    def display_phase_research(self, phase_metadata: Dict[str, Any]):
        """Display research phase details"""
        if not phase_metadata:
            print("\n    No phase metadata available")
            return

        # Group by phase
        phases = {}
        for key, data in phase_metadata.items():
            phase = data.get('phase', 'unknown')
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(data)

        for phase_name, phase_data_list in phases.items():
            print(f"\n    Phase: {phase_name}")
            print(f"    {'─' * 90}")

            for data in phase_data_list:
                researcher = data.get('researcher_name', data.get('researcher', 'Unknown'))
                candidates = data.get('candidates', [])
                confidence = data.get('confidence', 0.0)

                print(f"\n      Researcher: {researcher}")
                print(f"      Confidence: {confidence:.4f}")
                print(f"      Candidates Found: {len(candidates)}")

                if candidates:
                    print(f"\n      Top Candidates:")
                    for i, candidate in enumerate(candidates[:3], 1):
                        if isinstance(candidate, dict):
                            score = candidate.get('scores', {}).get('final', 0.0)
                            metadata = candidate.get('metadata', {})
                            reg_type = metadata.get('regulation_type', 'N/A')
                            reg_num = metadata.get('regulation_number', 'N/A')
                            print(f"        {i}. {reg_type} {reg_num} "
                                  f"(score: {score:.4f})")

    def display_timing_breakdown(self, metadata: Dict[str, Any]):
        """Display detailed timing information"""
        retrieval_time = metadata.get('retrieval_time', 0.0)
        generation_time = metadata.get('generation_time', 0.0)
        total_time = metadata.get('total_time', 0.0)

        print(f"\n    {'Operation':<30} {'Time (s)':<15} {'Percentage'}")
        print(f"    {'-' * 70}")

        if total_time > 0:
            retrieval_pct = (retrieval_time / total_time) * 100
            generation_pct = (generation_time / total_time) * 100
            other_pct = 100 - retrieval_pct - generation_pct

            print(f"    {'Document Retrieval':<30} {retrieval_time:>10.3f}     {retrieval_pct:>6.2f}%")
            print(f"    {'Answer Generation':<30} {generation_time:>10.3f}     {generation_pct:>6.2f}%")
            print(f"    {'Other (orchestration)':<30} {total_time - retrieval_time - generation_time:>10.3f}     {other_pct:>6.2f}%")
            print(f"    {'-' * 70}")
            print(f"    {'TOTAL TIME':<30} {total_time:>10.3f}     100.00%")
        else:
            print(f"    {'Document Retrieval':<30} {retrieval_time:>10.3f}")
            print(f"    {'Answer Generation':<30} {generation_time:>10.3f}")
            print(f"    {'TOTAL TIME':<30} {total_time:>10.3f}")

        # Token information
        tokens = metadata.get('tokens_generated', 0)
        if tokens > 0 and generation_time > 0:
            tokens_per_sec = tokens / generation_time
            print(f"\n    {'Tokens Generated':<30} {tokens:,}")
            print(f"    {'Tokens per Second':<30} {tokens_per_sec:,.2f}")

    def display_citations(self, citations: List[Dict[str, Any]]):
        """Display citation details"""
        if not citations:
            print("\n    No citations found")
            return

        print(f"\n    Total Citations: {len(citations)}")
        print(f"    {'-' * 90}")

        for i, citation in enumerate(citations, 1):
            reg_type = citation.get('regulation_type', 'N/A')
            reg_num = citation.get('regulation_number', 'N/A')
            year = citation.get('year', 'N/A')
            about = citation.get('about', 'N/A')

            print(f"\n    [{i}] {reg_type} No. {reg_num} Tahun {year}")
            print(f"        Tentang: {about}")

    def run_comprehensive_audit(self, query: str) -> bool:
        """Run comprehensive audit with full detail display"""

        self.print_section("COMPREHENSIVE RAG AUDIT TEST", "═")

        print(f"\n📋 Test Configuration")
        print(f"    Query: {query}")
        print(f"    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Mode: Full Audit (All Details)")

        try:
            # Initialize pipeline
            self.print_subsection("1. System Initialization")
            print("\n    Initializing RAG pipeline...")

            self.pipeline = RAGPipeline()
            if not self.pipeline.initialize():
                self.logger.error("❌ Pipeline initialization failed")
                return False

            print("    ✅ Pipeline initialized successfully")

            # Execute query
            self.print_subsection("2. Query Execution")
            print(f"\n    Executing query: \"{query}\"")
            print("    Please wait...")

            start_time = time.time()
            result = self.pipeline.query(query, stream=False)
            execution_time = time.time() - start_time

            if not result.get('success'):
                self.logger.error("❌ Query failed", {"error": result.get('error')})
                return False

            print(f"    ✅ Query completed in {execution_time:.2f}s")

            # Display answer
            self.print_section("3. GENERATED ANSWER")
            answer = result.get('answer', '')
            print(f"\n{answer}\n")

            # Display thinking (if available)
            thinking = result.get('thinking', '')
            if thinking:
                self.print_section("4. THINKING PROCESS (Internal Reasoning)")
                print(f"\n{thinking}\n")

            # Display timing breakdown
            metadata = result.get('metadata', {})
            self.print_section("5. TIMING BREAKDOWN")
            self.display_timing_breakdown(metadata)

            # Display query analysis
            self.print_section("6. QUERY ANALYSIS")
            print(f"\n    Query Type: {metadata.get('query_type', 'N/A')}")
            print(f"    Results Count: {metadata.get('results_count', 0)}")
            print(f"    From Cache: {metadata.get('from_cache', False)}")

            # Display RAG metadata (research phases)
            rag_metadata = metadata.get('rag_metadata', {})
            research_phases = rag_metadata.get('research_phases')

            if not research_phases:
                # Try getting from metadata directly
                research_phases = metadata.get('research_phases', {})

            if research_phases:
                self.print_section("7. RESEARCH PHASES (Multi-Researcher Analysis)")
                self.display_phase_research(research_phases)

            # Display all source documents with detailed scoring
            sources = result.get('sources', [])
            if sources:
                self.print_section("8. SOURCE DOCUMENTS (Detailed Scoring)")
                print(f"\n    Total Sources: {len(sources)}")

                for i, source in enumerate(sources, 1):
                    self.print_subsection(f"Source #{i}")

                    # Document metadata
                    print(f"\n    📄 Document Details:")
                    if 'record' in source:
                        self.display_document_metadata(source['record'])
                    elif isinstance(source, dict):
                        # If source is already flattened
                        self.display_document_metadata(source)

                    # Scoring breakdown
                    print(f"\n    📊 Scoring Breakdown:")
                    if 'scores' in source:
                        self.display_scoring_breakdown(source)

                    # Show excerpt if available
                    excerpt = source.get('excerpt', source.get('content', ''))
                    if excerpt:
                        print(f"\n    📝 Content Excerpt:")
                        excerpt_preview = excerpt[:300] + "..." if len(excerpt) > 300 else excerpt
                        print(f"    {excerpt_preview}\n")

            # Display citations
            citations = result.get('citations', [])
            if citations:
                self.print_section("9. CITATIONS")
                self.display_citations(citations)

            # Display complete metadata dump (JSON format for programmatic use)
            self.print_section("10. COMPLETE METADATA (JSON)")
            print("\n    Full metadata in JSON format (for programmatic access):\n")

            # Create clean metadata dict
            audit_data = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
                'answer': answer,
                'thinking': thinking,
                'metadata': metadata,
                'sources_count': len(sources),
                'citations_count': len(citations),
                'citations': citations,
                'timing': {
                    'retrieval_time': metadata.get('retrieval_time', 0),
                    'generation_time': metadata.get('generation_time', 0),
                    'total_time': metadata.get('total_time', 0)
                }
            }

            print(json.dumps(audit_data, indent=2, ensure_ascii=False))

            # Summary
            self.print_section("11. AUDIT SUMMARY", "═")
            print(f"\n    ✅ Audit completed successfully")
            print(f"    📊 Total sources analyzed: {len(sources)}")
            print(f"    📚 Total citations: {len(citations)}")
            print(f"    ⏱️  Total execution time: {execution_time:.2f}s")
            print(f"    🎯 Query type: {metadata.get('query_type', 'N/A')}\n")

            return True

        except Exception as e:
            self.logger.error("❌ Audit failed", {"error": str(e)})
            import traceback
            print("\n" + traceback.format_exc())
            return False

        finally:
            if self.pipeline:
                self.logger.info("Shutting down pipeline...")
                self.pipeline.shutdown()

    def run_multiple_queries_audit(self, queries: List[str]) -> bool:
        """Run audit on multiple queries for comparison"""

        self.print_section("MULTI-QUERY COMPARATIVE AUDIT", "═")

        try:
            # Initialize once
            print("\n    Initializing RAG pipeline...")
            self.pipeline = RAGPipeline()
            if not self.pipeline.initialize():
                self.logger.error("❌ Pipeline initialization failed")
                return False
            print("    ✅ Pipeline initialized\n")

            results = []

            for i, query in enumerate(queries, 1):
                self.print_section(f"QUERY {i}/{len(queries)}: {query}", "─")

                start_time = time.time()
                result = self.pipeline.query(query, stream=False)
                exec_time = time.time() - start_time

                if result.get('success'):
                    metadata = result.get('metadata', {})
                    sources = result.get('sources', [])

                    print(f"\n    ✅ Success")
                    print(f"    ⏱️  Time: {exec_time:.2f}s")
                    print(f"    📊 Sources: {len(sources)}")
                    print(f"    🎯 Type: {metadata.get('query_type', 'N/A')}")
                    print(f"\n    Answer Preview:")
                    answer = result.get('answer', '')
                    preview = answer[:200] + "..." if len(answer) > 200 else answer
                    print(f"    {preview}\n")

                    results.append({
                        'query': query,
                        'success': True,
                        'time': exec_time,
                        'sources': len(sources),
                        'query_type': metadata.get('query_type', 'N/A')
                    })
                else:
                    print(f"\n    ❌ Failed: {result.get('error', 'Unknown error')}\n")
                    results.append({
                        'query': query,
                        'success': False,
                        'error': result.get('error')
                    })

            # Comparison summary
            self.print_section("COMPARATIVE SUMMARY", "═")

            print(f"\n    {'Query':<50} {'Time (s)':<12} {'Sources':<10} {'Type'}")
            print(f"    {'-' * 95}")

            for r in results:
                if r.get('success'):
                    query_preview = r['query'][:47] + "..." if len(r['query']) > 50 else r['query']
                    print(f"    {query_preview:<50} {r['time']:>8.2f}     {r['sources']:>7}    {r['query_type']}")
                else:
                    query_preview = r['query'][:47] + "..." if len(r['query']) > 50 else r['query']
                    print(f"    {query_preview:<50} {'FAILED':<12}")

            print(f"    {'-' * 95}")

            successful = [r for r in results if r.get('success')]
            if successful:
                avg_time = sum(r['time'] for r in successful) / len(successful)
                avg_sources = sum(r['sources'] for r in successful) / len(successful)
                print(f"    {'Average (successful)':<50} {avg_time:>8.2f}     {avg_sources:>7.1f}")

            print()

            return True

        except Exception as e:
            self.logger.error("❌ Multi-query audit failed", {"error": str(e)})
            import traceback
            print("\n" + traceback.format_exc())
            return False

        finally:
            if self.pipeline:
                self.logger.info("Shutting down pipeline...")
                self.pipeline.shutdown()


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive audit test with full metadata")
    parser.add_argument('--query', type=str, help='Single query to audit')
    parser.add_argument('--multi', action='store_true', help='Run multiple queries for comparison')
    args = parser.parse_args()

    tester = AuditTester()

    try:
        if args.multi:
            # Multiple queries for comparison
            queries = [
                "Apa sanksi dalam UU ITE?",
                "Jelaskan tentang perlindungan konsumen",
                "Bagaimana prosedur pelaporan pelanggaran data pribadi?"
            ]
            success = tester.run_multiple_queries_audit(queries)
        else:
            # Single query with full detail
            query = args.query or "Jelaskan tentang UU Ketenagakerjaan dan sanksinya"
            success = tester.run_comprehensive_audit(query)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nAudit interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
