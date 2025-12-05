#!/usr/bin/env python3
"""
Audit Metadata Test - Complete Transparency into RAG Scoring and Decisions

This test provides FULL VISIBILITY into the RAG system's decision-making process:

1. ALL 6 SCORES per document:
   - Semantic Score: How well the document matches semantically (embedding similarity)
   - Keyword Score: TF-IDF keyword matching precision
   - KG Score: Knowledge graph relevance (entities, relationships)
   - Authority Score: Legal hierarchy weight (UU > PP > Peraturan Menteri)
   - Temporal Score: Recency and time relevance
   - Completeness Score: Document completeness and structure

2. RESEARCH PHASES:
   - Phase-by-phase breakdown showing which researchers found what
   - Confidence scores per phase
   - Candidate count progression

3. CONSENSUS DATA:
   - Which personas agreed on which documents
   - Voting ratios and agreement levels
   - Devil's advocate flags

4. COMPLETE METADATA:
   - Entity extraction results
   - Cross-references and citations
   - Regulation metadata (type, number, year, about)

Run with:
    python tests/integration/test_audit_metadata.py
    python tests/integration/test_audit_metadata.py --query "Your custom question"
    python tests/integration/test_audit_metadata.py --multi  # Compare multiple queries
    python tests/integration/test_audit_metadata.py --json   # Export full JSON dump
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging


def create_score_bar(score: float, width: int = 20, filled_char: str = "█", empty_char: str = "░") -> str:
    """Create a visual progress bar for a score (0-1)"""
    score = max(0, min(1, score))
    filled = int(score * width)
    empty = width - filled
    return filled_char * filled + empty_char * empty


def format_percentage(score: float) -> str:
    """Format score as percentage"""
    return f"{score * 100:.1f}%"


class AuditMetadataTester:
    """
    Complete audit and transparency testing for the RAG system
    """

    def __init__(self):
        initialize_logging(level="INFO")
        self.logger = get_logger("AuditTest")
        self.pipeline = None

    def initialize_pipeline(self) -> bool:
        """Initialize the RAG pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("AUDIT METADATA TEST - FULL TRANSPARENCY")
        self.logger.info("=" * 80)

        try:
            from pipeline import RAGPipeline

            self.logger.info("Creating RAG Pipeline...")
            self.pipeline = RAGPipeline()

            self.logger.info("Initializing all components...")
            start_time = time.time()

            if not self.pipeline.initialize():
                self.logger.error("Pipeline initialization failed")
                return False

            elapsed = time.time() - start_time
            self.logger.success(f"Pipeline initialized in {elapsed:.1f}s")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_query_with_full_audit(self, query: str) -> Dict[str, Any]:
        """
        Run a query and capture ALL metadata for auditing
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"QUERY: {query}")
        self.logger.info("=" * 80)

        start_time = time.time()
        result = self.pipeline.query(query, stream=False)
        total_time = time.time() - start_time

        return {
            'query': query,
            'result': result,
            'total_time': total_time
        }

    def display_document_scores(self, result: Dict[str, Any]) -> None:
        """
        Display ALL 6 SCORES for each retrieved document
        With visual score bars and detailed breakdown
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DOCUMENT SCORES - ALL 6 SCORING DIMENSIONS")
        self.logger.info("=" * 80)

        # Get sources from different possible locations
        sources = (
            result.get('sources', []) or
            result.get('citations', []) or
            []
        )

        # Also try to get from research_data or consensus_data for detailed scores
        consensus_data = result.get('consensus_data', {})
        validated_results = consensus_data.get('validated_results', [])

        if validated_results:
            self.logger.info(f"\n{len(validated_results)} documents with full scoring:")
            self.logger.info("-" * 80)

            for i, doc in enumerate(validated_results[:10], 1):
                record = doc.get('record', {})
                scores = doc.get('aggregated_scores', {})

                # Document header
                reg_type = record.get('regulation_type', 'N/A')
                reg_num = record.get('regulation_number', 'N/A')
                year = record.get('year', 'N/A')
                about = record.get('about', '')[:60]

                self.logger.info(f"\n{i}. {reg_type} No. {reg_num}/{year}")
                self.logger.info(f"   About: {about}...")

                # Get all available scores
                semantic = scores.get('semantic', doc.get('scores', {}).get('semantic', 0))
                keyword = scores.get('keyword', doc.get('scores', {}).get('keyword', 0))
                kg = scores.get('kg', doc.get('scores', {}).get('kg', 0))
                authority = scores.get('authority', record.get('kg_authority_score', 0))
                temporal = scores.get('temporal', record.get('kg_temporal_score', 0))
                completeness = scores.get('completeness', record.get('kg_completeness_score', 0))
                final = scores.get('final', doc.get('consensus_score', 0))

                # Display scores with visual bars
                self.logger.info(f"\n   SCORES (with visual bars):")
                self.logger.info(f"   ├─ Semantic:    {create_score_bar(semantic)} {format_percentage(semantic)}")
                self.logger.info(f"   ├─ Keyword:     {create_score_bar(keyword)} {format_percentage(keyword)}")
                self.logger.info(f"   ├─ KG:          {create_score_bar(kg)} {format_percentage(kg)}")
                self.logger.info(f"   ├─ Authority:   {create_score_bar(authority)} {format_percentage(authority)}")
                self.logger.info(f"   ├─ Temporal:    {create_score_bar(temporal)} {format_percentage(temporal)}")
                self.logger.info(f"   ├─ Completeness:{create_score_bar(completeness)} {format_percentage(completeness)}")
                self.logger.info(f"   └─ FINAL:       {create_score_bar(final)} {format_percentage(final)}")

                # Consensus metadata
                voting_ratio = doc.get('voting_ratio', 0)
                personas_agreed = doc.get('personas_agreed', [])

                self.logger.info(f"\n   CONSENSUS:")
                self.logger.info(f"   ├─ Voting Ratio: {format_percentage(voting_ratio)}")
                self.logger.info(f"   └─ Personas: {', '.join(personas_agreed) if personas_agreed else 'N/A'}")

                # KG Metadata
                self.logger.info(f"\n   KG METADATA:")
                entity_count = record.get('kg_entity_count', 0)
                cross_ref_count = record.get('kg_cross_ref_count', 0)
                domain = record.get('kg_primary_domain', 'N/A')
                has_obligations = record.get('kg_has_obligations', False)
                has_prohibitions = record.get('kg_has_prohibitions', False)
                has_permissions = record.get('kg_has_permissions', False)

                self.logger.info(f"   ├─ Entities: {entity_count}")
                self.logger.info(f"   ├─ Cross-References: {cross_ref_count}")
                self.logger.info(f"   ├─ Domain: {domain}")
                self.logger.info(f"   ├─ Has Obligations: {has_obligations}")
                self.logger.info(f"   ├─ Has Prohibitions: {has_prohibitions}")
                self.logger.info(f"   └─ Has Permissions: {has_permissions}")

        elif sources:
            # Fallback to simpler sources display
            self.logger.info(f"\n{len(sources)} sources (simplified view):")
            for i, source in enumerate(sources[:10], 1):
                reg_type = source.get('regulation_type', 'N/A')
                reg_num = source.get('regulation_number', 'N/A')
                year = source.get('year', 'N/A')
                score = source.get('final_score', source.get('score', 0))

                self.logger.info(f"  {i}. {reg_type} No. {reg_num}/{year} (score: {format_percentage(score)})")
        else:
            self.logger.warning("No source documents found in result")

    def display_research_phases(self, result: Dict[str, Any]) -> None:
        """
        Display phase-by-phase research breakdown
        Shows which researchers found what in each phase
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RESEARCH PHASES - WHO FOUND WHAT")
        self.logger.info("=" * 80)

        phase_metadata = result.get('phase_metadata', {}) or result.get('all_retrieved_metadata', {})
        research_data = result.get('research_data', {})

        if phase_metadata:
            # Group by phase
            phases = {}
            for key, data in phase_metadata.items():
                phase = data.get('phase', 'unknown')
                if phase not in phases:
                    phases[phase] = []
                phases[phase].append(data)

            for phase_name, phase_entries in phases.items():
                self.logger.info(f"\n{'Phase: ' + phase_name:-^60}")

                for entry in phase_entries:
                    researcher = entry.get('researcher', 'unknown')
                    researcher_name = entry.get('researcher_name', researcher)
                    candidates = entry.get('candidates', entry.get('results', []))
                    confidence = entry.get('confidence', 1.0)

                    self.logger.info(f"\n  Researcher: {researcher_name}")
                    self.logger.info(f"  Confidence: {format_percentage(confidence)}")
                    self.logger.info(f"  Candidates Found: {len(candidates)}")

                    if candidates:
                        self.logger.info("  Top 3 Findings:")
                        for i, c in enumerate(candidates[:3], 1):
                            record = c.get('record', {})
                            reg_type = record.get('regulation_type', 'N/A')
                            reg_num = record.get('regulation_number', 'N/A')
                            score = c.get('composite_score', c.get('final', 0))
                            self.logger.info(f"    {i}. {reg_type} {reg_num} (score: {format_percentage(score)})")

        elif research_data:
            # Direct research_data display
            phase_results = research_data.get('phase_results', {})
            rounds = research_data.get('rounds', [])

            self.logger.info(f"\nTotal Rounds: {len(rounds)}")

            for phase_name, results in phase_results.items():
                self.logger.info(f"\n{'Phase: ' + phase_name:-^60}")
                self.logger.info(f"  Results: {len(results)}")

                if results:
                    # Group by persona
                    by_persona = {}
                    for r in results:
                        persona = r.get('metadata', {}).get('persona', 'unknown')
                        if persona not in by_persona:
                            by_persona[persona] = []
                        by_persona[persona].append(r)

                    for persona, persona_results in by_persona.items():
                        self.logger.info(f"\n  {persona}: {len(persona_results)} candidates")
        else:
            self.logger.warning("No phase metadata available")

    def display_consensus_analysis(self, result: Dict[str, Any]) -> None:
        """
        Display consensus building analysis
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CONSENSUS ANALYSIS - TEAM AGREEMENT")
        self.logger.info("=" * 80)

        consensus_data = result.get('consensus_data', {})

        if consensus_data:
            agreement_level = consensus_data.get('agreement_level', 0)
            validated_count = len(consensus_data.get('validated_results', []))
            cross_validation = consensus_data.get('cross_validation_passed', [])
            devil_flags = consensus_data.get('devil_advocate_flags', [])

            self.logger.info(f"\n  Overall Agreement Level: {format_percentage(agreement_level)}")
            self.logger.info(f"  Validated Results: {validated_count}")
            self.logger.info(f"  Cross-Validation Passed: {len(cross_validation)}")
            self.logger.info(f"  Devil's Advocate Flags: {len(devil_flags)}")

            # Consensus scores distribution
            scores = consensus_data.get('consensus_scores', {})
            if scores:
                score_values = list(scores.values())
                self.logger.info(f"\n  Consensus Score Distribution:")
                self.logger.info(f"    Min:  {format_percentage(min(score_values))}")
                self.logger.info(f"    Max:  {format_percentage(max(score_values))}")
                self.logger.info(f"    Mean: {format_percentage(sum(score_values) / len(score_values))}")

            # Devil's advocate analysis
            if devil_flags:
                self.logger.info(f"\n  Devil's Advocate Concerns:")
                for flag in devil_flags[:5]:
                    self.logger.info(f"    - {flag}")
        else:
            self.logger.warning("No consensus data available")

    def display_timing_breakdown(self, result: Dict[str, Any], total_time: float) -> None:
        """
        Display timing breakdown
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TIMING BREAKDOWN")
        self.logger.info("=" * 80)

        metadata = result.get('metadata', {})

        retrieval_time = metadata.get('retrieval_time', 0)
        generation_time = metadata.get('generation_time', 0)
        other_time = total_time - retrieval_time - generation_time

        self.logger.info(f"\n  Total Time:      {total_time:.2f}s")
        self.logger.info(f"  Retrieval Time:  {retrieval_time:.2f}s ({retrieval_time/total_time*100:.0f}%)")
        self.logger.info(f"  Generation Time: {generation_time:.2f}s ({generation_time/total_time*100:.0f}%)")
        self.logger.info(f"  Other:           {other_time:.2f}s ({other_time/total_time*100:.0f}%)")

        # Tokens if available
        tokens = metadata.get('tokens_generated', 0)
        if tokens and generation_time > 0:
            self.logger.info(f"\n  Tokens Generated: {tokens}")
            self.logger.info(f"  Tokens/Second:    {tokens/generation_time:.1f}")

    def display_query_analysis(self, result: Dict[str, Any]) -> None:
        """
        Display query analysis results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("QUERY ANALYSIS")
        self.logger.info("=" * 80)

        metadata = result.get('metadata', {})
        rag_metadata = metadata.get('rag_metadata', {})
        query_analysis = rag_metadata.get('query_analysis', {})

        if query_analysis:
            query_type = query_analysis.get('query_type', 'unknown')
            strategy = query_analysis.get('strategy', 'unknown')
            confidence = query_analysis.get('confidence', 0)
            entities = query_analysis.get('entities', [])
            key_phrases = query_analysis.get('key_phrases', [])
            regulation_refs = query_analysis.get('regulation_references', [])

            self.logger.info(f"\n  Query Type: {query_type}")
            self.logger.info(f"  Strategy: {strategy}")
            self.logger.info(f"  Confidence: {format_percentage(confidence)}")

            if entities:
                self.logger.info(f"\n  Extracted Entities:")
                for entity in entities[:10]:
                    self.logger.info(f"    - {entity}")

            if key_phrases:
                self.logger.info(f"\n  Key Phrases:")
                for phrase in key_phrases[:10]:
                    self.logger.info(f"    - {phrase}")

            if regulation_refs:
                self.logger.info(f"\n  Regulation References:")
                for ref in regulation_refs[:5]:
                    ref_type = ref.get('type', 'N/A')
                    ref_num = ref.get('number', 'N/A')
                    ref_year = ref.get('year', 'N/A')
                    ref_conf = ref.get('confidence', 0)
                    self.logger.info(f"    - {ref_type} No. {ref_num}/{ref_year} (conf: {format_percentage(ref_conf)})")
        else:
            self.logger.info(f"  Query Type: {metadata.get('query_type', 'unknown')}")

    def display_full_json_dump(self, result: Dict[str, Any], query: str) -> None:
        """
        Display full JSON dump of all metadata
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FULL JSON METADATA DUMP")
        self.logger.info("=" * 80)

        # Create sanitized version (remove large content)
        def sanitize_for_display(obj, max_str_len=200):
            if isinstance(obj, dict):
                return {k: sanitize_for_display(v, max_str_len) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_display(item, max_str_len) for item in obj[:10]]  # Limit list size
            elif isinstance(obj, str):
                if len(obj) > max_str_len:
                    return obj[:max_str_len] + "..."
                return obj
            return obj

        # Create export data
        export_data = {
            'query': query,
            'success': result.get('success', False),
            'answer_length': len(result.get('answer', '')),
            'metadata': sanitize_for_display(result.get('metadata', {})),
            'consensus_data': sanitize_for_display(result.get('consensus_data', {})),
            'phase_count': len(result.get('phase_metadata', {})),
            'source_count': len(result.get('sources', []) or result.get('citations', []))
        }

        # Pretty print
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"\n{json_str}")

    def run_full_audit(self, query: str, show_json: bool = False) -> bool:
        """
        Run complete audit for a single query
        """
        audit_data = self.run_query_with_full_audit(query)
        result = audit_data['result']
        total_time = audit_data['total_time']

        # Display answer first
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANSWER")
        self.logger.info("=" * 80)
        answer = result.get('answer', '')
        self.logger.info(f"\n{answer[:1000]}...")
        if len(answer) > 1000:
            self.logger.info(f"\n[Answer truncated - total {len(answer)} characters]")

        # Display all audit sections
        self.display_query_analysis(result)
        self.display_document_scores(result)
        self.display_research_phases(result)
        self.display_consensus_analysis(result)
        self.display_timing_breakdown(result, total_time)

        if show_json:
            self.display_full_json_dump(result, query)

        return result.get('success', False)

    def run_multi_query_comparison(self, queries: List[str] = None) -> bool:
        """
        Compare audit results across multiple queries
        """
        queries = queries or [
            "Apa sanksi pelanggaran UU ITE?",
            "Bagaimana prosedur perizinan usaha?",
            "Jelaskan tentang perlindungan konsumen"
        ]

        self.logger.info("\n" + "=" * 80)
        self.logger.info("MULTI-QUERY COMPARISON")
        self.logger.info("=" * 80)

        results = []

        for query in queries:
            self.logger.info(f"\n{'Query: ' + query[:50]:-^60}")
            audit_data = self.run_query_with_full_audit(query)
            results.append(audit_data)

        # Comparison summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPARISON SUMMARY")
        self.logger.info("=" * 80)

        self.logger.info("\n{:<40} {:>10} {:>10} {:>10}".format(
            "Query", "Time", "Sources", "Agreement"
        ))
        self.logger.info("-" * 70)

        for audit in results:
            query = audit['query'][:35] + "..." if len(audit['query']) > 35 else audit['query']
            time_taken = audit['total_time']
            result = audit['result']

            sources = len(result.get('consensus_data', {}).get('validated_results', []))
            agreement = result.get('consensus_data', {}).get('agreement_level', 0)

            self.logger.info("{:<40} {:>8.2f}s {:>10} {:>9.1f}%".format(
                query, time_taken, sources, agreement * 100
            ))

        return True

    def shutdown(self):
        """Clean shutdown"""
        if self.pipeline:
            self.logger.info("Shutting down pipeline...")
            self.pipeline.shutdown()
            self.logger.success("Shutdown complete")

    def run_all_tests(self, query: str = None, show_json: bool = False) -> bool:
        """Run complete audit test suite"""
        self.logger.info("\n" + "AUDIT METADATA TEST SUITE".center(80))
        self.logger.info("=" * 80)
        self.logger.info("Complete transparency into RAG scoring and decisions")
        self.logger.info("=" * 80)

        # Initialize
        if not self.initialize_pipeline():
            self.logger.error("Cannot proceed without pipeline")
            return False

        success = True

        try:
            query = query or "Apa sanksi pelanggaran dalam UU Perlindungan Data Pribadi?"
            success = self.run_full_audit(query, show_json)

        finally:
            self.shutdown()

        return success


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Audit metadata test - full transparency into RAG scoring"
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        default=None,
        help='Custom query to audit'
    )
    parser.add_argument(
        '--multi',
        action='store_true',
        help='Run multi-query comparison'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Show full JSON dump of metadata'
    )

    args = parser.parse_args()

    tester = AuditMetadataTester()

    try:
        if args.multi:
            if tester.initialize_pipeline():
                success = tester.run_multi_query_comparison()
                tester.shutdown()
            else:
                success = False
        else:
            success = tester.run_all_tests(args.query, args.json)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        tester.shutdown()
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        tester.shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(main())
