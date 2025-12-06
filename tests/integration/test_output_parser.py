"""
Complete RAG Output Parser and Validator

This test module provides:
1. Parser for complete RAG output format (JSON export)
2. Validation of output structure and content
3. Extraction of specific components for auditing

Run with:
    python tests/integration/test_output_parser.py [--file <path>]
    python tests/integration/test_output_parser.py --generate  # Generate and parse
"""

import sys
import os
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger_utils import get_logger, initialize_logging


@dataclass
class LegalReference:
    """Represents a legal reference with full metadata"""
    regulation_type: str = ""
    regulation_number: str = ""
    year: str = ""
    about: str = ""
    final_score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    kg_score: float = 0.0
    authority_score: float = 0.0
    temporal_score: float = 0.0
    domain: str = ""
    hierarchy_level: int = 0
    team_consensus: bool = False
    researcher_agreement: int = 0
    personas_agreed: List[str] = field(default_factory=list)
    content_snippet: str = ""


@dataclass
class ResearchPhase:
    """Represents a research phase in the RAG process"""
    phase_name: str = ""
    researcher_name: str = ""
    document_count: int = 0
    confidence: float = 0.0
    candidates: List[Dict] = field(default_factory=list)


@dataclass
class QueryResult:
    """Represents a complete query result"""
    query_num: int = 0
    query: str = ""
    query_type: str = ""
    thinking_process: str = ""
    answer: str = ""
    success: bool = False
    streaming_chunks: int = 0
    streaming_duration: float = 0.0
    legal_references: List[LegalReference] = field(default_factory=list)
    research_phases: List[ResearchPhase] = field(default_factory=list)
    team_members: List[str] = field(default_factory=list)
    total_documents: int = 0
    total_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0


class OutputParser:
    """Parser for complete RAG output"""

    def __init__(self):
        initialize_logging()
        self.logger = get_logger("OutputParser")

    def parse_json_export(self, file_path: str) -> Tuple[bool, List[QueryResult]]:
        """Parse JSON export from test_complete_output.py"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = []
            for result_data in data.get('results', []):
                query_result = self._parse_result(result_data)
                results.append(query_result)

            self.logger.success(f"Parsed {len(results)} query results from {file_path}")
            return True, results

        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return False, []
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {e}")
            return False, []
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return False, []

    def _parse_result(self, result_data: Dict[str, Any]) -> QueryResult:
        """Parse a single query result"""
        qr = QueryResult()

        # Basic info
        qr.query_num = result_data.get('query_num', 0)
        qr.query = result_data.get('query', '')
        qr.success = result_data.get('success', False)
        qr.answer = result_data.get('answer', '')

        # Streaming stats
        stats = result_data.get('streaming_stats', {})
        qr.streaming_chunks = stats.get('chunk_count', 0)
        qr.streaming_duration = stats.get('duration', 0.0)

        # Metadata
        metadata = result_data.get('metadata', {})
        qr.query_type = metadata.get('query_type', 'general')
        qr.thinking_process = metadata.get('thinking', '')
        qr.total_time = metadata.get('total_time', 0.0)
        qr.retrieval_time = metadata.get('retrieval_time', 0.0)
        qr.generation_time = metadata.get('generation_time', qr.streaming_duration)

        # Extract legal references
        qr.legal_references = self._extract_legal_references(metadata)
        qr.total_documents = len(qr.legal_references)

        # Extract research phases
        qr.research_phases, qr.team_members = self._extract_research_info(metadata)

        return qr

    def _extract_legal_references(self, metadata: Dict[str, Any]) -> List[LegalReference]:
        """Extract all legal references from metadata"""
        references = []
        seen_ids = set()

        # Try all possible locations
        sources = [
            ('phase_metadata', self._extract_from_phase_metadata),
            ('research_data', self._extract_from_research_data),
            ('research_log', self._extract_from_research_log),
            ('consensus_data', self._extract_from_consensus_data),
            ('sources', self._extract_from_sources),
        ]

        for source_name, extractor in sources:
            if source_name in metadata and metadata[source_name]:
                new_refs = extractor(metadata[source_name])
                for ref in new_refs:
                    ref_id = f"{ref.regulation_type}-{ref.regulation_number}-{ref.year}"
                    if ref_id not in seen_ids:
                        seen_ids.add(ref_id)
                        references.append(ref)

        return references

    def _extract_from_phase_metadata(self, phase_metadata: Dict) -> List[LegalReference]:
        """Extract references from phase_metadata"""
        refs = []
        for phase_name, phase_data in phase_metadata.items():
            if isinstance(phase_data, dict):
                candidates = phase_data.get('candidates', phase_data.get('results', []))
                refs.extend(self._parse_documents(candidates))
        return refs

    def _extract_from_research_data(self, research_data: Dict) -> List[LegalReference]:
        """Extract references from research_data"""
        all_results = research_data.get('all_results', [])
        return self._parse_documents(all_results)

    def _extract_from_research_log(self, research_log: Dict) -> List[LegalReference]:
        """Extract references from research_log"""
        refs = []
        phase_results = research_log.get('phase_results', {})
        for phase_name, phase_data in phase_results.items():
            if isinstance(phase_data, dict):
                candidates = phase_data.get('candidates', phase_data.get('results', []))
                refs.extend(self._parse_documents(candidates))
        return refs

    def _extract_from_consensus_data(self, consensus_data: Dict) -> List[LegalReference]:
        """Extract references from consensus_data"""
        final_results = consensus_data.get('final_results', [])
        return self._parse_documents(final_results)

    def _extract_from_sources(self, sources: List[Dict]) -> List[LegalReference]:
        """Extract references from sources/citations list"""
        return self._parse_documents(sources)

    def _parse_documents(self, documents: List[Dict]) -> List[LegalReference]:
        """Parse a list of documents into LegalReference objects"""
        refs = []
        for doc in documents:
            ref = LegalReference()

            record = doc.get('record', doc)
            scores = doc.get('scores', {})

            ref.regulation_type = record.get('regulation_type', '')
            ref.regulation_number = str(record.get('regulation_number', ''))
            ref.year = str(record.get('year', ''))
            ref.about = record.get('about', '')

            ref.final_score = scores.get('final', doc.get('final_score', doc.get('composite_score', 0.0)))
            ref.semantic_score = scores.get('semantic', doc.get('semantic_score', 0.0))
            ref.keyword_score = scores.get('keyword', doc.get('keyword_score', 0.0))
            ref.kg_score = scores.get('kg', doc.get('kg_score', 0.0))
            ref.authority_score = scores.get('authority', doc.get('authority_score', 0.0))
            ref.temporal_score = scores.get('temporal', doc.get('temporal_score', 0.0))

            ref.domain = record.get('kg_primary_domain', record.get('primary_domain', ''))
            ref.hierarchy_level = record.get('kg_hierarchy_level', record.get('hierarchy_level', 0))

            ref.team_consensus = doc.get('team_consensus', False)
            ref.researcher_agreement = doc.get('researcher_agreement', 0)
            ref.personas_agreed = doc.get('personas_agreed', [])

            ref.content_snippet = record.get('content', '')[:200]

            refs.append(ref)

        return refs

    def _extract_research_info(self, metadata: Dict) -> Tuple[List[ResearchPhase], List[str]]:
        """Extract research phases and team members"""
        phases = []
        team_members = []

        research_log = metadata.get('research_log', metadata.get('research_data', {}))

        if research_log:
            # Team members
            members = research_log.get('team_members', [])
            for member in members:
                if isinstance(member, dict):
                    team_members.append(member.get('name', member.get('persona', 'Unknown')))
                else:
                    team_members.append(str(member))

            # Phase results
            phase_results = research_log.get('phase_results', research_log.get('phases', {}))
            for phase_name, phase_data in phase_results.items():
                if isinstance(phase_data, dict):
                    phase = ResearchPhase()
                    phase.phase_name = phase_name
                    phase.researcher_name = phase_data.get('researcher_name', phase_data.get('researcher', 'Unknown'))
                    phase.candidates = phase_data.get('candidates', phase_data.get('results', []))
                    phase.document_count = len(phase.candidates)
                    phase.confidence = phase_data.get('confidence', 0.0)
                    phases.append(phase)

        return phases, team_members


class OutputValidator:
    """Validator for parsed output"""

    def __init__(self):
        initialize_logging()
        self.logger = get_logger("OutputValidator")

    def validate_results(self, results: List[QueryResult]) -> Tuple[bool, List[str]]:
        """Validate all query results"""
        issues = []

        for qr in results:
            result_issues = self.validate_single_result(qr)
            issues.extend(result_issues)

        if issues:
            self.logger.warning(f"Validation found {len(issues)} issues")
        else:
            self.logger.success("All validation checks passed")

        return len(issues) == 0, issues

    def validate_single_result(self, qr: QueryResult) -> List[str]:
        """Validate a single query result"""
        issues = []
        prefix = f"Query {qr.query_num}"

        # Required fields
        if not qr.query:
            issues.append(f"{prefix}: Missing query text")

        if qr.success and not qr.answer:
            issues.append(f"{prefix}: Success but no answer")

        # Streaming validation
        if qr.success:
            if qr.streaming_chunks == 0:
                issues.append(f"{prefix}: No streaming chunks recorded")
            if qr.streaming_duration <= 0:
                issues.append(f"{prefix}: Invalid streaming duration")

        # Legal references validation
        for i, ref in enumerate(qr.legal_references):
            ref_issues = self._validate_reference(ref, prefix, i + 1)
            issues.extend(ref_issues)

        return issues

    def _validate_reference(self, ref: LegalReference, prefix: str, idx: int) -> List[str]:
        """Validate a legal reference"""
        issues = []
        ref_prefix = f"{prefix} Ref {idx}"

        # Regulation identification
        if not ref.regulation_type:
            issues.append(f"{ref_prefix}: Missing regulation type")

        # Score validation
        if ref.final_score < 0 or ref.final_score > 1:
            issues.append(f"{ref_prefix}: Invalid final score {ref.final_score}")

        return issues


class OutputReporter:
    """Generate reports from parsed output"""

    def __init__(self):
        initialize_logging()
        self.logger = get_logger("OutputReporter")

    def generate_audit_report(self, results: List[QueryResult], output_path: Optional[str] = None) -> str:
        """Generate an audit report from query results"""
        lines = []

        lines.append("=" * 100)
        lines.append("RAG OUTPUT AUDIT REPORT")
        lines.append("=" * 100)
        lines.append("")

        # Summary
        lines.append("## SUMMARY")
        lines.append("-" * 80)
        total = len(results)
        successful = sum(1 for r in results if r.success)
        total_docs = sum(r.total_documents for r in results)
        avg_chunks = sum(r.streaming_chunks for r in results) / total if total > 0 else 0

        lines.append(f"Total Queries: {total}")
        lines.append(f"Successful: {successful}")
        lines.append(f"Total Documents Retrieved: {total_docs}")
        lines.append(f"Average Streaming Chunks: {avg_chunks:.1f}")
        lines.append("")

        # Per-query details
        for qr in results:
            lines.append(f"## QUERY {qr.query_num}")
            lines.append("-" * 80)
            lines.append(f"Query: {qr.query}")
            lines.append(f"Type: {qr.query_type}")
            lines.append(f"Success: {qr.success}")
            lines.append(f"Streaming: {qr.streaming_chunks} chunks in {qr.streaming_duration:.2f}s")
            lines.append("")

            if qr.answer:
                lines.append("### Answer (first 500 chars)")
                lines.append(qr.answer[:500] + "..." if len(qr.answer) > 500 else qr.answer)
                lines.append("")

            if qr.legal_references:
                lines.append(f"### Legal References ({len(qr.legal_references)} documents)")
                for i, ref in enumerate(qr.legal_references[:10], 1):  # Show first 10
                    lines.append(f"   {i}. {ref.regulation_type} No. {ref.regulation_number}/{ref.year}")
                    lines.append(f"      Score: {ref.final_score:.4f} (Semantic: {ref.semantic_score:.4f}, KG: {ref.kg_score:.4f})")
                    if ref.domain:
                        lines.append(f"      Domain: {ref.domain}")
                    if ref.team_consensus:
                        lines.append(f"      Team Consensus: Yes ({ref.researcher_agreement} researchers)")

                if len(qr.legal_references) > 10:
                    lines.append(f"   ... and {len(qr.legal_references) - 10} more")
                lines.append("")

            if qr.research_phases:
                lines.append(f"### Research Phases ({len(qr.research_phases)} phases)")
                for phase in qr.research_phases:
                    lines.append(f"   - {phase.phase_name}: {phase.researcher_name} ({phase.document_count} docs, {phase.confidence:.1%} confidence)")
                lines.append("")

            if qr.team_members:
                lines.append(f"### Team Members ({len(qr.team_members)})")
                lines.append(f"   {', '.join(qr.team_members)}")
                lines.append("")

            lines.append("")

        lines.append("=" * 100)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Report saved to: {output_path}")

        return report

    def generate_reference_csv(self, results: List[QueryResult], output_path: str) -> bool:
        """Generate CSV of all legal references for analysis"""
        import csv

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Query_Num', 'Regulation_Type', 'Regulation_Number', 'Year', 'About',
                    'Final_Score', 'Semantic_Score', 'Keyword_Score', 'KG_Score',
                    'Authority_Score', 'Temporal_Score', 'Domain', 'Hierarchy_Level',
                    'Team_Consensus', 'Researcher_Agreement'
                ])

                for qr in results:
                    for ref in qr.legal_references:
                        writer.writerow([
                            qr.query_num, ref.regulation_type, ref.regulation_number, ref.year,
                            ref.about[:100], ref.final_score, ref.semantic_score, ref.keyword_score,
                            ref.kg_score, ref.authority_score, ref.temporal_score, ref.domain,
                            ref.hierarchy_level, ref.team_consensus, ref.researcher_agreement
                        ])

            self.logger.success(f"CSV exported to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return False


def run_parser_test(file_path: Optional[str] = None) -> bool:
    """Run complete parser test"""
    initialize_logging()
    logger = get_logger("ParserTest")

    logger.info("=" * 80)
    logger.info("COMPLETE OUTPUT PARSER TEST")
    logger.info("=" * 80)

    parser = OutputParser()
    validator = OutputValidator()
    reporter = OutputReporter()

    # Find or generate input file
    if file_path and os.path.exists(file_path):
        input_file = file_path
    else:
        # Look for existing export files
        test_dir = os.path.dirname(os.path.abspath(__file__))
        export_files = list(Path(test_dir).glob("complete_output_results_*.json"))

        if export_files:
            # Use most recent
            input_file = str(max(export_files, key=lambda p: p.stat().st_mtime))
            logger.info(f"Using existing export: {input_file}")
        else:
            logger.warning("No export file found. Run test_complete_output.py --export first.")
            return False

    # Parse
    logger.info(f"\nParsing: {input_file}")
    success, results = parser.parse_json_export(input_file)

    if not success:
        logger.error("Parsing failed")
        return False

    logger.success(f"Parsed {len(results)} query results")

    # Validate
    logger.info("\nValidating results...")
    valid, issues = validator.validate_results(results)

    if not valid:
        logger.warning(f"Validation issues found: {len(issues)}")
        for issue in issues[:10]:
            logger.warning(f"  - {issue}")

    # Generate report
    logger.info("\nGenerating audit report...")
    report = reporter.generate_audit_report(results)
    print("\n" + report)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PARSER TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Queries Parsed: {len(results)}")
    logger.info(f"Successful Queries: {sum(1 for r in results if r.success)}")
    logger.info(f"Total Documents: {sum(r.total_documents for r in results)}")
    logger.info(f"Validation: {'PASSED' if valid else 'FAILED'}")
    logger.info("=" * 80)

    return valid


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Parse and validate RAG output")
    parser.add_argument('--file', type=str, help='JSON export file to parse')
    parser.add_argument('--generate', action='store_true', help='Run test_complete_output first')
    parser.add_argument('--csv', type=str, help='Export references to CSV')
    parser.add_argument('--report', type=str, help='Save report to file')
    args = parser.parse_args()

    if args.generate:
        # Run complete output test first
        print("Running complete output test with export...")
        from test_complete_output import CompleteOutputTester

        tester = CompleteOutputTester()
        tester.run_all_queries()
        export_file = tester.export_results()

        print(f"\nParsing exported results: {export_file}")
        return run_parser_test(export_file)

    return 0 if run_parser_test(args.file) else 1


if __name__ == "__main__":
    sys.exit(main())
