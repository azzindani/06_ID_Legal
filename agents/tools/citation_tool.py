"""
Citation Tool - Legal Citation Lookup

Enables agents to find specific legal citations.
"""

from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.tool_registry import BaseTool


class CitationTool(BaseTool):
    """
    Citation tool for looking up specific regulations

    Finds documents by regulation type, number, and year.
    """

    def __init__(self, pipeline):
        super().__init__(
            name="citation",
            description="Look up specific legal regulation by type, number, and year"
        )
        self.pipeline = pipeline

    def execute(
        self,
        regulation_type: str = "",
        regulation_number: str = "",
        year: str = ""
    ) -> Dict[str, Any]:
        """
        Look up specific citation

        Args:
            regulation_type: Type (UU, PP, Perpres, etc.)
            regulation_number: Regulation number
            year: Year of regulation

        Returns:
            Citation information
        """
        try:
            # Build search query
            query_parts = []
            if regulation_type:
                query_parts.append(regulation_type)
            if regulation_number:
                query_parts.append(f"Nomor {regulation_number}")
            if year:
                query_parts.append(f"Tahun {year}")

            if not query_parts:
                return {"error": "At least one parameter required"}

            query = " ".join(query_parts)

            # Search for the regulation
            result = self.pipeline.query(query, stream=False)

            # Extract relevant citation info
            sources = result.get('metadata', {}).get('sources', [])

            citations = []
            for source in sources[:3]:
                if (
                    (not regulation_type or regulation_type.lower() in source.get('regulation_type', '').lower()) and
                    (not regulation_number or regulation_number in source.get('regulation_number', '')) and
                    (not year or year in source.get('year', ''))
                ):
                    citations.append({
                        "regulation_type": source.get('regulation_type', 'N/A'),
                        "regulation_number": source.get('regulation_number', 'N/A'),
                        "year": source.get('year', 'N/A'),
                        "about": source.get('about', 'N/A'),
                        "content_preview": source.get('content', '')[:200]
                    })

            return {
                "query": query,
                "found": len(citations) > 0,
                "citations": citations
            }

        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": [],
            "properties": {
                "regulation_type": {
                    "type": "string",
                    "description": "Regulation type (UU, PP, Perpres, Perda, etc.)"
                },
                "regulation_number": {
                    "type": "string",
                    "description": "Regulation number"
                },
                "year": {
                    "type": "string",
                    "description": "Year of regulation"
                }
            }
        }
