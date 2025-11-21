"""
Search Tool - Document Search Agent Tool

Enables agents to search legal documents.
"""

from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.tool_registry import BaseTool


class SearchTool(BaseTool):
    """
    Search tool for finding relevant legal documents

    Uses the RAG pipeline's search functionality.
    """

    def __init__(self, pipeline):
        super().__init__(
            name="search",
            description="Search for relevant legal documents based on a query"
        )
        self.pipeline = pipeline

    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Execute document search

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Search results with documents
        """
        try:
            # Use pipeline to search
            result = self.pipeline.query(query, stream=False)

            # Extract sources from metadata
            sources = result.get('metadata', {}).get('sources', [])[:max_results]

            documents = []
            for i, source in enumerate(sources):
                documents.append({
                    "rank": i + 1,
                    "content": source.get('content', '')[:300],
                    "regulation": f"{source.get('regulation_type', '')} {source.get('regulation_number', '')}/{source.get('year', '')}",
                    "about": source.get('about', 'N/A'),
                    "score": source.get('score', 0.0)
                })

            return {
                "query": query,
                "total_results": len(documents),
                "documents": documents
            }

        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for legal documents"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            }
        }
