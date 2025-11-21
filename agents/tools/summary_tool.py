"""
Summary Tool - Document Summarization

Enables agents to summarize legal documents or topics.
"""

from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.tool_registry import BaseTool


class SummaryTool(BaseTool):
    """
    Summary tool for condensing legal information

    Generates summaries of legal topics or documents.
    """

    def __init__(self, pipeline):
        super().__init__(
            name="summary",
            description="Generate a summary of a legal topic or regulation"
        )
        self.pipeline = pipeline

    def execute(self, topic: str, max_length: int = 200) -> Dict[str, Any]:
        """
        Generate summary of topic

        Args:
            topic: Topic to summarize
            max_length: Maximum summary length in words

        Returns:
            Summary of the topic
        """
        try:
            # Build summary query
            query = f"Berikan ringkasan singkat tentang {topic} dalam hukum Indonesia"

            # Get response
            result = self.pipeline.query(query, stream=False)
            answer = result.get('answer', '')

            # Truncate if needed
            words = answer.split()
            if len(words) > max_length:
                summary = ' '.join(words[:max_length]) + '...'
            else:
                summary = answer

            # Get related regulations
            sources = result.get('metadata', {}).get('sources', [])
            regulations = list(set([
                f"{s.get('regulation_type', '')} {s.get('regulation_number', '')}/{s.get('year', '')}"
                for s in sources[:5]
            ]))

            return {
                "topic": topic,
                "summary": summary,
                "word_count": len(summary.split()),
                "related_regulations": regulations
            }

        except Exception as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["topic"],
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Legal topic to summarize"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum summary length in words",
                    "default": 200
                }
            }
        }
