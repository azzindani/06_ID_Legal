"""
Batch Pipeline - Bulk Query Processing

Processes multiple queries efficiently with parallelization.
"""

from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .rag_pipeline import RAGPipeline
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class BatchPipeline(RAGPipeline):
    """
    Batch processing RAG pipeline

    Optimized for processing multiple queries efficiently.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_workers = self.config.get('max_workers', 4)
        self.batch_size = self.config.get('batch_size', 10)

    def batch_query(
        self,
        questions: List[str],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries

        Args:
            questions: List of questions
            parallel: Whether to process in parallel

        Returns:
            List of results
        """
        if not self._initialized:
            return [{'error': 'Pipeline not initialized'}] * len(questions)

        logger.info(f"Processing batch of {len(questions)} queries")

        if parallel:
            return self._parallel_query(questions)
        else:
            return self._sequential_query(questions)

    def _sequential_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process queries sequentially"""
        results = []

        for i, question in enumerate(questions):
            try:
                result = self.query(question, stream=False)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'question': question
                })

        return results

    def _parallel_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process queries in parallel"""
        results = [None] * len(questions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._safe_query, q): i
                for i, q in enumerate(questions)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    result['batch_index'] = idx
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Query {idx} failed: {e}")
                    results[idx] = {
                        'batch_index': idx,
                        'error': str(e),
                        'question': questions[idx]
                    }

        return results

    def _safe_query(self, question: str) -> Dict[str, Any]:
        """Thread-safe query execution"""
        try:
            return self.query(question, stream=False)
        except Exception as e:
            return {'error': str(e), 'question': question}

    def batch_query_with_context(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process queries with individual contexts

        Args:
            queries: List of {'question': str, 'context': list} dicts

        Returns:
            List of results
        """
        if not self._initialized:
            return [{'error': 'Pipeline not initialized'}] * len(queries)

        results = []

        for i, query_data in enumerate(queries):
            question = query_data.get('question', '')
            context = query_data.get('context')

            try:
                result = self.query(question, conversation_history=context, stream=False)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'question': question
                })

        return results

    def process_file(
        self,
        filepath: str,
        question_column: str = 'question'
    ) -> List[Dict[str, Any]]:
        """
        Process queries from a file

        Args:
            filepath: Path to CSV/JSON file
            question_column: Column name for questions

        Returns:
            List of results
        """
        import json

        questions = []

        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions = [
                        item.get(question_column, item) if isinstance(item, dict) else item
                        for item in data
                    ]
        elif filepath.endswith('.csv'):
            import csv
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                questions = [row[question_column] for row in reader]
        else:
            # Plain text, one question per line
            with open(filepath, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(questions)} questions from {filepath}")

        return self.batch_query(questions)

    def get_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for batch results

        Args:
            results: Batch results

        Returns:
            Summary statistics
        """
        total = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        failed = total - successful

        total_time = sum(
            r.get('metadata', {}).get('total_time', 0)
            for r in results if 'error' not in r
        )

        total_tokens = sum(
            r.get('metadata', {}).get('tokens_generated', 0)
            for r in results if 'error' not in r
        )

        return {
            'total_queries': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'total_time': total_time,
            'total_tokens': total_tokens,
            'avg_time': total_time / successful if successful > 0 else 0,
            'avg_tokens': total_tokens / successful if successful > 0 else 0
        }


def create_batch_pipeline(config: Optional[Dict[str, Any]] = None) -> BatchPipeline:
    """Factory function for batch pipeline"""
    return BatchPipeline(config)
