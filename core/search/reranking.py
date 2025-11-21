"""
Reranking Module
Final reranking of consensus results using reranker model
"""

import torch
from typing import Dict, Any, List, Optional
from logger_utils import get_logger


class RerankerEngine:
    """
    Reranks final candidates using dedicated reranker model
    Produces final ranked list for LLM generation
    """
    
    def __init__(self, reranker_model, config: Dict[str, Any]):
        self.reranker_model = reranker_model
        self.config = config
        self.logger = get_logger("RerankerEngine")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.final_top_k = config.get('final_top_k', 3)
        
        self.logger.info("RerankerEngine initialized", {
            "device": str(self.device),
            "final_top_k": self.final_top_k
        })
    
    def rerank(
        self,
        query: str,
        consensus_results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rerank consensus results using reranker model
        
        Args:
            query: Original search query
            consensus_results: Results from consensus building
            top_k: Number of top results to return (default: from config)
            
        Returns:
            Reranked results with metadata
        """
        top_k = top_k or self.final_top_k
        
        self.logger.info("Starting reranking", {
            "query": query[:50] + "..." if len(query) > 50 else query,
            "candidates": len(consensus_results),
            "target_top_k": top_k
        })
        
        if not consensus_results:
            self.logger.warning("No results to rerank")
            return {
                'reranked_results': [],
                'metadata': {
                    'total_candidates': 0,
                    'reranked_count': 0
                }
            }
        
        # Prepare candidates for reranking
        candidates = []
        for result in consensus_results:
            record = result['record']
            
            # Create document text for reranking
            doc_text = self._create_document_text(record)
            candidates.append({
                'result': result,
                'text': doc_text,
                'global_id': record['global_id']
            })
        
        self.logger.debug("Candidates prepared", {
            "count": len(candidates)
        })
        
        # Perform reranking
        reranked = self._rerank_with_model(query, candidates, top_k)
        
        self.logger.success("Reranking completed", {
            "reranked_count": len(reranked),
            "top_score": f"{reranked[0]['rerank_score']:.4f}" if reranked else "N/A"
        })
        
        return {
            'reranked_results': reranked,
            'metadata': {
                'total_candidates': len(consensus_results),
                'reranked_count': len(reranked),
                'query': query
            }
        }
    
    def _create_document_text(self, record: Dict[str, Any]) -> str:
        """Create document text for reranking"""
        
        parts = []
        
        # Add regulation metadata
        if record.get('regulation_type'):
            parts.append(f"Jenis: {record['regulation_type']}")
        
        if record.get('regulation_number'):
            parts.append(f"Nomor: {record['regulation_number']}")
        
        if record.get('year'):
            parts.append(f"Tahun: {record['year']}")
        
        if record.get('enacting_body'):
            parts.append(f"Lembaga: {record['enacting_body']}")
        
        # Add about
        if record.get('about'):
            parts.append(f"Tentang: {record['about']}")
        
        # Add article info
        if record.get('article') and record['article'] != 'N/A':
            parts.append(f"Pasal: {record['article']}")
        
        if record.get('chapter') and record['chapter'] != 'N/A':
            parts.append(f"Bab: {record['chapter']}")
        
        # Add content (truncated)
        if record.get('content'):
            content = record['content'][:500]
            parts.append(f"Isi: {content}")
        
        return " | ".join(parts)
    
    def _rerank_with_model(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank using reranker model"""
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, candidate['text']] for candidate in candidates]
            
            self.logger.debug("Reranking with model", {
                "pairs": len(pairs)
            })
            
            # Get reranker scores
            with torch.no_grad():
                scores = self.reranker_model.compute_score(
                    pairs,
                    normalize=True
                )
            
            # Handle both single score and list of scores
            if isinstance(scores, (int, float)):
                scores = [scores]
            
            self.logger.debug("Reranker scores computed", {
                "scores_count": len(scores),
                "max_score": f"{max(scores):.4f}" if scores else "N/A",
                "min_score": f"{min(scores):.4f}" if scores else "N/A"
            })
            
            # Combine with original results
            reranked = []
            for idx, (candidate, rerank_score) in enumerate(zip(candidates, scores)):
                result = candidate['result'].copy()
                result['rerank_score'] = float(rerank_score)
                
                # Combine rerank score with consensus score
                consensus_score = result.get('consensus_score', result['aggregated_scores']['final'])
                result['final_score'] = 0.6 * rerank_score + 0.4 * consensus_score
                
                reranked.append(result)
            
            # Sort by final score
            reranked.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Return top-k
            return reranked[:top_k]
            
        except Exception as e:
            self.logger.error("Error during reranking", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Fallback: return top-k by consensus score
            self.logger.warning("Falling back to consensus scores")
            return sorted(
                [c['result'] for c in candidates],
                key=lambda x: x.get('consensus_score', x['aggregated_scores']['final']),
                reverse=True
            )[:top_k]
    
    def get_rerank_summary(self, rerank_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary from reranking results"""
        
        results = rerank_data['reranked_results']
        
        if not results:
            return {
                'total_results': 0,
                'score_statistics': {}
            }
        
        summary = {
            'total_results': len(results),
            'score_statistics': {
                'rerank_scores': {
                    'min': min(r['rerank_score'] for r in results),
                    'max': max(r['rerank_score'] for r in results),
                    'mean': sum(r['rerank_score'] for r in results) / len(results)
                },
                'final_scores': {
                    'min': min(r['final_score'] for r in results),
                    'max': max(r['final_score'] for r in results),
                    'mean': sum(r['final_score'] for r in results) / len(results)
                }
            },
            'regulation_types': {},
            'years': {}
        }
        
        # Analyze regulation types
        for result in results:
            reg_type = result['record'].get('regulation_type', 'Unknown')
            summary['regulation_types'][reg_type] = summary['regulation_types'].get(reg_type, 0) + 1
            
            year = result['record'].get('year', 'Unknown')
            summary['years'][year] = summary['years'].get(year, 0) + 1
        
        self.logger.info("Rerank summary generated", {
            "results": len(results),
            "avg_rerank_score": f"{summary['score_statistics']['rerank_scores']['mean']:.4f}",
            "unique_reg_types": len(summary['regulation_types'])
        })
        
        return summary