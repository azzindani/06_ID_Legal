"""
Consensus Building Module - FIXED VERSION
More lenient filtering with better debugging
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
from logger_utils import get_logger
from config import RESEARCH_TEAM_PERSONAS


class ConsensusBuilder:
    """
    FIXED: More lenient consensus building with detailed logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("ConsensusBuilder")
        
        # FIXED: Default to lower threshold
        self.consensus_threshold = config.get('consensus_threshold', 0.3)
        self.enable_cross_validation = config.get('enable_cross_validation', True)
        self.enable_devil_advocate = config.get('enable_devil_advocate', True)
        
        self.logger.info("ConsensusBuilder initialized", {
            "consensus_threshold": self.consensus_threshold,
            "cross_validation": self.enable_cross_validation,
            "devil_advocate": self.enable_devil_advocate
        })
    
    def build_consensus(
        self,
        research_data: Dict[str, Any],
        team_composition: List[str]
    ) -> Dict[str, Any]:
        """
        FIXED: Build consensus with more lenient filtering
        """
        self.logger.info("Building consensus", {
            "total_results": len(research_data.get('all_results', [])),
            "team_size": len(team_composition)
        })
        
        consensus_data = {
            'validated_results': [],
            'consensus_scores': {},
            'validation_metadata': {},
            'agreement_level': 0.0,
            'devil_advocate_flags': [],
            'cross_validation_passed': []
        }
        
        # Group results by document (global_id)
        results_by_doc = defaultdict(list)
        for result in research_data.get('all_results', []):
            global_id = result['record']['global_id']
            results_by_doc[global_id].append(result)
        
        self.logger.debug("Results grouped by document", {
            "unique_documents": len(results_by_doc)
        })
        
        # FIXED: Track filtering statistics
        total_docs = len(results_by_doc)
        passed_threshold = 0
        
        # Calculate consensus for each document
        for global_id, doc_results in results_by_doc.items():
            consensus_result = self._calculate_document_consensus(
                global_id=global_id,
                doc_results=doc_results,
                team_composition=team_composition
            )
            
            if consensus_result:
                consensus_data['validated_results'].append(consensus_result)
                consensus_data['consensus_scores'][global_id] = consensus_result['consensus_score']
                passed_threshold += 1
        
        # FIXED: Log filtering statistics
        self.logger.info("Consensus filtering results", {
            "total_docs": total_docs,
            "passed_threshold": passed_threshold,
            "filtered_out": total_docs - passed_threshold,
            "threshold": f"{self.consensus_threshold:.0%}"
        })
        
        # FIXED: If no results, log detailed debug info
        if len(consensus_data['validated_results']) == 0:
            self.logger.warning("NO RESULTS PASSED CONSENSUS!")
            self._log_consensus_debug(results_by_doc, team_composition)
        
        # Sort by consensus score
        consensus_data['validated_results'].sort(
            key=lambda x: x['consensus_score'],
            reverse=True
        )
        
        # Cross-validation
        if self.enable_cross_validation:
            self.logger.info("Performing cross-validation")
            consensus_data['cross_validation_passed'] = self._cross_validate(
                consensus_data['validated_results']
            )
        
        # Devil's advocate review
        if self.enable_devil_advocate and 'devils_advocate' in team_composition:
            self.logger.info("Performing devil's advocate review")
            consensus_data['devil_advocate_flags'] = self._devils_advocate_review(
                consensus_data['validated_results'],
                research_data.get('persona_results', {}).get('devils_advocate', [])
            )
        
        # Calculate overall agreement level
        consensus_data['agreement_level'] = self._calculate_agreement_level(
            results_by_doc,
            team_composition
        )
        
        self.logger.success("Consensus building completed", {
            "validated_results": len(consensus_data['validated_results']),
            "agreement_level": f"{consensus_data['agreement_level']:.2%}",
            "cross_validation_passed": len(consensus_data['cross_validation_passed']),
            "devil_advocate_flags": len(consensus_data['devil_advocate_flags'])
        })
        
        return consensus_data
    
    def _log_consensus_debug(self, results_by_doc: Dict, team_composition: List[str]):
        """FIXED: Log detailed debug info when no results pass"""
        self.logger.warning("Consensus Debug Information:")
        self.logger.info(f"  Team size: {len(team_composition)}")
        self.logger.info(f"  Required threshold: {self.consensus_threshold:.0%}")
        self.logger.info(f"  Required personas: {int(len(team_composition) * self.consensus_threshold)}")
        
        # Sample 5 documents
        for global_id, doc_results in list(results_by_doc.items())[:5]:
            personas = set(r['metadata'].get('persona', 'unknown') for r in doc_results)
            voting_ratio = len(personas) / len(team_composition)
            
            self.logger.info(f"  Doc {global_id}:")
            self.logger.info(f"    Personas found: {len(personas)}/{len(team_composition)} ({voting_ratio:.0%})")
            self.logger.info(f"    Personas: {', '.join(personas)}")
            self.logger.info(f"    Passed: {'YES' if voting_ratio >= self.consensus_threshold else 'NO'}")
        
        self.logger.warning(f"SOLUTION: Lower consensus_threshold to {self.consensus_threshold * 0.5:.2f} or ensure all personas search")
    
    def _calculate_document_consensus(
        self,
        global_id: str,
        doc_results: List[Dict[str, Any]],
        team_composition: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        FIXED: Calculate consensus with logging
        """
        # Count how many team members found this document
        personas_found = set()
        for result in doc_results:
            persona = result['metadata'].get('persona', 'unknown')
            personas_found.add(persona)
        
        # Calculate voting ratio
        voting_ratio = len(personas_found) / len(team_composition)
        
        # FIXED: Log rejected documents
        if voting_ratio < self.consensus_threshold:
            self.logger.debug(f"Doc {global_id} rejected", {
                "voting_ratio": f"{voting_ratio:.0%}",
                "threshold": f"{self.consensus_threshold:.0%}",
                "personas": len(personas_found)
            })
            return None
        
        # Aggregate scores from different personas
        score_aggregation = {
            'final': [],
            'semantic': [],
            'keyword': [],
            'kg': [],
            'authority': [],
            'temporal': [],
            'completeness': []
        }
        
        for result in doc_results:
            for score_type in score_aggregation.keys():
                score_aggregation[score_type].append(result['scores'][score_type])
        
        # Calculate mean scores
        aggregated_scores = {
            score_type: sum(scores) / len(scores)
            for score_type, scores in score_aggregation.items()
        }
        
        # Calculate consensus score (weighted by voting ratio and score quality)
        consensus_score = aggregated_scores['final'] * (0.7 + 0.3 * voting_ratio)
        
        # Get representative result (highest scoring one)
        representative = max(doc_results, key=lambda x: x['scores']['final'])
        
        consensus_result = {
            'global_id': global_id,
            'record': representative['record'],
            'consensus_score': consensus_score,
            'voting_ratio': voting_ratio,
            'personas_agreed': list(personas_found),
            'aggregated_scores': aggregated_scores,
            'score_variance': self._calculate_score_variance(score_aggregation['final']),
            'metadata': representative['metadata']
        }
        
        return consensus_result
    
    def _calculate_score_variance(self, scores: List[float]) -> float:
        """Calculate variance in scores"""
        if len(scores) <= 1:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance
    
    def _cross_validate(self, results: List[Dict[str, Any]]) -> List[str]:
        """Cross-validate results for consistency"""
        validated = []
        
        for result in results:
            # FIXED: More lenient cross-validation
            # Check if multiple personas agreed (even 50%)
            if result['voting_ratio'] >= 0.5:
                validated.append(result['global_id'])
            
            # Or check if score variance is low
            elif result['score_variance'] < 0.1:
                validated.append(result['global_id'])
        
        self.logger.debug("Cross-validation completed", {
            "validated": len(validated),
            "total": len(results)
        })
        
        return validated
    
    def _devils_advocate_review(
        self,
        results: List[Dict[str, Any]],
        devil_advocate_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Review from devil's advocate perspective"""
        flags = []
        
        # Get devil's advocate's findings
        devil_findings = {
            r['record']['global_id']: r
            for r in devil_advocate_results
        }
        
        for result in results:
            global_id = result['global_id']
            
            # Check if devil's advocate found this result
            if global_id in devil_findings:
                devil_result = devil_findings[global_id]
                
                # Compare scores
                consensus_score = result['consensus_score']
                devil_score = devil_result['scores']['final']
                
                # Flag if significantly different
                if abs(consensus_score - devil_score) > 0.2:
                    flags.append({
                        'global_id': global_id,
                        'type': 'score_discrepancy',
                        'consensus_score': consensus_score,
                        'devil_score': devil_score,
                        'difference': abs(consensus_score - devil_score),
                        'message': 'Devil\'s advocate scored this significantly differently'
                    })
            else:
                # Flag if devil's advocate didn't find it
                if result['voting_ratio'] < 0.8:
                    flags.append({
                        'global_id': global_id,
                        'type': 'missing_from_devil',
                        'voting_ratio': result['voting_ratio'],
                        'message': 'Devil\'s advocate did not identify this result'
                    })
        
        self.logger.debug("Devil's advocate review completed", {
            "flags_raised": len(flags)
        })
        
        return flags
    
    def _calculate_agreement_level(
        self,
        results_by_doc: Dict[str, List[Dict[str, Any]]],
        team_composition: List[str]
    ) -> float:
        """Calculate overall team agreement level"""
        if not results_by_doc:
            return 0.0
        
        agreement_scores = []
        
        for doc_results in results_by_doc.values():
            personas_found = set(
                r['metadata'].get('persona', 'unknown')
                for r in doc_results
            )
            agreement = len(personas_found) / len(team_composition)
            agreement_scores.append(agreement)
        
        return sum(agreement_scores) / len(agreement_scores)
    
    def filter_by_consensus(
        self,
        consensus_data: Dict[str, Any],
        min_voting_ratio: float = None,
        require_cross_validation: bool = False
    ) -> List[Dict[str, Any]]:
        """Filter results based on consensus criteria"""
        min_ratio = min_voting_ratio or self.consensus_threshold
        
        filtered = []
        
        for result in consensus_data['validated_results']:
            # Check voting ratio
            if result['voting_ratio'] < min_ratio:
                continue
            
            # Check cross-validation if required
            if require_cross_validation:
                if result['global_id'] not in consensus_data['cross_validation_passed']:
                    continue
            
            filtered.append(result)
        
        self.logger.info("Consensus filtering completed", {
            "before": len(consensus_data['validated_results']),
            "after": len(filtered),
            "min_voting_ratio": f"{min_ratio:.2f}"
        })
        
        return filtered