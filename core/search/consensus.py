"""
Consensus Building Module - FIXED VERSION
More lenient filtering with better debugging
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
from utils.logger_utils import get_logger
from config import RESEARCH_TEAM_PERSONAS


class ConsensusBuilder:
    """
    FIXED: More lenient consensus building with detailed logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("ConsensusBuilder")
        
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
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
        total_results_count = len(research_data.get('all_results', []))

        self.logger.info("Building consensus", {
            "total_results": total_results_count,
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

        unique_docs = len(results_by_doc)

        # Adaptive threshold based on pool size
        # Large pools (>1000 unique docs) suggest aggressive expansion - lower threshold
        adaptive_threshold = self.consensus_threshold
        original_threshold = self.consensus_threshold

        if unique_docs > 10000:
            adaptive_threshold = max(0.25, self.consensus_threshold * 0.4)
            self.logger.warning(f"Very large pool detected ({unique_docs} docs), "
                              f"lowering threshold: {original_threshold:.0%} → {adaptive_threshold:.0%}")
        elif unique_docs > 5000:
            adaptive_threshold = max(0.30, self.consensus_threshold * 0.5)
            self.logger.info(f"Large pool detected ({unique_docs} docs), "
                           f"lowering threshold: {original_threshold:.0%} → {adaptive_threshold:.0%}")
        elif unique_docs > 1000:
            adaptive_threshold = max(0.40, self.consensus_threshold * 0.7)
            self.logger.info(f"Moderate pool detected ({unique_docs} docs), "
                           f"lowering threshold: {original_threshold:.0%} → {adaptive_threshold:.0%}")

        # Temporarily override consensus threshold
        original_consensus_threshold = self.consensus_threshold
        self.consensus_threshold = adaptive_threshold

        self.logger.debug("Results grouped by document", {
            "unique_documents": unique_docs,
            "active_threshold": f"{self.consensus_threshold:.0%}"
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
        
        # FIXED: If no results, apply fallback with progressively lower thresholds
        if len(consensus_data['validated_results']) == 0:
            self.logger.warning("NO RESULTS PASSED CONSENSUS!")
            self._log_consensus_debug(results_by_doc, team_composition)

            # FALLBACK: Try progressively lower thresholds
            fallback_thresholds = [0.5, 0.4, 0.3, 0.25, 0.0]

            for fallback_threshold in fallback_thresholds:
                self.logger.info(f"Trying fallback threshold: {fallback_threshold:.0%}")

                for global_id, doc_results in results_by_doc.items():
                    # Count how many team members found this document
                    personas_found = set()
                    for result in doc_results:
                        persona = result['metadata'].get('persona', 'unknown')
                        personas_found.add(persona)

                    # Calculate voting ratio
                    voting_ratio = len(personas_found) / len(team_composition) if team_composition else 0

                    # Check if passes fallback threshold
                    if voting_ratio >= fallback_threshold or fallback_threshold == 0.0:
                        # Already processed this document
                        if any(r['global_id'] == global_id for r in consensus_data['validated_results']):
                            continue

                        # Aggregate scores
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

                        aggregated_scores = {
                            score_type: sum(scores) / len(scores)
                            for score_type, scores in score_aggregation.items()
                        }

                        consensus_score = aggregated_scores['final'] * (0.7 + 0.3 * voting_ratio)
                        representative = max(doc_results, key=lambda x: x['scores']['final'])

                        consensus_result = {
                            'global_id': global_id,
                            'record': representative['record'],
                            'consensus_score': consensus_score,
                            'voting_ratio': voting_ratio,
                            'personas_agreed': list(personas_found),
                            'aggregated_scores': aggregated_scores,
                            'score_variance': self._calculate_score_variance(score_aggregation['final']),
                            'metadata': representative['metadata'],
                            'fallback_applied': True,
                            'fallback_threshold': fallback_threshold
                        }

                        consensus_data['validated_results'].append(consensus_result)
                        consensus_data['consensus_scores'][global_id] = consensus_score

                # Stop if we have enough results
                if len(consensus_data['validated_results']) >= 3:
                    self.logger.info(f"Fallback succeeded with threshold {fallback_threshold:.0%}", {
                        "results": len(consensus_data['validated_results'])
                    })
                    break

            if len(consensus_data['validated_results']) == 0:
                self.logger.error("FALLBACK FAILED - No results even with 0% threshold!")

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
        
        # Restore original threshold
        self.consensus_threshold = original_consensus_threshold

        self.logger.success("Consensus building completed", {
            "validated_results": len(consensus_data['validated_results']),
            "agreement_level": f"{consensus_data['agreement_level']:.2%}",
            "cross_validation_passed": len(consensus_data['cross_validation_passed']),
            "devil_advocate_flags": len(consensus_data['devil_advocate_flags']),
            "adaptive_threshold_used": adaptive_threshold != original_threshold
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
        FIXED: Calculate consensus like original code - compare SCORE against threshold, not voting ratio
        """
        # Count how many team members found this document
        personas_found = set()
        researcher_types = set()
        researcher_scores = {}

        for result in doc_results:
            persona = result['metadata'].get('persona', 'unknown')
            personas_found.add(persona)

            # Track researcher types and scores
            # Map display name back to ID for persona lookup
            persona_id = None
            for pid, pdata in RESEARCH_TEAM_PERSONAS.items():
                if pdata['name'] == persona:
                    persona_id = pid
                    researcher_types.add(pdata['approach'])
                    break

            if persona_id:
                researcher_scores[persona_id] = result['scores']['final']
            else:
                researcher_scores[persona] = result['scores']['final']

        # Calculate voting ratio (for bonuses, not threshold)
        voting_ratio = len(personas_found) / len(team_composition) if team_composition else 0

        # Calculate WEIGHTED SCORE like original code
        total_weight = 0
        weighted_score = 0

        for researcher_id, score in researcher_scores.items():
            # Get persona data
            persona_data = RESEARCH_TEAM_PERSONAS.get(researcher_id)
            if persona_data:
                weight = (persona_data['experience_years'] / 15.0) + persona_data.get('accuracy_bonus', 0)
            else:
                weight = 1.0  # Default weight

            weighted_score += score * weight
            total_weight += weight

        final_weighted_score = weighted_score / total_weight if total_weight > 0 else sum(researcher_scores.values()) / len(researcher_scores)

        # Apply consensus bonuses (like original code)
        if len(personas_found) > 1:
            consensus_bonus = min(0.10, 0.03 * (len(personas_found) - 1))
            final_weighted_score += consensus_bonus

        if len(researcher_types) > 1:
            final_weighted_score += 0.05

        # Adjust threshold for strong agreement
        adjusted_threshold = self.consensus_threshold
        if len(personas_found) >= 3:
            adjusted_threshold *= 0.9

        # FIXED: Compare SCORE against threshold (like original code)
        if final_weighted_score < adjusted_threshold:
            self.logger.debug(f"Doc {global_id} rejected", {
                "score": f"{final_weighted_score:.3f}",
                "threshold": f"{adjusted_threshold:.3f}",
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

        # Use final_weighted_score as consensus_score
        consensus_score = min(1.0, final_weighted_score)

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