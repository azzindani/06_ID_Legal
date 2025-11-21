"""
Multi-Stage Research Engine - FIXED VERSION
Proper threshold degradation and adaptive search with correct return values
"""

from typing import Dict, Any, List
from collections import defaultdict
from logger_utils import get_logger
from config import DEFAULT_SEARCH_PHASES
from core.search.hybrid_search import HybridSearchEngine


class StagesResearchEngine:
    """
    FIXED multi-stage research with proper threshold degradation
    """
    
    def __init__(self, hybrid_search: HybridSearchEngine, config: Dict[str, Any]):
        self.hybrid_search = hybrid_search
        self.config = config
        self.logger = get_logger("StagesResearch")
        
        self.search_phases = config.get('search_phases', DEFAULT_SEARCH_PHASES)
        self.max_rounds = config.get('max_rounds', 5)
        self.initial_quality = config.get('initial_quality', 0.95)
        self.quality_degradation = config.get('quality_degradation', 0.1)
        self.min_quality = config.get('min_quality', 0.5)
        
        self.logger.info("StagesResearchEngine initialized", {
            "max_rounds": self.max_rounds,
            "initial_quality": self.initial_quality,
            "degradation_rate": self.quality_degradation,
            "min_quality": self.min_quality
        })
    
    def conduct_research(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        team_composition: List[str]
    ) -> Dict[str, Any]:
        """
        FIXED research with adaptive threshold degradation
        Returns dict with all required keys
        """
        self.logger.info("Starting multi-stage research", {
            "query": query[:50] + "..." if len(query) > 50 else query,
            "team_size": len(team_composition),
            "complexity": f"{query_analysis.get('complexity_score', 0):.2f}"
        })
        
        research_data = {
            'query': query,
            'query_analysis': query_analysis,
            'team_composition': team_composition,
            'rounds': [],
            'rounds_executed': 0,  # FIX: Add this key
            'all_results': [],
            'phase_results': defaultdict(list),
            'persona_results': defaultdict(list),
            'total_candidates_evaluated': 0,
            'quality_progression': []
        }
        
        current_quality = self.initial_quality
        round_num = 1
        min_results_threshold = 5
        
        while round_num <= self.max_rounds:
            self.logger.info(f"Round {round_num}/{self.max_rounds}", {
                "quality_multiplier": f"{current_quality:.3f}",
                "effective_threshold": f"{current_quality * 0.2:.4f}"
            })
            
            # Execute round
            round_data = self._execute_research_round(
                query=query,
                query_analysis=query_analysis,
                team_composition=team_composition,
                quality_multiplier=current_quality,
                round_number=round_num
            )
            
            research_data['rounds'].append(round_data)
            research_data['rounds_executed'] = round_num  # FIX: Update count
            research_data['quality_progression'].append(current_quality)
            research_data['total_candidates_evaluated'] += round_data['candidates_evaluated']
            
            # Collect unique results
            existing_ids = {r['record']['global_id'] for r in research_data['all_results']}
            new_results = 0
            
            for result in round_data['results']:
                global_id = result['record']['global_id']
                
                if global_id not in existing_ids:
                    research_data['all_results'].append(result)
                    existing_ids.add(global_id)
                    new_results += 1
                    
                    # Organize by phase
                    phase = result['metadata'].get('phase', 'unknown')
                    research_data['phase_results'][phase].append(result)
                    
                    # Organize by persona
                    persona = result['metadata'].get('persona', 'unknown')
                    research_data['persona_results'][persona].append(result)
            
            self.logger.info(f"Round {round_num} completed", {
                "new_results": new_results,
                "total_unique": len(research_data['all_results']),
                "round_results": len(round_data['results'])
            })
            
            # Check stopping conditions
            total_results = len(research_data['all_results'])
            
            if total_results >= min_results_threshold:
                high_quality_count = sum(
                    1 for r in research_data['all_results'] 
                    if r['scores']['final'] >= 0.6
                )
                
                if high_quality_count >= min_results_threshold:
                    self.logger.success("Found sufficient high-quality results", {
                        "high_quality": high_quality_count,
                        "total": total_results
                    })
                    break
            
            if new_results == 0 and round_num > 1:
                self.logger.info("No new results found, stopping search")
                break
            
            # Degrade quality
            current_quality = max(
                self.min_quality,
                current_quality - self.quality_degradation
            )
            
            if current_quality <= self.min_quality:
                self.logger.info("Minimum quality threshold reached")
                if round_num < self.max_rounds:
                    round_num += 1
                    continue
                break
            
            round_num += 1
        
        # Sort results
        research_data['all_results'].sort(
            key=lambda x: x['scores']['final'],
            reverse=True
        )
        
        # Convert defaultdicts to regular dicts for serialization
        research_data['phase_results'] = dict(research_data['phase_results'])
        research_data['persona_results'] = dict(research_data['persona_results'])
        
        self.logger.success("Multi-stage research completed", {
            "rounds": research_data['rounds_executed'],
            "unique_results": len(research_data['all_results']),
            "total_evaluated": research_data['total_candidates_evaluated'],
            "top_score": f"{research_data['all_results'][0]['scores']['final']:.4f}" if research_data['all_results'] else "N/A"
        })
        
        return research_data
    
    def _execute_research_round(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        team_composition: List[str],
        quality_multiplier: float,
        round_number: int
    ) -> Dict[str, Any]:
        """Execute single research round with quality multiplier"""
        
        round_results = {
            'round_number': round_number,
            'quality_multiplier': quality_multiplier,
            'results': [],
            'candidates_evaluated': 0,
            'phase_breakdown': {},
            'persona_breakdown': {}
        }
        
        enabled_phases = query_analysis['enabled_phases']
        priority_weights = query_analysis['priority_weights']
        
        # Execute each phase
        for phase_name in enabled_phases:
            if phase_name not in self.search_phases:
                continue
            
            phase_config = self.search_phases[phase_name].copy()
            
            if not phase_config.get('enabled', True):
                continue
            
            self.logger.debug(f"Executing phase: {phase_name} (round {round_number})")
            
            phase_results = []
            
            # Each team member searches
            for persona_name in team_composition:
                try:
                    persona_results = self.hybrid_search.search_with_persona(
                        query=query,
                        persona_name=persona_name,
                        phase_config=phase_config,
                        priority_weights=priority_weights,
                        top_k=50,
                        round_number=round_number,
                        quality_multiplier=quality_multiplier
                    )
                    
                    phase_results.extend(persona_results)
                    
                    if persona_name not in round_results['persona_breakdown']:
                        round_results['persona_breakdown'][persona_name] = 0
                    round_results['persona_breakdown'][persona_name] += len(persona_results)
                    
                except Exception as e:
                    self.logger.error(f"Persona search error", {
                        "persona": persona_name,
                        "phase": phase_name,
                        "error": str(e)
                    })
            
            round_results['phase_breakdown'][phase_name] = len(phase_results)
            round_results['results'].extend(phase_results)
            round_results['candidates_evaluated'] += len(phase_results)
        
        # Deduplicate within round
        unique_results = {}
        for result in round_results['results']:
            global_id = result['record']['global_id']
            if global_id not in unique_results or result['scores']['final'] > unique_results[global_id]['scores']['final']:
                unique_results[global_id] = result
        
        round_results['results'] = sorted(
            unique_results.values(),
            key=lambda x: x['scores']['final'],
            reverse=True
        )
        
        return round_results
    
    def get_research_summary(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research summary"""
        
        summary = {
            'total_rounds': research_data.get('rounds_executed', len(research_data['rounds'])),
            'total_unique_results': len(research_data['all_results']),
            'total_candidates_evaluated': research_data['total_candidates_evaluated'],
            'quality_progression': research_data.get('quality_progression', []),
            'phase_distribution': {},
            'persona_distribution': {},
            'score_statistics': {}
        }
        
        # Phase distribution
        for phase, results in research_data['phase_results'].items():
            summary['phase_distribution'][phase] = len(results)
        
        # Persona distribution
        for persona, results in research_data['persona_results'].items():
            summary['persona_distribution'][persona] = len(results)
        
        # Score statistics
        if research_data['all_results']:
            scores = [r['scores']['final'] for r in research_data['all_results']]
            summary['score_statistics'] = {
                'min': min(scores),
                'max': max(scores),
                'mean': sum(scores) / len(scores),
                'median': sorted(scores)[len(scores)//2],
                'top_10_mean': sum(sorted(scores, reverse=True)[:10]) / min(10, len(scores))
            }
        
        return summary