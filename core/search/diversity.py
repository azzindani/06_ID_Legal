# core/search/diversity.py
"""
Diversity filtering for search results.
Extracted from EnhancedKGSearchEngine._apply_diversity_filter()
"""
from typing import List, Dict, Set
from utils.logging_config import get_logger, LogBlock

logger = get_logger(__name__)

class DiversityFilter:
    """
    Applies diversity filtering to ensure varied results.
    """
    
    @staticmethod
    def apply_diversity(candidates: List[Dict], target_count: int) -> List[Dict]:
        """
        Apply diversity filtering to candidates.
        
        Args:
            candidates: List of scored candidates
            target_count: Desired number of diverse results
        
        Returns:
            List of diverse candidates
        """
        if len(candidates) <= target_count:
            logger.info(f"No filtering needed: {len(candidates)} ≤ {target_count}")
            return candidates
        
        logger.info(f"Applying diversity filter: {len(candidates)} → {target_count}")
        
        with LogBlock(logger, "Diversity filtering", level=logging.DEBUG):
            diverse_results = []
            seen_reg_types: Set[str] = set()
            seen_domains: Set[str] = set()
            seen_hierarchy: Set[int] = set()
            
            # First pass: prioritize diversity
            for candidate in candidates:
                if len(diverse_results) >= target_count:
                    break
                
                try:
                    record = candidate['record']
                    reg_type = record['regulation_type']
                    domain = record.get('kg_primary_domain', 'Unknown')
                    hierarchy = record.get('kg_hierarchy_level', 5)
                    
                    # Add if brings new diversity
                    is_diverse = (
                        reg_type not in seen_reg_types or
                        domain not in seen_domains or
                        hierarchy not in seen_hierarchy or
                        len(diverse_results) < target_count // 2  # Always take top half
                    )
                    
                    if is_diverse:
                        diverse_results.append(candidate)
                        seen_reg_types.add(reg_type)
                        seen_domains.add(domain)
                        seen_hierarchy.add(hierarchy)
                        
                        logger.debug(
                            f"Added diverse candidate {len(diverse_results)}: "
                            f"{reg_type}, domain={domain}, hierarchy={hierarchy}"
                        )
                except Exception as e:
                    logger.warning(f"Error processing candidate for diversity: {e}")
                    continue
            
            # Second pass: fill remaining slots with highest scores
            remaining = target_count - len(diverse_results)
            if remaining > 0:
                logger.debug(f"Filling {remaining} remaining slots with top scores")
                for candidate in candidates:
                    if remaining <= 0:
                        break
                    if candidate not in diverse_results:
                        diverse_results.append(candidate)
                        remaining -= 1
            
            logger.info(
                f"Diversity filter complete: {len(diverse_results)} results, "
                f"{len(seen_reg_types)} reg types, {len(seen_domains)} domains"
            )
            
            return diverse_results[:target_count]

import logging