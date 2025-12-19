"""
Hybrid Search Engine - HIGH-PERFORMANCE VERSION with FAISS
Properly handles thresholds, normalization, and multi-round degradation

PERFORMANCE OPTIMIZATIONS:
- FAISS indexing for 10-100x faster semantic search on large datasets
- Supports millions of documents with sub-second search times
- Index persistence (save/load) to avoid rebuilding
- Incremental indexing support
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import sparse
from pathlib import Path
from utils.logger_utils import get_logger
from config import RESEARCH_TEAM_PERSONAS, KG_WEIGHTS

# Import FAISS index manager and query cache
from .faiss_index_manager import FAISSIndexManager, choose_optimal_index_config, FAISS_AVAILABLE
from .query_cache import QueryResultCache


class HybridSearchEngine:
    """
    High-performance hybrid search with FAISS indexing

    Features:
    - Semantic search: FAISS approximate nearest neighbor (10-100x faster than linear)
    - Keyword search: TF-IDF with sparse matrices
    - Knowledge graph scoring: Entity matching and relationship analysis
    - Supports millions of documents with sub-second search times
    """

    def __init__(
        self,
        data_loader,
        embedding_model,
        reranker_model,
        use_faiss: bool = True,
        faiss_index_path: Optional[str] = None,
        faiss_index_type: str = 'auto',
        use_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl_seconds: int = 3600,
        minimum_relevance_threshold: float = 0.15
    ):
        """
        Initialize hybrid search engine with optional FAISS indexing and caching

        Args:
            data_loader: Data loader with document embeddings
            embedding_model: Model for query embedding
            reranker_model: Model for reranking results
            use_faiss: Whether to use FAISS indexing (recommended for >10k docs)
            faiss_index_path: Path to save/load FAISS index
            faiss_index_type: Index type ('auto', 'Flat', 'IVF', 'HNSW')
            use_cache: Whether to cache query results (recommended)
            cache_size: Maximum number of cached queries
            cache_ttl_seconds: Time-to-live for cache entries in seconds
            minimum_relevance_threshold: Minimum combined relevance score (semantic+keyword)/2
                                        to prevent irrelevant documents from ranking high
        """
        self.data_loader = data_loader
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.logger = get_logger("HybridSearch")

        # Relevance gating threshold
        self.minimum_relevance_threshold = minimum_relevance_threshold

        # Use the same device as embedding model (supports multi-GPU distribution)
        # In multi-GPU setup, embedding model may be on cuda:1, not cuda:0
        if hasattr(embedding_model, 'device'):
            self.device = embedding_model.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cache for normalized embeddings
        self._normalized_embeddings = None

        # FAISS index manager
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index_manager = None
        self.faiss_index_path = faiss_index_path

        if self.use_faiss:
            self._initialize_faiss_index(faiss_index_type)
        elif use_faiss and not FAISS_AVAILABLE:
            self.logger.warning("FAISS requested but not available - install with: pip install faiss-cpu")

        # Query result cache
        self.query_cache = QueryResultCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl_seconds,
            enabled=use_cache
        )

        self.logger.info("HybridSearchEngine initialized", {
            "device": str(self.device),
            "use_faiss": self.use_faiss,
            "faiss_available": FAISS_AVAILABLE,
            "cache_enabled": use_cache,
            "cache_size": cache_size
        })

    def _initialize_faiss_index(self, index_type: str = 'auto'):
        """Initialize FAISS index for semantic search"""
        try:
            # Get embedding dimension
            embeddings = self.data_loader.data_embeddings
            if embeddings is None:
                self.logger.warning("No embeddings available - FAISS index not created")
                self.use_faiss = False
                return

            embedding_dim = embeddings.shape[1]
            num_docs = embeddings.shape[0]

            # Choose optimal config if auto
            if index_type == 'auto':
                config = choose_optimal_index_config(num_docs, embedding_dim)
                index_type = config['index_type']
                nlist = config.get('nlist', 100)
                nprobe = config.get('nprobe', 10)
                self.logger.info(f"Auto-selected FAISS config: {config['description']}")
            else:
                nlist = int(np.sqrt(num_docs))
                nprobe = min(20, nlist // 5)

            # Create FAISS index manager
            self.faiss_index_manager = FAISSIndexManager(
                embedding_dim=embedding_dim,
                index_type=index_type,
                nlist=nlist,
                nprobe=nprobe,
                use_gpu=self.device.type == 'cuda',
                gpu_id=self.device.index if self.device.type == 'cuda' else 0
            )

            # Try to load existing index
            if self.faiss_index_path and Path(self.faiss_index_path).exists():
                self.logger.info(f"Loading FAISS index from {self.faiss_index_path}")
                self.faiss_index_manager.load_index(self.faiss_index_path)
                self.logger.info("FAISS index loaded successfully")
            else:
                # Build new index
                self.logger.info(f"Building FAISS index for {num_docs} documents...")

                # Convert to numpy and normalize
                if isinstance(embeddings, torch.Tensor):
                    embeddings_np = embeddings.cpu().numpy()
                else:
                    embeddings_np = embeddings

                build_time = self.faiss_index_manager.build_index(
                    embeddings_np,
                    normalize=True
                )

                # Save index if path provided
                if self.faiss_index_path:
                    self.logger.info(f"Saving FAISS index to {self.faiss_index_path}")
                    self.faiss_index_manager.save_index(self.faiss_index_path)

                self.logger.info(f"FAISS index built in {build_time:.2f}s ({num_docs/build_time:.0f} docs/sec)")

        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            self.use_faiss = False
    
    def search_with_persona(
        self,
        query: str,
        persona_name: str,
        phase_config: Dict[str, Any],
        priority_weights: Dict[str, float],
        top_k: int = 50,
        round_number: int = 1,
        quality_multiplier: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Execute search with persona - HIGH-PERFORMANCE VERSION with caching

        Args:
            query: Search query
            persona_name: Persona name
            phase_config: Phase configuration
            priority_weights: Priority weights
            top_k: Number of results
            round_number: Current round number
            quality_multiplier: Quality degradation multiplier (1.0 = no degradation)

        Returns:
            List of search results with scores and metadata
        """
        # Check cache first
        cache_key_params = {
            "persona": persona_name,
            "top_k": top_k,
            "round": round_number,
            "quality_mult": f"{quality_multiplier:.3f}",
            "phase": phase_config.get('description', 'default')
        }

        cached_result = self.query_cache.get(query, **cache_key_params)
        if cached_result is not None:
            self.logger.info("Cache hit - returning cached results", {
                "persona": persona_name,
                "cache_hit_rate": f"{self.query_cache.get_hit_rate():.1%}"
            })
            return cached_result

        # Cache miss - perform search
        self.logger.info("Cache miss - executing search", {
            "persona": persona_name,
            "phase": phase_config.get('description', 'N/A'),
            "round": round_number,
            "quality_mult": f"{quality_multiplier:.3f}"
        })

        # Get persona
        persona = RESEARCH_TEAM_PERSONAS.get(persona_name)
        if not persona:
            self.logger.error("Persona not found", {"persona": persona_name})
            return []
        
        # Apply persona weights
        persona_weights = self._apply_persona_weights(priority_weights, persona)
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            return []
        
        # Execute search with FIXED logic
        results = self._hybrid_search_fixed(
            query=query,
            query_embedding=query_embedding,
            phase_config=phase_config,
            weights=persona_weights,
            top_k=top_k,
            persona=persona,
            quality_multiplier=quality_multiplier
        )

        # Store in cache
        self.query_cache.put(query, results, **cache_key_params)

        self.logger.info("Persona search completed", {
            "persona": persona_name,
            "results": len(results),
            "cache_stats": self.query_cache.get_stats()
        })

        return results
    
    def _apply_persona_weights(
        self,
        base_weights: Dict[str, float],
        persona: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply persona's search style to base weights"""

        persona_style = persona['search_style']
        weights = base_weights.copy()

        # Apply persona preferences
        if 'semantic_weight' in persona_style:
            weights['semantic_match'] = persona_style['semantic_weight']

        if 'authority_weight' in persona_style:
            weights['authority_hierarchy'] = persona_style['authority_weight']

        if 'kg_weight' in persona_style:
            weights['knowledge_graph'] = persona_style['kg_weight']

        if 'temporal_weight' in persona_style:
            weights['temporal_relevance'] = persona_style['temporal_weight']

        # Normalize - FIXED: Add fallback for zero total
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        else:
            # Fallback to equal weights if all weights are zero
            self.logger.warning("All weights are zero, using equal distribution")
            num_weights = len(weights)
            weights = {k: 1.0/num_weights for k in weights.keys()}

        return weights
    
    def _get_query_embedding(self, query: str) -> Optional[torch.Tensor]:
        """Get and normalize query embedding - FIXED"""
        try:
            with torch.no_grad():
                inputs = self.embedding_model.tokenize(
                    [query], 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.embedding_model(**inputs)
                
                # Mean pooling
                embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                query_embedding = sum_embeddings / sum_mask
                
                # IMPORTANT: Normalize immediately
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
                
                return query_embedding[0]
                
        except Exception as e:
            self.logger.error("Error getting query embedding", {"error": str(e)})
            return None
    
    def _get_normalized_doc_embeddings(self) -> torch.Tensor:
        """Get cached normalized document embeddings"""
        if self._normalized_embeddings is None:
            self.logger.info("Normalizing document embeddings (one-time operation)")
            # Move embeddings to same device as embedding model (for multi-GPU support)
            doc_embeddings = self.data_loader.embeddings.to(self.device)
            self._normalized_embeddings = torch.nn.functional.normalize(
                doc_embeddings,
                p=2,
                dim=1
            )
        return self._normalized_embeddings
    
    def _hybrid_search_fixed(
        self,
        query: str,
        query_embedding: torch.Tensor,
        phase_config: Dict[str, Any],
        weights: Dict[str, float],
        top_k: int,
        persona: Dict[str, Any],
        quality_multiplier: float
    ) -> List[Dict[str, Any]]:
        """
        FIXED hybrid search - proper threshold handling and score combination
        """
        
        candidates = phase_config['candidates']
        
        # FIXED: Apply quality multiplier to thresholds BEFORE filtering
        # Lower thresholds = more permissive (opposite of multiplying scores)
        adjusted_semantic_threshold = phase_config['semantic_threshold'] * quality_multiplier
        adjusted_keyword_threshold = phase_config['keyword_threshold'] * quality_multiplier
        
        self.logger.debug("Adjusted thresholds", {
            "semantic": f"{adjusted_semantic_threshold:.4f}",
            "keyword": f"{adjusted_keyword_threshold:.4f}",
            "quality_mult": f"{quality_multiplier:.3f}"
        })
        
        # 1. Semantic search with MORE candidates
        semantic_scores, semantic_indices = self._semantic_search_fixed(
            query_embedding, 
            top_k=candidates * 2  # Get 2x candidates for filtering
        )
        
        # 2. Keyword search with MORE candidates
        keyword_scores, keyword_indices = self._keyword_search_fixed(
            query,
            top_k=candidates * 2  # Get 2x candidates for filtering
        )
        
        # 3. FIXED: Create candidate pool from UNION (not intersection)
        all_indices = set()
        
        # Add semantic candidates that pass threshold
        for idx, score in zip(semantic_indices.tolist(), semantic_scores.tolist()):
            if score >= adjusted_semantic_threshold:
                all_indices.add(idx)
        
        # Add keyword candidates that pass threshold
        for idx, score in zip(keyword_indices, keyword_scores):
            if score >= adjusted_keyword_threshold:
                all_indices.add(idx)
        
        # If too few candidates, relax thresholds progressively
        if len(all_indices) < top_k:
            self.logger.warning("Too few candidates, relaxing thresholds", {
                "current_count": len(all_indices),
                "target": top_k
            })
            
            # Add top semantic results regardless of threshold
            for idx in semantic_indices[:top_k].tolist():
                all_indices.add(idx)
            
            # Add top keyword results regardless of threshold
            for idx in keyword_indices[:top_k]:
                all_indices.add(idx)
        
        self.logger.debug("Candidate pool created", {
            "total_candidates": len(all_indices),
            "semantic_passed": sum(1 for s in semantic_scores if s >= adjusted_semantic_threshold),
            "keyword_passed": sum(1 for s in keyword_scores if s >= adjusted_keyword_threshold)
        })
        
        # 4. Score all candidates with hybrid approach
        scored_results = []
        
        for idx in all_indices:
            if idx >= len(self.data_loader.all_records):
                continue
            
            record = self.data_loader.all_records[idx]
            
            # Get individual scores (normalized 0-1)
            semantic_score = 0.0
            if idx in semantic_indices:
                pos = (semantic_indices == idx).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    semantic_score = semantic_scores[pos[0]].item()
            
            keyword_score = 0.0
            if idx in keyword_indices:
                pos = np.where(keyword_indices == idx)[0]
                if len(pos) > 0:
                    keyword_score = keyword_scores[pos[0]]
            
            # Normalize scores to 0-1 range if needed
            semantic_score = max(0.0, min(1.0, semantic_score))
            keyword_score = max(0.0, min(1.0, keyword_score))

            # RELEVANCE GATING: Filter out documents with insufficient query relevance
            # This prevents irrelevant but "high quality" documents from ranking high
            # A document must have SOME relevance to the query to be considered
            relevance_score = (semantic_score + keyword_score) / 2.0

            if relevance_score < self.minimum_relevance_threshold:
                # Skip this document - too irrelevant regardless of other scores
                # Example: Banking regulation for a tax query (high authority, zero relevance)
                # Configurable via minimum_relevance_threshold parameter (default: 0.15)
                continue

            # Get KG score
            kg_score = self._calculate_kg_score(record, query)
            
            # Get other scores (already 0-1)
            authority_score = record['kg_authority_score']
            temporal_score = record['kg_temporal_score']
            completeness_score = record['kg_completeness_score']
            
            # FIXED: Weighted combination with normalized scores
            final_score = (
                weights.get('semantic_match', 0.25) * semantic_score +
                weights.get('keyword_precision', 0.15) * keyword_score +
                weights.get('knowledge_graph', 0.20) * kg_score +
                weights.get('authority_hierarchy', 0.20) * authority_score +
                weights.get('temporal_relevance', 0.10) * temporal_score +
                weights.get('legal_completeness', 0.10) * completeness_score
            )
            
            # Apply persona accuracy bonus
            accuracy_bonus = persona.get('accuracy_bonus', 0.0)
            final_score *= (1.0 + accuracy_bonus)
            
            # Ensure score is in valid range
            final_score = max(0.0, min(1.0, final_score))
            
            result = {
                'index': idx,
                'record': record,
                'scores': {
                    'final': final_score,
                    'semantic': semantic_score,
                    'keyword': keyword_score,
                    'kg': kg_score,
                    'authority': authority_score,
                    'temporal': temporal_score,
                    'completeness': completeness_score
                },
                'metadata': {
                    'persona': persona['name'],
                    'phase': phase_config.get('description', 'N/A'),
                    'global_id': record['global_id'],
                    'regulation_type': record['regulation_type'],
                    'regulation_number': record['regulation_number'],
                    'year': record['year']
                }
            }
            
            scored_results.append(result)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x['scores']['final'], reverse=True)
        
        # Return top-k
        final_results = scored_results[:top_k]
        
        self.logger.info("Hybrid search completed", {
            "candidates_scored": len(scored_results),
            "results_returned": len(final_results),
            "top_score": f"{final_results[0]['scores']['final']:.4f}" if final_results else "N/A",
            "min_score": f"{final_results[-1]['scores']['final']:.4f}" if final_results else "N/A"
        })
        
        return final_results
    
    def _semantic_search_fixed(
        self,
        query_embedding: torch.Tensor,
        top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        High-performance semantic search with FAISS support

        Falls back to linear search if FAISS is not available.

        Performance:
        - FAISS (1M docs): ~10-50ms
        - Linear (1M docs): ~1000ms
        - Speedup: 10-100x with FAISS
        """
        try:
            # Use FAISS if available (10-100x faster)
            if self.use_faiss and self.faiss_index_manager is not None:
                # Convert to numpy
                if isinstance(query_embedding, torch.Tensor):
                    query_np = query_embedding.cpu().numpy()
                else:
                    query_np = query_embedding

                # Search with FAISS (returns inner product scores for normalized vectors)
                scores_np, indices_np = self.faiss_index_manager.search(
                    query_np,
                    top_k=top_k,
                    normalize=True
                )

                # Convert inner product [-1, 1] to [0, 1] range
                scores_np = (scores_np + 1) / 2

                # Convert back to torch
                scores = torch.from_numpy(scores_np)
                indices = torch.from_numpy(indices_np).long()

                return scores, indices

            # Fallback to linear search (slower but works without FAISS)
            else:
                # Get normalized document embeddings
                doc_embeddings_norm = self._get_normalized_doc_embeddings()

                # Query is already normalized in _get_query_embedding
                query_norm = query_embedding.unsqueeze(0)

                # Compute cosine similarity (already in [-1, 1])
                similarities = torch.mm(query_norm, doc_embeddings_norm.t()).squeeze(0)

                # Convert to [0, 1] range: (sim + 1) / 2
                similarities = (similarities + 1) / 2

                # Get top-k
                scores, indices = torch.topk(similarities, min(top_k, len(similarities)))

                return scores.cpu(), indices.cpu()

        except Exception as e:
            self.logger.error("Semantic search error", {"error": str(e), "use_faiss": self.use_faiss})
            return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    
    def _keyword_search_fixed(
        self, 
        query: str, 
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FIXED keyword search with proper normalization"""
        try:
            if self.data_loader.tfidf_matrix is None:
                self.logger.warning("TF-IDF matrix not available")
                return np.zeros(0), np.array([], dtype=np.int64)
            
            # Transform query
            query_tfidf = self.data_loader.tfidf_vectorizer.transform([query])
            
            # Compute similarity
            similarities = (self.data_loader.tfidf_matrix * query_tfidf.T).toarray().flatten()
            
            # Normalize to [0, 1] - TF-IDF scores are already positive
            max_sim = similarities.max()
            if max_sim > 0:
                similarities = similarities / max_sim
            
            # Get top-k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]
            
            return top_scores, top_indices
            
        except Exception as e:
            self.logger.error("Keyword search error", {"error": str(e)})
            return np.zeros(0), np.array([], dtype=np.int64)
    
    def _calculate_kg_score(self, record: Dict[str, Any], query: str) -> float:
        """Calculate KG score - returns normalized 0-1 value"""
        
        kg_score = 0.0
        query_lower = query.lower()
        
        # Entity matching (max 0.3)
        entity_count = record.get('kg_entity_count', 0)
        if entity_count > 0:
            kg_score += min(0.3, entity_count * 0.05)
        
        # Cross-reference (max 0.2)
        cross_ref_count = record.get('kg_cross_ref_count', 0)
        if cross_ref_count > 0:
            kg_score += min(0.2, cross_ref_count * 0.03) * KG_WEIGHTS['cross_reference']
        
        # Domain matching (0.2)
        domain = record.get('kg_primary_domain', '').lower()
        if domain and domain in query_lower:
            kg_score += 0.2 * KG_WEIGHTS['domain_match']
        
        # Connectivity (0.15)
        connectivity = record.get('kg_connectivity_score', 0.0)
        kg_score += connectivity * 0.15 * KG_WEIGHTS['connectivity_boost']
        
        # Legal actions (0.1 each)
        if record.get('kg_has_obligations') and any(w in query_lower for w in ['wajib', 'harus', 'kewajiban']):
            kg_score += 0.1 * KG_WEIGHTS['legal_action_match']
        
        if record.get('kg_has_prohibitions') and any(w in query_lower for w in ['dilarang', 'tidak boleh', 'larangan']):
            kg_score += 0.1 * KG_WEIGHTS['legal_action_match']
        
        if record.get('kg_has_permissions') and any(w in query_lower for w in ['dapat', 'boleh', 'izin']):
            kg_score += 0.1 * KG_WEIGHTS['legal_action_match']
        
        # Sanctions (0.15)
        if any(w in query_lower for w in ['sanksi', 'pidana', 'denda', 'hukuman']):
            if record.get('kg_has_prohibitions') or 'sanksi' in str(record.get('content', '')).lower():
                kg_score += 0.15 * KG_WEIGHTS['sanction_relevance']

        return min(1.0, kg_score)

    def metadata_first_search(
        self,
        regulation_references: List[Dict[str, Any]],
        top_k: int = 20,
        query: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Direct metadata search with PERFECT SCORE OVERRIDE for exact matches.

        This implements the original notebook's metadata-first strategy where
        explicit regulation references (e.g., "UU No. 13 Tahun 2003") get
        perfect scores to ensure they appear at the top.

        Args:
            regulation_references: List of dicts with 'type', 'number', 'year', 'confidence'
            top_k: Maximum results to return

        Returns:
            List of results with perfect scores for exact matches
        """
        if not regulation_references:
            return []

        self.logger.info("Executing metadata-first search", {
            "references": len(regulation_references)
        })

        results = []
        seen_ids = set()

        for ref in regulation_references:
            ref_type = ref.get('type', '').upper()
            ref_number = ref.get('number')
            ref_year = ref.get('year')
            confidence = ref.get('confidence', 0.3)

            for idx, record in enumerate(self.data_loader.all_records):
                global_id = record.get('global_id', idx)

                if global_id in seen_ids:
                    continue

                # Get record metadata
                doc_type = str(record.get('regulation_type', '')).upper()
                doc_number = str(record.get('regulation_number', ''))
                doc_year = str(record.get('year', ''))

                # Normalize types for comparison
                type_match = self._normalize_regulation_type(ref_type) == self._normalize_regulation_type(doc_type)
                number_match = ref_number and doc_number and str(ref_number) == str(doc_number)
                year_match = ref_year and doc_year and str(ref_year) == str(doc_year)

                # Calculate match score
                match_score = 0.0
                match_type = None

                # PERFECT SCORE OVERRIDE: Triple match (type + number + year)
                if type_match and number_match and year_match:
                    match_score = 1.0  # PERFECT SCORE
                    match_type = 'exact_triple'
                    self.logger.debug(f"Perfect match: {doc_type} {doc_number}/{doc_year}")

                # Strong match: Type + Number (without year)
                elif type_match and number_match:
                    match_score = 0.85
                    match_type = 'type_number'

                # Partial match: Type + Year
                elif type_match and year_match:
                    match_score = 0.7
                    match_type = 'type_year'

                # Type only match (for vague references)
                elif type_match and confidence <= 0.3:
                    match_score = 0.4
                    match_type = 'type_only'

                if match_score > 0:
                    seen_ids.add(global_id)

                    # Calculate KG score for better ranking
                    kg_score = self._calculate_kg_score(record, query) if query else 0.0

                    result = {
                        'index': idx,
                        'record': record,
                        'scores': {
                            'final': match_score,
                            'metadata_match': match_score,
                            'semantic': 0.0,
                            'keyword': 0.0,
                            'kg': kg_score,
                            'authority': 0.0,
                            'temporal': 0.0,
                            'completeness': 0.0
                        },
                        'metadata': {
                            'global_id': global_id,
                            'regulation_type': doc_type,
                            'regulation_number': doc_number,
                            'year': doc_year,
                            'match_type': match_type,
                            'reference_confidence': confidence,
                        }
                    }
                    results.append(result)

        # Sort by score (perfect matches first)
        results.sort(key=lambda x: x['scores']['final'], reverse=True)

        self.logger.info("Metadata-first search completed", {
            "results": len(results),
            "perfect_matches": sum(1 for r in results if r['scores']['final'] == 1.0)
        })

        return results[:top_k]

    def _normalize_regulation_type(self, reg_type: str) -> str:
        """Normalize regulation type for comparison"""
        reg_type = reg_type.upper().strip()

        # Normalize "UNDANG UNDANG" (with space) to "UNDANG-UNDANG" (with hyphen)
        reg_type = reg_type.replace('UNDANG UNDANG', 'UNDANG-UNDANG')

        # Map full names to abbreviations
        mappings = {
            'UNDANG-UNDANG': 'UU',
            'PERATURAN PEMERINTAH': 'PP',
            'PERATURAN PRESIDEN': 'PERPRES',
            'PERATURAN MENTERI': 'PERMEN',
            'PERATURAN DAERAH': 'PERDA',
            'KEPUTUSAN PRESIDEN': 'KEPPRES',
            'KEPUTUSAN MENTERI': 'KEPMEN',
        }

        for full_name, abbrev in mappings.items():
            if full_name in reg_type or reg_type == abbrev:
                return abbrev

        return reg_type