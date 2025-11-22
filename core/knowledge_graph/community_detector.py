"""
Dynamic Community Detector - Thematic Clustering for Legal Documents

Uses igraph with Louvain algorithm to detect thematic clusters
in the document network based on cross-references and similarity.
"""

from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from logger_utils import get_logger

logger = get_logger(__name__)

# Try to import igraph, fall back to networkx if not available
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    logger.warning("igraph not available, using simplified clustering")


class DynamicCommunityDetector:
    """
    Detects thematic communities in document network using Louvain algorithm.

    This helps identify related document clusters for better context
    and thematic organization of search results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.resolution = self.config.get('community_resolution', 1.0)
        self.min_community_size = self.config.get('min_community_size', 3)

        # Cache for computed communities
        self._community_cache: Dict[str, List[Dict]] = {}

        logger.info("DynamicCommunityDetector initialized", {
            "igraph_available": IGRAPH_AVAILABLE,
            "resolution": self.resolution
        })

    def detect_communities(
        self,
        documents: List[Dict[str, Any]],
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detect thematic communities in document set.

        Args:
            documents: List of documents with cross-references and metadata
            similarity_threshold: Minimum similarity to create edge

        Returns:
            List of community dicts with members and metadata
        """
        if not documents:
            return []

        if len(documents) < self.min_community_size:
            # Return single community for small sets
            return [{
                'id': 0,
                'size': len(documents),
                'members': [d.get('global_id', i) for i, d in enumerate(documents)],
                'dominant_type': self._get_dominant_type(documents),
                'themes': self._extract_themes(documents)
            }]

        if IGRAPH_AVAILABLE:
            return self._detect_with_igraph(documents, similarity_threshold)
        else:
            return self._detect_simplified(documents, similarity_threshold)

    def _detect_with_igraph(
        self,
        documents: List[Dict[str, Any]],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Use igraph Louvain algorithm for community detection"""

        # Build graph
        n_docs = len(documents)
        edges = []
        weights = []

        # Create document ID mapping
        doc_ids = [d.get('global_id', i) for i, d in enumerate(documents)]
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        # Add edges based on cross-references
        for i, doc in enumerate(documents):
            cross_refs = doc.get('kg_cross_references', [])
            record = doc.get('record', doc)
            if isinstance(record, dict):
                cross_refs = record.get('kg_cross_references', cross_refs)

            for ref in cross_refs:
                # Find if reference is in our document set
                ref_id = ref if isinstance(ref, (int, str)) else ref.get('id')
                if ref_id in id_to_idx:
                    j = id_to_idx[ref_id]
                    if i != j:
                        edges.append((i, j))
                        weights.append(1.0)

        # Add edges based on regulation type similarity
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                doc_i = documents[i].get('record', documents[i])
                doc_j = documents[j].get('record', documents[j])

                # Same regulation type = edge
                type_i = doc_i.get('regulation_type', '')
                type_j = doc_j.get('regulation_type', '')

                if type_i and type_i == type_j:
                    if (i, j) not in edges and (j, i) not in edges:
                        edges.append((i, j))
                        weights.append(0.5)

                # Same domain = edge
                domain_i = doc_i.get('kg_primary_domain', '')
                domain_j = doc_j.get('kg_primary_domain', '')

                if domain_i and domain_i == domain_j:
                    if (i, j) not in edges and (j, i) not in edges:
                        edges.append((i, j))
                        weights.append(0.7)

        if not edges:
            # No connections, return single community
            return [{
                'id': 0,
                'size': n_docs,
                'members': doc_ids,
                'dominant_type': self._get_dominant_type(documents),
                'themes': self._extract_themes(documents)
            }]

        # Create igraph graph
        g = ig.Graph(n=n_docs, edges=edges, directed=False)
        g.es['weight'] = weights if weights else [1.0] * len(edges)

        # Run Louvain community detection
        communities = g.community_multilevel(weights='weight', resolution=self.resolution)

        # Process communities
        result = []
        for comm_id, members in enumerate(communities):
            if len(members) >= self.min_community_size:
                member_docs = [documents[i] for i in members]
                member_ids = [doc_ids[i] for i in members]

                result.append({
                    'id': comm_id,
                    'size': len(members),
                    'members': member_ids,
                    'dominant_type': self._get_dominant_type(member_docs),
                    'themes': self._extract_themes(member_docs),
                    'modularity': communities.modularity
                })

        # Sort by size
        result.sort(key=lambda x: x['size'], reverse=True)

        logger.info("Community detection completed", {
            "total_docs": n_docs,
            "communities": len(result),
            "modularity": f"{communities.modularity:.3f}" if result else "N/A"
        })

        return result

    def _detect_simplified(
        self,
        documents: List[Dict[str, Any]],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Simplified clustering without igraph (fallback)"""

        # Group by regulation type
        type_groups = defaultdict(list)

        for i, doc in enumerate(documents):
            record = doc.get('record', doc)
            reg_type = record.get('regulation_type', 'Unknown')
            type_groups[reg_type].append(i)

        # Create communities from groups
        doc_ids = [d.get('global_id', i) for i, d in enumerate(documents)]
        result = []

        for comm_id, (reg_type, indices) in enumerate(type_groups.items()):
            if len(indices) >= self.min_community_size:
                member_docs = [documents[i] for i in indices]
                member_ids = [doc_ids[i] for i in indices]

                result.append({
                    'id': comm_id,
                    'size': len(indices),
                    'members': member_ids,
                    'dominant_type': reg_type,
                    'themes': self._extract_themes(member_docs)
                })

        # Sort by size
        result.sort(key=lambda x: x['size'], reverse=True)

        logger.info("Simplified clustering completed", {
            "total_docs": len(documents),
            "communities": len(result)
        })

        return result

    def _get_dominant_type(self, documents: List[Dict[str, Any]]) -> str:
        """Get most common regulation type in document set"""
        type_counts = defaultdict(int)

        for doc in documents:
            record = doc.get('record', doc)
            reg_type = record.get('regulation_type', 'Unknown')
            type_counts[reg_type] += 1

        if not type_counts:
            return 'Unknown'

        return max(type_counts.items(), key=lambda x: x[1])[0]

    def _extract_themes(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from document set"""
        domain_counts = defaultdict(int)

        for doc in documents:
            record = doc.get('record', doc)
            domain = record.get('kg_primary_domain', '')
            if domain:
                domain_counts[domain] += 1

        # Return top 3 domains
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        return [d[0] for d in sorted_domains[:3]]

    def format_clusters_for_display(self, communities: List[Dict[str, Any]]) -> str:
        """Format community detection results for UI display"""
        if not communities:
            return ""

        parts = ["### 🌐 Discovered Thematic Clusters\n"]

        for i, comm in enumerate(communities[:5], 1):  # Top 5 clusters
            themes = ", ".join(comm['themes']) if comm['themes'] else "General"
            parts.append(
                f"• **Cluster {i}** ({comm['size']} docs): "
                f"{comm['dominant_type']} - {themes}\n"
            )

        return "\n".join(parts)
