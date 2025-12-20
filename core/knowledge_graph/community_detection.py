"""
Community Detection - Document Clustering

Identifies communities/clusters of related legal documents.
"""

from typing import Dict, Any, List, Optional, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_utils import get_logger

logger = get_logger(__name__)

try:
    import networkx as nx
    from networkx.algorithms import community
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class CommunityDetector:
    """
    Detect communities/clusters in legal document graph

    Uses graph-based algorithms to identify related document groups.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.communities: List[Set[str]] = []

    def detect_communities(
        self,
        graph,
        method: str = 'louvain'
    ) -> List[Set[str]]:
        """
        Detect communities in document graph

        Args:
            graph: NetworkX graph or RelationshipGraph
            method: Detection method ('louvain', 'greedy', 'label_propagation')

        Returns:
            List of community sets (each set contains document IDs)
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX required for community detection")
            return []

        # Get underlying graph
        if hasattr(graph, 'graph'):
            G = graph.graph
        else:
            G = graph

        if G is None or G.number_of_nodes() == 0:
            return []

        # Convert to undirected for community detection
        G_undirected = G.to_undirected()

        try:
            if method == 'louvain':
                communities = community.louvain_communities(G_undirected)
            elif method == 'greedy':
                communities = community.greedy_modularity_communities(G_undirected)
            elif method == 'label_propagation':
                communities = community.label_propagation_communities(G_undirected)
            else:
                communities = community.louvain_communities(G_undirected)

            self.communities = [set(c) for c in communities]
            logger.info(f"Detected {len(self.communities)} communities")

            return self.communities

        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return []

    def get_document_community(self, doc_id: str) -> Optional[int]:
        """
        Get community index for a document

        Args:
            doc_id: Document ID

        Returns:
            Community index or None if not found
        """
        for i, comm in enumerate(self.communities):
            if doc_id in comm:
                return i
        return None

    def get_community_members(self, community_index: int) -> Set[str]:
        """
        Get all documents in a community

        Args:
            community_index: Community index

        Returns:
            Set of document IDs
        """
        if 0 <= community_index < len(self.communities):
            return self.communities[community_index]
        return set()

    def get_community_stats(self) -> Dict[str, Any]:
        """Get community statistics"""
        if not self.communities:
            return {'num_communities': 0}

        sizes = [len(c) for c in self.communities]

        return {
            'num_communities': len(self.communities),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'avg_size': sum(sizes) / len(sizes),
            'sizes': sizes
        }

    def expand_query_with_community(
        self,
        doc_ids: List[str],
        max_expansion: int = 5
    ) -> List[str]:
        """
        Expand document list with community members

        Args:
            doc_ids: Initial document IDs
            max_expansion: Maximum documents to add

        Returns:
            Expanded document list
        """
        expanded = set(doc_ids)

        for doc_id in doc_ids:
            comm_idx = self.get_document_community(doc_id)
            if comm_idx is not None:
                members = self.get_community_members(comm_idx)
                for member in members:
                    if len(expanded) >= len(doc_ids) + max_expansion:
                        break
                    expanded.add(member)

        return list(expanded)

    def get_inter_community_documents(
        self,
        graph,
        top_k: int = 10
    ) -> List[str]:
        """
        Find documents that bridge multiple communities

        Args:
            graph: Document graph
            top_k: Number of top bridging documents

        Returns:
            List of bridging document IDs
        """
        if not HAS_NETWORKX or not self.communities:
            return []

        if hasattr(graph, 'graph'):
            G = graph.graph
        else:
            G = graph

        if G is None:
            return []

        # Calculate betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G)
            sorted_docs = sorted(
                betweenness.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [doc_id for doc_id, _ in sorted_docs[:top_k]]

        except Exception as e:
            logger.error(f"Betweenness calculation failed: {e}")
            return []
