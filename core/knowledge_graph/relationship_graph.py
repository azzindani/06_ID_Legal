"""
Relationship Graph - Legal Document Network Analysis

Builds and analyzes relationships between legal documents.
"""

from typing import Dict, Any, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_utils import get_logger

logger = get_logger(__name__)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("NetworkX not installed. Graph features limited.")


class RelationshipGraph:
    """
    Graph-based relationship analysis for legal documents

    Builds a network of document relationships based on:
    - Citation references
    - Shared entities
    - Regulatory hierarchy
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
            self._adjacency = {}

    def add_document(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """
        Add a document node to the graph

        Args:
            doc_id: Document identifier
            metadata: Document metadata
        """
        if self.graph is not None:
            self.graph.add_node(doc_id, **metadata)
        else:
            if doc_id not in self._adjacency:
                self._adjacency[doc_id] = {'metadata': metadata, 'edges': []}

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0
    ) -> None:
        """
        Add a relationship edge between documents

        Args:
            source_id: Source document ID
            target_id: Target document ID
            relationship_type: Type of relationship
            weight: Edge weight
        """
        if self.graph is not None:
            self.graph.add_edge(
                source_id,
                target_id,
                type=relationship_type,
                weight=weight
            )
        else:
            if source_id in self._adjacency:
                self._adjacency[source_id]['edges'].append({
                    'target': target_id,
                    'type': relationship_type,
                    'weight': weight
                })

    def build_from_documents(
        self,
        documents: List[Dict[str, Any]],
        kg_core
    ) -> None:
        """
        Build graph from document collection

        Args:
            documents: List of documents
            kg_core: KnowledgeGraphCore instance for entity extraction
        """
        logger.info(f"Building relationship graph from {len(documents)} documents")

        # Add all documents as nodes
        for doc in documents:
            doc_id = doc.get('id', str(hash(doc.get('content', '')[:100])))
            self.add_document(doc_id, {
                'regulation_type': doc.get('regulation_type', ''),
                'regulation_number': doc.get('regulation_number', ''),
                'year': doc.get('year', ''),
                'about': doc.get('about', '')
            })

        # Build relationships based on entity overlap
        for i, doc1 in enumerate(documents):
            doc1_id = doc1.get('id', str(hash(doc1.get('content', '')[:100])))
            entities1 = kg_core.extract_entities(doc1.get('content', ''))

            for j, doc2 in enumerate(documents):
                if i >= j:
                    continue

                doc2_id = doc2.get('id', str(hash(doc2.get('content', '')[:100])))
                entities2 = kg_core.extract_entities(doc2.get('content', ''))

                # Check for shared regulations
                shared_regs = set(entities1.get('regulation', [])) & set(entities2.get('regulation', []))
                if shared_regs:
                    self.add_relationship(
                        doc1_id, doc2_id,
                        'cites_same_regulation',
                        weight=len(shared_regs)
                    )

        logger.info("Relationship graph built successfully")

    def get_related_documents(
        self,
        doc_id: str,
        max_depth: int = 2,
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get documents related to a given document

        Args:
            doc_id: Source document ID
            max_depth: Maximum traversal depth
            max_results: Maximum results to return

        Returns:
            List of (doc_id, relevance_score) tuples
        """
        if self.graph is None:
            return self._get_related_simple(doc_id, max_results)

        if doc_id not in self.graph:
            return []

        # Use personalized PageRank for relevance
        try:
            personalization = {doc_id: 1.0}
            scores = nx.pagerank(
                self.graph,
                personalization=personalization,
                max_iter=100
            )

            # Sort by score and exclude source
            related = [
                (node, score) for node, score in scores.items()
                if node != doc_id
            ]
            related.sort(key=lambda x: x[1], reverse=True)

            return related[:max_results]

        except Exception as e:
            logger.error(f"PageRank failed: {e}")
            return []

    def _get_related_simple(self, doc_id: str, max_results: int) -> List[Tuple[str, float]]:
        """Simple fallback for related documents without NetworkX"""
        if doc_id not in self._adjacency:
            return []

        edges = self._adjacency[doc_id]['edges']
        related = [(e['target'], e['weight']) for e in edges]
        related.sort(key=lambda x: x[1], reverse=True)

        return related[:max_results]

    def get_document_centrality(self, doc_id: str) -> float:
        """
        Get centrality score for a document

        Args:
            doc_id: Document ID

        Returns:
            Centrality score
        """
        if self.graph is None or doc_id not in self.graph:
            return 0.0

        try:
            centrality = nx.degree_centrality(self.graph)
            return centrality.get(doc_id, 0.0)
        except Exception:
            return 0.0

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if self.graph is None:
            return {
                'nodes': len(self._adjacency),
                'edges': sum(len(v['edges']) for v in self._adjacency.values())
            }

        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
