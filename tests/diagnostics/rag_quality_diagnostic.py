"""
RAG Quality Diagnostic Tool

Evaluates each component of the RAG pipeline to identify quality issues:
1. Semantic search quality
2. Keyword search quality
3. Knowledge graph scoring
4. Weight combination
5. Reranking impact
6. Document relevance

Usage:
    python tests/diagnostics/rag_quality_diagnostic.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict

print("Loading dependencies...")

class RAGDiagnostic:
    """Diagnostic tool for RAG pipeline evaluation"""

    def __init__(self):
        print("Initializing RAG Diagnostic...")
        self.data_loader = None
        self.embedding_model = None
        self.search_engine = None
        self.results = {}

    def initialize(self):
        """Load models and data"""
        print("\n" + "="*80)
        print("INITIALIZING COMPONENTS")
        print("="*80)

        try:
            # Load data
            print("\n1. Loading dataset...")
            from loader import DataLoader
            self.data_loader = DataLoader()
            self.data_loader.load_data()
            print(f"   ✓ Loaded {len(self.data_loader.all_records):,} documents")

            # Load models
            print("\n2. Loading embedding model...")
            from model_loader import ModelLoader
            model_loader = ModelLoader()
            self.embedding_model = model_loader.load_embedding_model()
            print(f"   ✓ Embedding model loaded")

            print("\n3. Loading reranker model...")
            self.reranker_model = model_loader.load_reranker_model()
            print(f"   ✓ Reranker model loaded")

            # Create search engine
            print("\n4. Creating search engine...")
            from core.search.hybrid_search import HybridSearchEngine
            self.search_engine = HybridSearchEngine(
                data_loader=self.data_loader,
                embedding_model=self.embedding_model,
                reranker_model=self.reranker_model,
                use_faiss=False,  # Disable FAISS for diagnostic (use exact search)
                use_cache=False   # Disable cache to see fresh results
            )
            print(f"   ✓ Search engine initialized (FAISS disabled for diagnostic)")

            return True

        except Exception as e:
            print(f"\n✗ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_dataset_coverage(self, query: str):
        """Analyze if dataset has relevant documents for the query"""
        print("\n" + "="*80)
        print("DATASET COVERAGE ANALYSIS")
        print("="*80)

        print(f"\nQuery: {query}\n")

        # Extract key terms from query
        key_terms = {
            "keberatan pajak": 0,
            "UU KUP": 0,
            "pengadilan pajak": 0,
            "banding": 0,
            "wajib pajak": 0,
            "sanksi pajak": 0,
            "pajak": 0
        }

        regulation_types = defaultdict(int)
        years = defaultdict(int)

        print("Searching for relevant documents in dataset...")
        relevant_docs = []

        for idx, record in enumerate(self.data_loader.all_records):
            title = record.get('regulation_title', '').lower()
            content = record.get('content', '').lower()
            reg_type = record.get('regulation_type', 'Unknown')
            year = record.get('year', 'Unknown')

            regulation_types[reg_type] += 1
            years[year] += 1

            # Check for key terms
            combined_text = title + " " + content
            relevance_score = 0

            for term in key_terms:
                if term in combined_text:
                    key_terms[term] += 1
                    relevance_score += 1

            if relevance_score > 0:
                relevant_docs.append({
                    'index': idx,
                    'global_id': record['global_id'],
                    'regulation_type': reg_type,
                    'regulation_number': record.get('regulation_number', 'N/A'),
                    'year': year,
                    'title': record.get('regulation_title', 'N/A'),
                    'relevance_score': relevance_score,
                    'matched_terms': [term for term in key_terms if term in combined_text]
                })

        # Sort by relevance
        relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)

        print(f"\n📊 Dataset Statistics:")
        print(f"   Total documents: {len(self.data_loader.all_records):,}")
        print(f"   Potentially relevant documents: {len(relevant_docs):,}")
        print(f"   Coverage: {len(relevant_docs)/len(self.data_loader.all_records)*100:.1f}%")

        print(f"\n🔍 Key Term Frequencies:")
        for term, count in sorted(key_terms.items(), key=lambda x: x[1], reverse=True):
            print(f"   '{term}': {count} documents")

        print(f"\n📋 Top 10 Regulation Types:")
        for reg_type, count in sorted(regulation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {reg_type}: {count} documents")

        if relevant_docs:
            print(f"\n✅ Top 10 Most Relevant Documents Found:")
            for i, doc in enumerate(relevant_docs[:10], 1):
                print(f"\n   [{i}] {doc['regulation_type']} No. {doc['regulation_number']}/{doc['year']}")
                print(f"       Relevance: {doc['relevance_score']} terms matched: {', '.join(doc['matched_terms'])}")
                print(f"       Title: {doc['title'][:100]}...")
                print(f"       Index: {doc['index']}, Global ID: {doc['global_id']}")
        else:
            print(f"\n⚠️  WARNING: No relevant documents found for this query!")
            print(f"   Dataset may not contain tax law (UU KUP) documents")

        return relevant_docs

    def diagnose_search_components(self, query: str, top_k: int = 20):
        """Diagnose each search component separately"""
        print("\n" + "="*80)
        print("COMPONENT-BY-COMPONENT DIAGNOSTIC")
        print("="*80)

        print(f"\nQuery: {query}\n")

        # Test semantic search
        print("\n1️⃣  SEMANTIC SEARCH (Embedding-based)")
        print("-" * 80)

        try:
            import torch
            query_text = query.strip()
            query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=True, show_progress_bar=False)

            doc_embeddings = self.data_loader.embeddings

            # Compute cosine similarity
            query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
            doc_embeddings_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
            similarities = torch.mm(query_norm, doc_embeddings_norm.t()).squeeze(0)

            top_k_semantic = min(top_k, len(similarities))
            semantic_scores, semantic_indices = torch.topk(similarities, top_k_semantic)

            print(f"   Top {top_k_semantic} semantic search results:")
            for i in range(min(5, top_k_semantic)):
                idx = semantic_indices[i].item()
                score = semantic_scores[i].item()
                record = self.data_loader.all_records[idx]

                print(f"\n   Rank {i+1}: Score={score:.4f}")
                print(f"   {record['regulation_type']} No. {record['regulation_number']}/{record['year']}")
                print(f"   Title: {record.get('regulation_title', 'N/A')[:80]}...")

            semantic_results = {
                'scores': semantic_scores.cpu().numpy(),
                'indices': semantic_indices.cpu().numpy()
            }

        except Exception as e:
            print(f"   ✗ Semantic search failed: {e}")
            semantic_results = None

        # Test keyword search
        print("\n\n2️⃣  KEYWORD SEARCH (TF-IDF)")
        print("-" * 80)

        try:
            if hasattr(self.search_engine, 'tfidf_vectorizer') and self.search_engine.tfidf_vectorizer:
                query_tfidf = self.search_engine.tfidf_vectorizer.transform([query])
                tfidf_scores = (self.search_engine.tfidf_matrix * query_tfidf.T).toarray().flatten()

                top_k_keyword = min(top_k, len(tfidf_scores))
                keyword_indices = np.argsort(tfidf_scores)[::-1][:top_k_keyword]
                keyword_scores = tfidf_scores[keyword_indices]

                print(f"   Top {top_k_keyword} TF-IDF keyword search results:")
                for i in range(min(5, top_k_keyword)):
                    idx = keyword_indices[i]
                    score = keyword_scores[i]
                    record = self.data_loader.all_records[idx]

                    print(f"\n   Rank {i+1}: Score={score:.4f}")
                    print(f"   {record['regulation_type']} No. {record['regulation_number']}/{record['year']}")
                    print(f"   Title: {record.get('regulation_title', 'N/A')[:80]}...")

                keyword_results = {
                    'scores': keyword_scores,
                    'indices': keyword_indices
                }
            else:
                print("   ✗ TF-IDF not initialized")
                keyword_results = None

        except Exception as e:
            print(f"   ✗ Keyword search failed: {e}")
            keyword_results = None

        # Test hybrid search
        print("\n\n3️⃣  HYBRID SEARCH (Combined)")
        print("-" * 80)

        try:
            results = self.search_engine.search_with_persona(
                query=query,
                persona_name="Legal_Researcher",
                phase_config={'description': 'diagnostic', 'enabled': True},
                priority_weights={
                    'semantic_match': 0.30,
                    'keyword_precision': 0.25,
                    'knowledge_graph': 0.20,
                    'authority_hierarchy': 0.10,
                    'temporal_relevance': 0.10,
                    'legal_completeness': 0.05
                },
                top_k=top_k
            )

            print(f"   Top {min(5, len(results))} hybrid search results:")
            for i, result in enumerate(results[:5], 1):
                record = result['record']
                scores = result['scores']

                print(f"\n   Rank {i}: Final Score={scores['final']:.4f}")
                print(f"   {record['regulation_type']} No. {record['regulation_number']}/{record['year']}")
                print(f"   Title: {record.get('regulation_title', 'N/A')[:80]}...")
                print(f"   Component Scores:")
                print(f"     - Semantic: {scores['semantic']:.4f}")
                print(f"     - Keyword: {scores['keyword']:.4f}")
                print(f"     - KG: {scores['kg']:.4f}")
                print(f"     - Authority: {scores['authority']:.4f}")
                print(f"     - Temporal: {scores['temporal']:.4f}")
                print(f"     - Completeness: {scores['completeness']:.4f}")

            hybrid_results = results

        except Exception as e:
            print(f"   ✗ Hybrid search failed: {e}")
            import traceback
            traceback.print_exc()
            hybrid_results = None

        return {
            'semantic': semantic_results,
            'keyword': keyword_results,
            'hybrid': hybrid_results
        }

    def compare_weight_configurations(self, query: str):
        """Test different weight configurations to find optimal balance"""
        print("\n" + "="*80)
        print("WEIGHT CONFIGURATION ANALYSIS")
        print("="*80)

        weight_configs = {
            "Current (Balanced)": {
                'semantic_match': 0.30,
                'keyword_precision': 0.25,
                'knowledge_graph': 0.20,
                'authority_hierarchy': 0.10,
                'temporal_relevance': 0.10,
                'legal_completeness': 0.05
            },
            "Semantic Heavy": {
                'semantic_match': 0.50,
                'keyword_precision': 0.20,
                'knowledge_graph': 0.15,
                'authority_hierarchy': 0.05,
                'temporal_relevance': 0.05,
                'legal_completeness': 0.05
            },
            "Keyword Heavy": {
                'semantic_match': 0.20,
                'keyword_precision': 0.50,
                'knowledge_graph': 0.15,
                'authority_hierarchy': 0.05,
                'temporal_relevance': 0.05,
                'legal_completeness': 0.05
            },
            "KG Heavy": {
                'semantic_match': 0.20,
                'keyword_precision': 0.20,
                'knowledge_graph': 0.40,
                'authority_hierarchy': 0.08,
                'temporal_relevance': 0.08,
                'legal_completeness': 0.04
            }
        }

        print(f"\nQuery: {query}\n")
        print("Testing different weight configurations...\n")

        for config_name, weights in weight_configs.items():
            print(f"\n{config_name}:")
            print(f"  Weights: {weights}")

            try:
                results = self.search_engine.search_with_persona(
                    query=query,
                    persona_name="Legal_Researcher",
                    phase_config={'description': 'diagnostic', 'enabled': True},
                    priority_weights=weights,
                    top_k=5
                )

                print(f"  Top 3 results:")
                for i, result in enumerate(results[:3], 1):
                    record = result['record']
                    scores = result['scores']
                    print(f"    {i}. [{scores['final']:.3f}] {record['regulation_type']} {record['regulation_number']}/{record['year']}")
                    print(f"       {record.get('regulation_title', 'N/A')[:60]}...")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""

        # Test query (the one reported as problematic)
        test_query = """
        Jelaskan secara lengkap dan komprehensif tentang:
        1. Prosedur pengajuan keberatan pajak menurut UU KUP beserta persyaratan dan jangka waktunya
        2. Sanksi administratif dan pidana yang dapat dikenakan jika terlambat mengajukan keberatan
        3. Hubungan antara keberatan pajak dengan banding di Pengadilan Pajak
        4. Hak-hak wajib pajak selama proses keberatan berlangsung
        5. Contoh kasus dan yurisprudensi terkait keberatan pajak
        """.strip()

        # 1. Dataset coverage analysis
        relevant_docs = self.analyze_dataset_coverage(test_query)

        # 2. Component diagnostic
        search_results = self.diagnose_search_components(test_query)

        # 3. Weight configuration comparison
        self.compare_weight_configurations(test_query)

        # Summary and recommendations
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
        print("="*80)

        if not relevant_docs:
            print("\n⚠️  CRITICAL ISSUE: Dataset does not contain relevant documents!")
            print("   Recommendation: Verify that UU KUP and tax law documents are in the dataset")
        elif len(relevant_docs) < 10:
            print("\n⚠️  WARNING: Very few relevant documents found")
            print(f"   Only {len(relevant_docs)} documents contain tax law keywords")
            print("   Recommendation: Expand dataset to include more tax law documents")
        else:
            print(f"\n✓ Dataset contains {len(relevant_docs)} potentially relevant documents")
            print("   Issue may be in search/ranking algorithms")

        print("\nNext Steps:")
        print("  1. Review semantic embedding quality")
        print("  2. Verify TF-IDF is properly trained on Indonesian legal text")
        print("  3. Check knowledge graph entity extraction")
        print("  4. Tune weight configuration based on above tests")
        print("  5. Consider adding query expansion for Indonesian legal terms")


def main():
    """Run RAG diagnostic"""
    print("\n" + "="*80)
    print("RAG QUALITY DIAGNOSTIC TOOL")
    print("Indonesian Legal RAG System - Accuracy Evaluation")
    print("="*80)

    diagnostic = RAGDiagnostic()

    if not diagnostic.initialize():
        print("\n✗ Initialization failed. Exiting.")
        return 1

    diagnostic.run_full_diagnostic()

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
