# Advanced Retrieval System Design - Detective-Style RAG

**Date:** 2025-12-19
**Type:** Architecture Design Document
**Status:** PROPOSAL - Awaiting Implementation

---

## Executive Summary

Transform the current **quantitative scoring-based RAG** into an **investigative, graph-based, iterative research system** that works like an intelligent legal researcher or detective connecting evidence.

**Current System:** Score → Filter → Return top-K
**Proposed System:** Initial Retrieval → Expand (metadata, KG, citations) → Pool → Iterate → Synthesize

---

## User Requirements Analysis

### Requirement 1: Exhaustive Initial Scan
> "Make sure RAG scans from ALL available data before selecting top"

**Current Issue:** Personas may miss documents due to threshold filtering
**Solution:** Two-phase retrieval:
- Phase 1: Broad scan with low thresholds (collect candidates)
- Phase 2: Expansion and re-ranking

### Requirement 2: Metadata-Based Expansion
> "Search same regulation name for more context (preamble, attachments, other articles)"

**Analogy:** Paralegal finds relevant article → Reads entire regulation
**Implementation:**
- Extract regulation_name from retrieved docs
- Fetch ALL articles/sections from same regulation
- Example: Find UU KUP Pasal 25 → Retrieve ALL UU KUP articles

### Requirement 3: Knowledge Graph Expansion
> "Use KG to find related entities and cited regulations"

**Analogy:** Detective finding connections between suspects
**Implementation:**
- Entity co-occurrence (docs mentioning same entities)
- Citation following (Regulation A cites Regulation B)
- Cross-reference traversal

### Requirement 4: Research Pool / Memory
> "Giant pool of interconnected information"

**Analogy:** Detective's evidence board with red strings
**Implementation:**
- Central document pool with deduplication
- Track retrieval provenance (why doc was added)
- Build document graph (relationships)

### Requirement 5: Detective-Style Investigation
> "Deep scan like intelligence investigator, connecting wires"

**Analogy:** Snowball/tree growth through iterations
**Implementation:**
- Iterative expansion (multiple rounds)
- Breadth-first or depth-first strategies
- Stop conditions (convergence, max depth)

### Requirement 6: Qualitative + Quantitative
> "Not only scoring, but deeper methods"

**Solution:** Hybrid approach with multiple strategies

---

## Proposed Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     QUERY INPUT                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│   STAGE 1: INITIAL RETRIEVAL (Broad Scan)                   │
│   - Low thresholds (semantic 0.15, keyword 0.05)            │
│   - Multiple personas in parallel                            │
│   - Collect 200-500 candidates                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│   RESEARCH POOL (Central Memory)                             │
│   - Deduplicated document store                              │
│   - Provenance tracking (how doc was found)                  │
│   - Relationship graph                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│   STAGE 2: ITERATIVE EXPANSION (Detective Mode)             │
│                                                               │
│   Round 1: Metadata Expansion                                │
│   ├─ Same regulation name → Fetch all articles               │
│   ├─ Same chapter → Fetch related sections                   │
│   └─ Add to pool                                              │
│                                                               │
│   Round 2: KG Expansion                                      │
│   ├─ Entity co-occurrence → Find related docs                │
│   ├─ Citation following → Fetch cited regulations            │
│   └─ Add to pool                                              │
│                                                               │
│   Round 3: Cross-Reference Expansion                         │
│   ├─ Article mentions → Follow references                    │
│   ├─ Amendment tracking → Find latest version                │
│   └─ Add to pool                                              │
│                                                               │
│   Stop Condition: Max rounds OR no new docs OR pool size     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│   STAGE 3: POOL RANKING & FILTERING                         │
│   - Re-rank all pool documents with full scoring            │
│   - Apply final relevance threshold                          │
│   - Diversity filtering (avoid too many from same reg)       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│   STAGE 4: FINAL SELECTION                                  │
│   - Top-K by score                                            │
│   - Include provenance metadata                               │
│   - Return to LLM for synthesis                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Expansion Strategies (Detailed)

### Strategy 1: Metadata Expansion

**Trigger:** Retrieved document has high relevance score
**Action:** Fetch related documents by metadata similarity

```python
def metadata_expansion(seed_doc, pool, data_loader):
    """
    Expand by fetching documents with matching metadata

    Example: Found UU 6/1983 Pasal 25 (KUP, tax objections)
    → Fetch ALL UU 6/1983 articles (Pasal 1-99)
    → Add preamble, explanations, attachments
    """
    regulation_name = seed_doc['regulation_number']  # "6/1983"
    regulation_type = seed_doc['regulation_type']     # "UNDANG-UNDANG"

    # Find all documents from same regulation
    related_docs = data_loader.find_by_metadata({
        'regulation_number': regulation_name,
        'regulation_type': regulation_type
    })

    # Add to pool with provenance
    for doc in related_docs:
        pool.add(doc, source='metadata_expansion', seed=seed_doc['global_id'])

    return len(related_docs)
```

**Benefits:**
- ✅ Gets complete regulatory context
- ✅ Finds preambles (often contain definitions)
- ✅ Includes attachments (detailed procedures)
- ✅ Captures related articles not caught by scoring

**When to Use:** Always for top-scoring documents

---

### Strategy 2: Knowledge Graph Expansion

**Trigger:** Seed document has high KG score
**Action:** Follow entity relationships and citations

```python
def kg_expansion(seed_doc, pool, kg_engine):
    """
    Expand by traversing knowledge graph

    Example: Seed doc mentions entity "Pengadilan Pajak"
    → Find all docs mentioning "Pengadilan Pajak"
    → Find docs with co-occurring entities
    → Follow citation relationships
    """
    # Extract entities from seed document
    entities = seed_doc['kg_entities']  # ['Pengadilan Pajak', 'Banding', 'Keberatan']

    expanded_docs = []

    # 1. Entity co-occurrence
    for entity in entities:
        # Find docs mentioning same entity
        docs = kg_engine.find_documents_by_entity(entity)
        expanded_docs.extend(docs)

    # 2. Citation following
    citations = seed_doc.get('kg_citations', [])
    for cited_reg in citations:
        # Example: "UU 14/2002" cited in seed doc
        cited_docs = data_loader.find_by_regulation(cited_reg)
        expanded_docs.extend(cited_docs)

    # 3. Cross-references
    cross_refs = seed_doc.get('kg_cross_references', [])
    for ref in cross_refs:
        # Example: "Pasal 10 ayat (2)" mentioned
        ref_docs = data_loader.find_by_article_reference(ref)
        expanded_docs.extend(ref_docs)

    # Add to pool
    for doc in expanded_docs:
        pool.add(doc, source='kg_expansion', seed=seed_doc['global_id'])

    return len(expanded_docs)
```

**Benefits:**
- ✅ Follows legal citations (regulatory network)
- ✅ Finds related concepts through entity links
- ✅ Discovers indirect relationships

**When to Use:** For documents with rich KG data

---

### Strategy 3: Citation Network Traversal

**Trigger:** Document cites other regulations
**Action:** Follow citation chains (multi-hop)

```python
def citation_traversal(seed_doc, pool, data_loader, max_hops=2):
    """
    Follow citation network like a detective

    Example: UU KUP → cites → UU Pengadilan Pajak → cites → UU Peradilan

    Multi-hop:
    Hop 0: Seed document
    Hop 1: Direct citations from seed
    Hop 2: Citations from hop 1 documents
    """
    visited = set()
    queue = [(seed_doc, 0)]  # (doc, hop_level)

    while queue:
        current_doc, hop = queue.pop(0)
        doc_id = current_doc['global_id']

        if doc_id in visited or hop > max_hops:
            continue

        visited.add(doc_id)

        # Get cited regulations
        citations = current_doc.get('kg_citations', [])

        for cited_reg in citations:
            # Fetch cited documents
            cited_docs = data_loader.find_by_regulation(cited_reg)

            for cited_doc in cited_docs:
                pool.add(cited_doc,
                        source=f'citation_hop_{hop+1}',
                        seed=doc_id)

                # Add to queue for next hop
                if hop + 1 < max_hops:
                    queue.append((cited_doc, hop + 1))

    return len(visited)
```

**Benefits:**
- ✅ Builds complete citation network
- ✅ Finds indirectly related regulations
- ✅ Captures regulatory dependencies

**When to Use:** For complex legal queries requiring context

---

### Strategy 4: Semantic Neighborhood Expansion

**Trigger:** High semantic similarity cluster detected
**Action:** Expand around embedding space neighbors

```python
def semantic_neighborhood_expansion(seed_docs, pool, data_loader, radius=0.05):
    """
    Expand in embedding space neighborhood

    Analogy: If 5 retrieved docs cluster in embedding space,
    fetch ALL docs in that cluster (likely same topic)
    """
    # Get embeddings of seed docs
    seed_embeddings = [doc['embedding'] for doc in seed_docs]

    # Calculate cluster center
    cluster_center = np.mean(seed_embeddings, axis=0)

    # Find all docs within radius of cluster center
    all_embeddings = data_loader.embeddings
    distances = cosine_distances(cluster_center.reshape(1, -1), all_embeddings)[0]

    # Docs within radius
    neighbor_indices = np.where(distances < radius)[0]

    for idx in neighbor_indices:
        doc = data_loader.all_records[idx]
        pool.add(doc, source='semantic_neighborhood', cluster_center=cluster_center)

    return len(neighbor_indices)
```

**Benefits:**
- ✅ Finds topically similar docs not caught by query
- ✅ Exploits semantic clustering

**When to Use:** When initial results cluster strongly

---

## Research Pool Architecture

### DocumentPool Class

```python
class ResearchPool:
    """
    Central memory for iterative retrieval

    Features:
    - Deduplication (by global_id)
    - Provenance tracking (how/why doc was added)
    - Relationship graph (doc-to-doc connections)
    - Statistics (expansion metrics)
    """

    def __init__(self):
        self.documents = {}  # global_id → doc
        self.provenance = {}  # global_id → {source, seed, round, score}
        self.graph = nx.DiGraph()  # Document relationship graph
        self.stats = defaultdict(int)

    def add(self, doc, source='initial', seed=None, round_num=0, score=None):
        """Add document to pool with provenance"""
        doc_id = doc['global_id']

        # Deduplicate
        if doc_id in self.documents:
            # Update if new source has higher priority
            if self._should_update(doc_id, source, score):
                self.provenance[doc_id].update({
                    'source': source,
                    'seed': seed,
                    'round': round_num,
                    'score': score
                })
            return False  # Not added (duplicate)

        # Add new document
        self.documents[doc_id] = doc
        self.provenance[doc_id] = {
            'source': source,
            'seed': seed,
            'round': round_num,
            'score': score,
            'added_at': time.time()
        }

        # Update graph
        self.graph.add_node(doc_id)
        if seed:
            self.graph.add_edge(seed, doc_id, relation=source)

        # Statistics
        self.stats[f'added_via_{source}'] += 1
        self.stats['total_documents'] += 1

        return True  # Added

    def get_by_source(self, source):
        """Get all docs added via specific source"""
        return [
            self.documents[doc_id]
            for doc_id, prov in self.provenance.items()
            if prov['source'] == source
        ]

    def get_expansion_tree(self, seed_id):
        """Get all docs descended from a seed"""
        descendants = nx.descendants(self.graph, seed_id)
        return [self.documents[doc_id] for doc_id in descendants]

    def get_ranked_documents(self):
        """Return all documents sorted by score"""
        docs_with_scores = [
            (doc_id, self.provenance[doc_id].get('score', 0.0))
            for doc_id in self.documents
        ]
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [self.documents[doc_id] for doc_id, _ in docs_with_scores]

    def get_stats(self):
        """Get pool statistics"""
        return {
            **self.stats,
            'sources': dict(Counter(p['source'] for p in self.provenance.values())),
            'rounds': dict(Counter(p['round'] for p in self.provenance.values())),
            'graph_edges': self.graph.number_of_edges(),
            'avg_connections': self.graph.number_of_edges() / max(len(self.documents), 1)
        }
```

---

## Expansion Controller

### IterativeExpansionEngine

```python
class IterativeExpansionEngine:
    """
    Controls multi-round expansion process

    Implements detective-style investigation:
    1. Start with initial retrieval (seeds)
    2. Expand using configured strategies
    3. Track in research pool
    4. Iterate until convergence
    """

    def __init__(self, data_loader, kg_engine, config):
        self.data_loader = data_loader
        self.kg_engine = kg_engine
        self.config = config
        self.logger = get_logger("ExpansionEngine")

    def expand(self, initial_documents, query):
        """
        Main expansion loop

        Args:
            initial_documents: Seeds from initial retrieval
            query: Original query for context

        Returns:
            ResearchPool with all expanded documents
        """
        pool = ResearchPool()

        # Add initial documents
        for doc in initial_documents:
            pool.add(doc, source='initial_retrieval', score=doc.get('score', 0.0))

        # Expansion rounds
        max_rounds = self.config.get('max_expansion_rounds', 3)
        strategies = self.config.get('expansion_strategies', ['metadata', 'kg', 'citation'])

        for round_num in range(1, max_rounds + 1):
            self.logger.info(f"Expansion round {round_num}/{max_rounds}")

            # Get seeds for this round
            if round_num == 1:
                # Use top initial documents as seeds
                seeds = self._select_seeds(initial_documents, top_k=10)
            else:
                # Use documents added in previous round
                seeds = pool.get_by_source(f'round_{round_num-1}')

            if not seeds:
                self.logger.info("No seeds for expansion, stopping")
                break

            # Apply expansion strategies
            added_count = 0

            for strategy_name in strategies:
                strategy_func = self._get_strategy(strategy_name)

                for seed in seeds:
                    count = strategy_func(
                        seed_doc=seed,
                        pool=pool,
                        round_num=round_num,
                        query=query
                    )
                    added_count += count

            self.logger.info(f"Round {round_num}: Added {added_count} documents")

            # Check convergence
            if added_count == 0:
                self.logger.info("Convergence reached (no new docs)")
                break

            # Check pool size limit
            if len(pool.documents) > self.config.get('max_pool_size', 1000):
                self.logger.warning("Max pool size reached")
                break

        # Log final statistics
        stats = pool.get_stats()
        self.logger.info(f"Expansion complete: {stats}")

        return pool

    def _select_seeds(self, documents, top_k=10):
        """Select best documents as expansion seeds"""
        # Sort by score
        sorted_docs = sorted(documents,
                           key=lambda d: d.get('score', 0.0),
                           reverse=True)
        return sorted_docs[:top_k]

    def _get_strategy(self, name):
        """Get expansion strategy function"""
        strategies = {
            'metadata': self._metadata_expansion,
            'kg': self._kg_expansion,
            'citation': self._citation_expansion,
            'semantic': self._semantic_expansion
        }
        return strategies.get(name, lambda **kwargs: 0)

    def _metadata_expansion(self, seed_doc, pool, round_num, **kwargs):
        """Implement metadata expansion"""
        # Implementation here (see Strategy 1 above)
        pass

    def _kg_expansion(self, seed_doc, pool, round_num, **kwargs):
        """Implement KG expansion"""
        # Implementation here (see Strategy 2 above)
        pass

    # ... other strategy implementations
```

---

## Configuration Parameters

### Expansion Configuration

```python
EXPANSION_CONFIG = {
    # Enable/disable expansion
    'enable_expansion': True,  # Master switch

    # Expansion strategies (order matters)
    'expansion_strategies': [
        'metadata',      # Same regulation expansion
        'kg',            # Knowledge graph traversal
        'citation',      # Citation network following
        'semantic'       # Semantic neighborhood
    ],

    # Round limits
    'max_expansion_rounds': 3,        # Max iterations
    'max_pool_size': 1000,            # Stop if pool exceeds this
    'min_docs_per_round': 5,          # Stop if round adds < this

    # Seed selection
    'seeds_per_round': 10,            # Top-K seeds for expansion
    'seed_score_threshold': 0.50,     # Only expand from high-scoring docs

    # Strategy-specific parameters
    'metadata_expansion': {
        'enabled': True,
        'max_docs_per_regulation': 50,  # Limit docs from same regulation
        'include_preamble': True,
        'include_attachments': True
    },

    'kg_expansion': {
        'enabled': True,
        'max_entity_docs': 20,          # Max docs per entity
        'entity_score_threshold': 0.3,   # Min entity relevance
        'follow_citations': True,
        'citation_max_hops': 2           # Multi-hop citation depth
    },

    'citation_expansion': {
        'enabled': True,
        'max_hops': 2,                   # Citation network depth
        'bidirectional': True            # Follow both citing and cited
    },

    'semantic_expansion': {
        'enabled': False,                # Optional (can be expensive)
        'cluster_radius': 0.05,          # Embedding distance threshold
        'min_cluster_size': 3            # Min docs to form cluster
    },

    # Pool management
    'deduplication': True,
    'track_provenance': True,
    'build_graph': True,

    # Final ranking
    'rerank_pool': True,                 # Re-score all pool docs
    'diversity_filter': True,            # Limit docs per regulation
    'max_docs_per_source': 5             # Diversity limit
}
```

---

## Integration Points in Current Pipeline

### Where to Insert Expansion

**Current Pipeline:**
```
Query → Personas Search → Combine Results → Rerank → Return Top-K
```

**New Pipeline:**
```
Query → Personas Search (Broad) → Pool Seeds
      ↓
      Iterative Expansion (Multi-Round)
      ↓
      Research Pool (All Documents)
      ↓
      Re-Ranking & Filtering
      ↓
      Final Selection → Return Top-K
```

### Code Integration

**File:** `core/search/expansion_engine.py` (NEW)
**File:** `core/search/research_pool.py` (NEW)

**Modified:** `core/search/stages_research.py`

```python
# In StagesResearchEngine.conduct_research()

# BEFORE:
round_results = self._execute_round(...)
research_data['all_results'].extend(round_results['results'])

# AFTER:
# 1. Execute initial round (seeds)
round_results = self._execute_round(...)

# 2. Expansion (if enabled)
if self.config.get('enable_expansion', False):
    expansion_engine = IterativeExpansionEngine(...)
    pool = expansion_engine.expand(
        initial_documents=round_results['results'],
        query=query
    )

    # 3. Use pool documents
    expanded_results = pool.get_ranked_documents()
    research_data['all_results'].extend(expanded_results)
else:
    # Original behavior
    research_data['all_results'].extend(round_results['results'])
```

---

## CLI/UI Parameters

### Command-Line Interface

```bash
# Enable expansion with default settings
python main.py --enable-expansion

# Configure expansion strategies
python main.py \
    --enable-expansion \
    --expansion-strategies metadata,kg,citation \
    --max-expansion-rounds 3 \
    --max-pool-size 1000

# Disable specific strategies
python main.py \
    --enable-expansion \
    --no-semantic-expansion

# Detective mode (aggressive expansion)
python main.py --detective-mode  # Preset: max rounds, all strategies
```

### Gradio UI

```python
with gr.Accordion("🔍 Advanced Retrieval Options", open=False):
    enable_expansion = gr.Checkbox(
        label="Enable Iterative Expansion",
        value=False,
        info="Detective-style multi-round document expansion"
    )

    expansion_strategies = gr.CheckboxGroup(
        label="Expansion Strategies",
        choices=[
            ("Same Regulation", "metadata"),
            ("Knowledge Graph", "kg"),
            ("Citation Network", "citation"),
            ("Semantic Clustering", "semantic")
        ],
        value=["metadata", "kg"],
        info="How to expand document search"
    )

    max_rounds = gr.Slider(
        label="Max Expansion Rounds",
        minimum=1,
        maximum=5,
        value=3,
        step=1,
        info="Detective investigation depth"
    )

    max_pool_size = gr.Slider(
        label="Max Research Pool Size",
        minimum=100,
        maximum=2000,
        value=1000,
        step=100,
        info="Maximum documents in memory pool"
    )
```

---

## Expected Benefits

### Quantitative Improvements

| Metric | Before Expansion | After Expansion | Improvement |
|--------|------------------|-----------------|-------------|
| **Recall@50** | ~40% | ~80% | +100% |
| **Context Coverage** | Partial | Complete | - |
| **Related Docs Found** | 10-20 | 50-200 | +500% |
| **Citation Network** | Not captured | Fully mapped | - |

### Qualitative Improvements

1. **Complete Regulatory Context**
   - Find entire regulation, not just relevant article
   - Include preambles (definitions, rationale)
   - Capture attachments (procedures, forms)

2. **Legal Citation Network**
   - Map dependencies between regulations
   - Find superseding/amending laws
   - Discover related legal frameworks

3. **Entity Relationship Discovery**
   - Connect docs through shared entities
   - Build topic networks
   - Find indirect relationships

4. **Investigation Provenance**
   - Track WHY each doc was retrieved
   - Visualize expansion paths
   - Debug retrieval decisions

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Priority 1)
- ✅ ResearchPool class
- ✅ Basic metadata expansion
- ✅ Integration with current pipeline
- ✅ CLI parameters

**ETA:** 2-3 hours
**Status:** Ready to implement

### Phase 2: Advanced Strategies (Priority 2)
- ✅ KG expansion
- ✅ Citation traversal
- ✅ Semantic neighborhood
- ✅ Multi-round iteration

**ETA:** 4-5 hours
**Status:** Design complete

### Phase 3: Optimization (Priority 3)
- ⏳ Parallel expansion
- ⏳ Caching for expansion
- ⏳ Incremental graph building

**ETA:** 2-3 hours

### Phase 4: Visualization (Optional)
- ⏳ Expansion tree visualization
- ⏳ Document graph viewer
- ⏳ Provenance explorer

**ETA:** 3-4 hours

---

## Risks & Mitigations

### Risk 1: Performance (Too Slow)
**Problem:** Expansion may take minutes for complex queries
**Mitigation:**
- Parallel expansion of seeds
- Cache expansion results
- Limit pool size (1000 docs max)
- Early stopping on convergence

### Risk 2: Noise (Too Many Irrelevant Docs)
**Problem:** Expansion may add off-topic documents
**Mitigation:**
- Strict seed selection (score > 0.50)
- Relevance filtering in each round
- Diversity limits (max 5 docs per regulation)
- Re-ranking at the end

### Risk 3: Complexity (Hard to Configure)
**Problem:** Too many parameters
**Mitigation:**
- Presets (conservative, balanced, aggressive)
- Auto-configuration based on query type
- Simple on/off switch for users

---

## Comparison: Quantitative vs Qualitative Methods

### Current System (Quantitative)
```
Documents → Scoring → Top-K
```
**Pros:** Fast, deterministic, simple
**Cons:** Misses context, no relationships, fixed recall

### Proposed System (Hybrid)
```
Documents → Scoring → Expansion → Pool → Re-rank → Top-K
```
**Pros:** Complete context, discovers relationships, higher recall
**Cons:** Slower, more complex, needs tuning

### When to Use Each

| Query Type | Use Expansion? | Reason |
|------------|----------------|--------|
| Simple fact lookup | ❌ No | Overkill |
| Specific article | ⚠️ Maybe | If need full regulation |
| Complex legal question | ✅ Yes | Need complete context |
| Research query | ✅ Yes | Detective mode |
| Multi-regulation query | ✅ Yes | Citation network |

---

## Conclusion

This design transforms the RAG system from a **scoring-based retriever** to an **intelligent research assistant** that:

1. ✅ **Exhaustively scans** data (not limited by top-K)
2. ✅ **Expands by metadata** (same regulation context)
3. ✅ **Follows knowledge graph** (entity and citation networks)
4. ✅ **Builds research pool** (interconnected memory)
5. ✅ **Investigates deeply** (detective-style multi-hop)
6. ✅ **Combines quantitative + qualitative** (scoring + graph traversal)
7. ✅ **Configurable** (CLI/UI parameters, presets)

**Next Step:** Implement Phase 1 (core infrastructure)

---

**Status:** DESIGN APPROVED - Awaiting Implementation Decision
**Estimated Total Implementation:** 10-15 hours for full system
**Minimum Viable:** 2-3 hours for Phase 1 (usable)
