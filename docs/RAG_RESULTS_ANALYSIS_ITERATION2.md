# RAG Results Analysis - Second Iteration

**Date:** 2025-12-19 (After first round of fixes)
**Status:** IMPROVED but needs stronger adjustments

---

## Current Results Analysis

### Query
```
Tax objection procedures (keberatan pajak, UU KUP)
```

### Results Summary

| Rank | Score | Document | Relevance |
|------|-------|----------|-----------|
| 1 | 0.727 | UU 1967 - Cooperatives | ❌ IRRELEVANT |
| 2 | 0.726 | UU 1995 - Customs | ⚠️ TAX-ADJACENT |
| 3 | 0.724 | UU 2017 - Budget Standards | ❌ IRRELEVANT |
| **4** | **0.722** | **UU 1997 - Tax Dispute Resolution** | ✅ **RELEVANT** |
| 5 | 0.721 | UU 1999 - Elections | ❌ IRRELEVANT |
| 6 | 0.719 | UU 1957 - Tax Collection | ⚠️ PARTIAL |
| **7** | **0.719** | **UU 1997 - Non-Tax Revenue (keberatan)** | ✅ **RELEVANT** |
| 8 | 0.719 | UU 1994 - WTO Agreement | ❌ IRRELEVANT |
| ... | ... | ... | ... |
| **14** | **0.712** | **PP 80/2007 - Tax Procedures (KUP!)** | ✅ **HIGHLY RELEVANT** |
| **15** | **0.711** | **UU 1997 - Tax Dispute Resolution** | ✅ **RELEVANT** |
| **16** | **0.710** | **UU 21/1997 - Land Tax (keberatan)** | ✅ **RELEVANT** |
| **17** | **0.707** | **UU 12/1985 - Land Tax (keberatan)** | ✅ **RELEVANT** |

### Metrics

- **Precision@3:** 0% (0/3 in top 3)
- **Precision@10:** ~30% (3/10)
- **Precision@20:** ~35% (7/20)
- **Best relevant doc rank:** #4 (should be #1!)

---

## Progress vs Previous

### Before ANY Fixes
```
Top 3: Banking, Land Use, Irrelevant
Precision@3: 0%
Precision@10: ~10%
```

### After First Fixes (Current)
```
Top 3: Cooperatives, Customs, Budget
Precision@3: 0% (but #4 is relevant!)
Precision@10: ~30%
Precision@20: ~35%
```

**Improvement:**
- ✅ Found relevant documents (they exist in results now!)
- ✅ More relevant docs in top 20
- ❌ Still not ranking at the top

---

## Root Cause: Why Relevant Docs Rank #4-17 Instead of #1-3

### Score Analysis

The score differences are **extremely small**:
- Irrelevant #1: 0.727
- Relevant #4: 0.722
- **Difference: Only 0.005 (0.7%!)**

This suggests:

1. **Relevance gating threshold too low (0.15)**
   - Even cooperatives law (irrelevant) has relevance > 0.15
   - Possible reason: Word "keberatan" (objection) appears in cooperative law (member objections)
   - Need higher threshold to filter these out

2. **Semantic embeddings not discriminating well**
   - Cooperatives law and Tax law scoring similarly
   - Embeddings may be capturing "legal document" similarity, not topic similarity
   - All Indonesian legal texts may have similar embedding patterns

3. **Keyword matching weak**
   - TF-IDF may be giving similar scores to different laws
   - Common legal terms ("pasal", "undang-undang", "penetapan") boost all docs
   - Tax-specific terms ("pajak", "keberatan pajak", "KUP") not weighted heavily enough

4. **Relevance weights (65%) still not dominant enough**
   - Metadata (35%) still allows irrelevant docs to compensate
   - Score gap of 0.005 shows weights are too balanced

---

## Why Cooperatives Law Ranks #1

**Hypothesis:** The word "keberatan" (objection) appears in both:
- Tax law: "keberatan pajak" (tax objection) ✅ Relevant
- Cooperatives law: "keberatan anggota" (member objection) ❌ Irrelevant

**Result:**
- Semantic: Moderate (both about objections, legal documents)
- Keyword: Moderate (contains "keberatan")
- Authority: High (it's an old foundational law)
- Temporal: Lower (1967, old)
- **Final:** Ranks high despite being irrelevant to TAX objections

---

## Solutions Required

### Solution 1: More Aggressive Relevance Gating

**Current:** minimum_relevance_threshold = 0.15 (too lenient)

**Proposed:** 0.20-0.25 (stricter)

```python
# Current (too lenient)
if relevance_score < 0.15:  # Cooperatives law passes
    continue

# Proposed (stricter)
if relevance_score < 0.25:  # Filter out topic-adjacent docs
    continue
```

**Impact:** Filter out documents that mention "keberatan" in wrong context

### Solution 2: Increase Relevance Weight Further

**Current:** 65% relevance, 35% metadata

**Proposed:** 75-80% relevance, 20-25% metadata

```python
DEFAULT_HUMAN_PRIORITIES = {
    'semantic_match': 0.50,       # ↑ from 0.40 (+25%)
    'keyword_precision': 0.30,    # ↑ from 0.25 (+20%)
    # Total: 80% relevance

    'knowledge_graph': 0.10,      # ↓ from 0.15 (-33%)
    'authority_hierarchy': 0.05,  # ↓ from 0.10 (-50%)
    'temporal_relevance': 0.03,   # ↓ from 0.05 (-40%)
    'legal_completeness': 0.02,   # ↓ from 0.05 (-60%)
    # Total: 20% metadata
}
```

**Rationale:** Need relevance to be OVERWHELMING factor

### Solution 3: Boost Tax-Specific Keywords

Add query-pattern-specific keyword boosting:

```python
TAX_KEYWORDS = [
    'pajak', 'keberatan pajak', 'KUP', 'ketentuan umum perpajakan',
    'banding pajak', 'pengadilan pajak', 'wajib pajak', 'surat ketetapan pajak'
]

# Boost keyword score if tax-specific terms match
if is_tax_query and has_tax_keywords:
    keyword_score *= 1.5  # 50% boost for tax terms
```

### Solution 4: Domain-Specific Semantic Boost

Check document regulation type:

```python
# If query is about tax AND document is tax regulation
if query_domain == 'tax' and doc_regulation_type in ['UU Pajak', 'PP Pajak']:
    semantic_score *= 1.3  # 30% boost for domain match
```

### Solution 5: Penalize Cross-Domain Documents

If document is clearly from different domain:

```python
LEGAL_DOMAINS = {
    'tax': ['pajak', 'KUP', 'bea', 'cukai'],
    'labor': ['ketenagakerjaan', 'buruh', 'pekerja'],
    'cooperatives': ['koperasi', 'anggota koperasi'],
    'elections': ['pemilu', 'pemilihan umum']
}

# If query domain doesn't match document domain
if query_domain != doc_domain:
    final_score *= 0.7  # 30% penalty for domain mismatch
```

---

## Recommended Action Plan

### Immediate (Now)

1. ✅ **Raise relevance threshold:** 0.15 → 0.25
2. ✅ **Increase relevance weights:** 65% → 80%
3. ✅ **Lower metadata weights:** 35% → 20%

### Short-term (Today)

4. **Add tax keyword boosting**
5. **Add domain mismatch penalty**
6. **Test with multiple query types**

### Medium-term (This Week)

7. **Analyze semantic embeddings:** Why cooperatives ≈ tax?
8. **Improve keyword extraction:** Better TF-IDF or BM25
9. **Add cross-encoder reranking**

---

## Expected Outcome After Stronger Fixes

### Current
```
Top 3: Cooperatives (0.727), Customs (0.726), Budget (0.724)
```

### After Stronger Fixes (Predicted)
```
Top 3: Tax Dispute (0.78), Tax Procedures (0.75), Land Tax (0.71)
```

**Reasoning:**
- Relevance threshold 0.25 filters cooperatives (relevance ~0.18)
- 80% relevance weight makes tax docs dominate
- Tax keyword boost pushes tax docs higher
- Domain mismatch penalty drops non-tax docs

---

## Risk Assessment

**Risk:** Might filter TOO aggressively, losing some relevant docs

**Mitigation:**
- Test with diverse queries (tax, labor, criminal law)
- Monitor number of results returned
- If zero results, threshold too high
- Adjust based on feedback

**Rollback Plan:**
- Keep previous config values in comments
- Easy to revert if needed

---

**Status:** Ready to implement stronger fixes
**Priority:** HIGH - System still not production-ready
**ETA:** 30 minutes for implementation + testing
