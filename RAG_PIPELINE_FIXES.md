# RAG Pipeline Fix Summary

## ✅ All Changes Implemented

### Files Modified

1. **`core/search/consensus.py`** (Lines 120-225)
   - **Fix:** Fallback now uses `final_top_k` from config instead of hard-coded `3`
   - **Fix:** Added emergency last-resort fallback that takes top-N documents by raw score if even 0.0 threshold fails
   - **Impact:** Should ensure we always return `final_top_k` documents (3 in your case)

2. **`core/search/reranking.py`** (Lines 259-271)
   - **Fix:** Added defensive warning when reranker can't meet top_k due to insufficient candidates from consensus
   - **Impact:** Better visibility into why we get fewer documents than expected

3. **`core/search/langgraph_orchestrator.py`** (Lines 314-332)
   - **Fix:** Added validation to detect when pipeline returns fewer documents than expected
   - **Fix:** Logs recommendation to lower consensus_threshold if this happens
   - **Impact:** Helps diagnose configuration issues

---

## What Changed in Detail

### Before Fix
```python
# consensus.py line 182 (OLD)
if len(consensus_data['validated_results']) >= 3:  # Hard-coded!
    break
```

### After Fix
```python
# consensus.py line 122-124 (NEW)
required_docs = self.config.get('final_top_k', 3)  # Use config value

# consensus.py line 183-188 (NEW)  
if len(consensus_data['validated_results']) >= required_docs:  # Dynamic!
    self.logger.info(f"Fallback succeeded with threshold {fallback_threshold:.0%}", {
        "results": len(consensus_data['validated_results']),
        "required": required_docs
    })
    break
```

### Emergency Fallback (NEW)
```python
# consensus.py lines 191-221 (NEW)
if len(consensus_data['validated_results']) == 0:
    self.logger.error("FALLBACK FAILED - No results even with 0% threshold!")
    self.logger.warning(f"LAST RESORT: Taking top {required_docs} documents by raw score")
    
    # Collect all documents with their best scores
    all_docs_scores = []
    for global_id, doc_results in results_by_doc.items():
        best_result = max(doc_results, key=lambda x: x['scores']['final'])
        all_docs_scores.append({...})  # Create consensus result
    
    # Sort by score and take top-N
    all_docs_scores.sort(key=lambda x: x['consensus_score'], reverse=True)
    consensus_data['validated_results'] = all_docs_scores[:required_docs]
```

---

## Expected Behavior After Fix

### Test Case 1: Your Original Query
**Query:** "Apakah terdapat pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?"

**Expected Log Output:**
```
[2025-12-20 XX:XX:XX] ✅ [SUCCESS] [StagesResearch] Multi-stage research completed | rounds=2 | unique_results=407
[2025-12-20 XX:XX:XX] ⚠️  [WARNING] [ConsensusBuilder] NO RESULTS PASSED CONSENSUS!
[2025-12-20 XX:XX:XX] ℹ️  [INFO] [ConsensusBuilder] Trying fallback threshold: 50%
[2025-12-20 XX:XX:XX] ℹ️  [INFO] [ConsensusBuilder] Trying fallback threshold: 40%
[2025-12-20 XX:XX:XX] ℹ️  [INFO] [ConsensusBuilder] Fallback succeeded with threshold 40% | results=3 | required=3
[2025-12-20 XX:XX:XX] ✅ [SUCCESS] [ConsensusBuilder] Consensus building completed | validated_results=3
[2025-12-20 XX:XX:XX] ✅ [SUCCESS] [RerankerEngine] Reranking completed | reranked_count=3
[2025-12-20 XX:XX:XX] ✅ [SUCCESS] [RAGOrchestrator] Reranking completed | final_count=3 | expected_top_k=3
```

**Expected Output:**
- 3 documents (not 1) ✅
- Documents about teacher/lecturer regulations (not budget) - hopefully ✅
- Better relevance scores due to having more candidates to choose from

### Test Case 2: If Fallback Still Fails
**If even 0.0 threshold doesn't work:**
```
[2025-12-20 XX:XX:XX] ⚠️  [WARNING] [ConsensusBuilder] LAST RESORT: Taking top 3 documents by raw score
[2025-12-20 XX:XX:XX] ⚠️  [WARNING] [ConsensusBuilder] Emergency fallback: Added 3 documents
```

**What this means:**
- System will take the top 3 documents by score regardless of consensus
- You'll get 3 documents, but they may not be the best quality
- Indicates the search itself may need tuning (semantic thresholds too high)

---

## How to Test

### Quick Test
```bash
cd d:\Antigravity\06_ID_Legal
python main.py --query "Apakah terdapat pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?"
```

### What to Check
1. **Count documents:** Should see `[Dokumen 1]`, `[Dokumen 2]`, `[Dokumen 3]` (not just 1)
2. **Check relevance:** Documents should mention:
   - "guru" (teacher)
   - "dosen" (lecturer)
   - "tunjangan" (allowance/benefit)
   - NOT about "STANDARBIAYATAHUNANGGARAN" (budget)
3. **Check logs:** Look for "Fallback succeeded" or "Emergency fallback" messages
4. **Check final count:** Should see "Reranking completed | final_count=3"

### If Still Getting 1 Document
This would indicate a deeper issue (search not finding enough candidates). In that case, check:
- Is the search returning 407 documents? (should see "unique_results=407")
- Are all 407 being rejected by consensus? (should see "NO RESULTS PASSED CONSENSUS")
- Does emergency fallback trigger? (should see "LAST RESORT: Taking top 3")

If emergency fallback triggers but still returns 1, then `results_by_doc` had only 1 unique document (very unlikely).

---

## Next Steps

1. **Run your test** with the same query
2. **Report results:**
   - How many documents did you get?
   - Are they relevant to teacher/lecturer allowances?
   - What do the logs say about fallback?
3. **If still issues:**
   - Share the full log output
   - We may need to adjust `consensus_threshold` in config (lower from 0.6 to 0.5 or 0.4)
   - Or adjust search phase thresholds

---

## Configuration Tuning (If Needed)

If you still get irrelevant documents after fixing the count issue, you can tune:

### Option 1: Lower Consensus Threshold (Recommended First)
```python
# config.py line 395
DEFAULT_CONFIG = {
    'consensus_threshold': 0.4,  # Changed from 0.6
    # ... rest of config
}
```

### Option 2: Adjust Search Phase Thresholds
```python
# config.py lines 412-413
'initial_scan': {
    'semantic_threshold': 0.20,  # Lower from 0.25 to find more candidates
    'keyword_threshold': 0.08,   # Lower from 0.10
}
```

But test first with current settings - the fix should help!
