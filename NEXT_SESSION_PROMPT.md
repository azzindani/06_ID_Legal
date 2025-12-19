# Session Continuation Prompt for Indonesian Legal RAG System

**Use this prompt to start your next development session.**

---

## For Claude Code / AI Development Assistant

```
I need to continue development on the Indonesian Legal RAG System.

CONTEXT:
- Project: Indonesian Legal RAG System (Knowledge Graph-Enhanced Retrieval-Augmented Generation)
- Location: /home/user/06_ID_Legal
- Branch: claude/review-legal-rag-production-01UPzKEg4RJXiSiC2kzm28dA
- Last Review: December 19, 2025
- Review Document: CODEBASE_REVIEW_2025-12-19.md

CRITICAL INFORMATION:
A comprehensive codebase review was completed on December 19, 2025.
Please READ the review document FIRST before making any changes.

MANDATORY FIRST STEP:
Read the file: CODEBASE_REVIEW_2025-12-19.md

This review document contains:
✓ 12 critical security issues with exact locations and fixes
✓ Architectural problems (inconsistent manager usage, memory leaks)
✓ 653 lines of orphaned code to remove (agents/ directory)
✓ Complete duplicate code inventory with line numbers
✓ Performance bottlenecks and optimization opportunities
✓ Testing gaps (60% of components untested)
✓ Prioritized TODO list with 21 recommendations
✓ Complete system architecture documentation

CRITICAL PRIORITY ITEMS (Security-Critical - Fix Immediately):

1. IMPLEMENT AUTHENTICATION SYSTEM [3 days]
   - No authentication anywhere in the system
   - Anyone can access all endpoints
   - Need: JWT or API key authentication
   - File to create: api/middleware/auth.py
   - Impact: HIGH SECURITY RISK

2. FIX CORS CONFIGURATION [1 hour]
   - Current: allow_origins=["*"] with allow_credentials=True
   - This is a security anti-pattern enabling CSRF attacks
   - Location: api/server.py line 67
   - Fix: Whitelist specific origins

3. SANITIZE ERROR MESSAGES [1 day]
   - All errors expose raw exception messages to clients
   - Leaks file paths, stack traces, internal structure
   - Location: All API error handlers
   - Fix: Log errors server-side, return generic messages

4. ADD REQUEST LOGGING [2 days]
   - Zero logging in API route handlers
   - No audit trail, impossible debugging
   - grep found only 5 logger calls in entire API
   - Need: Request/response/error logging

5. FIX STATE MANAGEMENT [2 hours]
   - health.py uses global state instead of dependency injection
   - Inconsistent with other routes
   - Potential data leakage
   - Location: api/routes/health.py lines 34, 57

6. UNIFY ON MEMORYMANA GER [1 day]
   - System uses 2 different managers: MemoryManager (new) vs ConversationManager (old)
   - API and CLI use old manager, missing features (caching, summarization, key facts)
   - Gradio UI uses correct MemoryManager
   - Files to fix: api/server.py line 103, main.py line 191

7. FIX MEMORY LEAKS [2 days]
   - regulations_cited set grows unbounded
   - key_facts_storage accumulates without limit
   - session_summaries.key_points grows indefinitely
   - Locations: conversation/manager.py:76,132, memory_manager.py:74,416

8. REMOVE ORPHANED AGENTS SYSTEM [2 hours]
   - agents/ directory: 653 lines, completely unused
   - Zero imports outside agents/
   - Zero tests
   - Contains security vulnerabilities
   - All tools just call pipeline.query() - redundant
   - Command: git rm -rf agents/

TASK SELECTION:
Which priority should I focus on?

OPTION A: Security Critical (Items 1-8 above)
OPTION B: Code Quality (duplicate removal, reorganization)
OPTION C: Performance (FAISS indexing, caching, batching)
OPTION D: Testing (add 60% missing unit tests)
OPTION E: Specific task (tell me which item number from review doc)

If you select a task, I will:
1. Read the detailed description from CODEBASE_REVIEW_2025-12-19.md
2. Understand the exact location, code, and required fix
3. Implement the solution
4. Test the changes
5. Commit with clear message

IMPORTANT NOTES:
- Current thinking mode implementation is COMPLETE and working
- Recent commits (last 5):
  * Control thinking length via max_new_tokens config, not prompts
  * Ubah SEMUA instruksi thinking ke bahasa Indonesia murni
  * Make MEDIUM and HIGH thinking truly verbose and iterative
  * Expand MEDIUM and HIGH thinking modes for longer analysis
  * Merge thinking_pipeline.py into prompt_builder.py for simplicity

- System is FUNCTIONAL but NOT production-ready (5.5/10 rating)
- Main blockers: Security (3/10), No persistence, Insufficient logging

WHAT WOULD YOU LIKE ME TO WORK ON?
```

---

## For Human Developers

### Quick Start

1. **Read the review:**
   ```bash
   cat CODEBASE_REVIEW_2025-12-19.md
   ```

2. **Check current status:**
   ```bash
   git status
   git log -5 --oneline
   ```

3. **Pick a priority:**
   - See "PRIORITIZED RECOMMENDATIONS" section in review doc
   - Or work through "OUTSTANDING TODO LIST"

### Quick Priority Reference

**Priority 1 - Security (Week 1):**
1. Implement authentication (api/middleware/auth.py)
2. Fix CORS (api/server.py:67)
3. Sanitize errors (all error handlers)
4. Add logging (all API routes)
5. Fix health.py state access
6. Unify managers (API + CLI → MemoryManager)
7. Fix memory leaks (3 locations)
8. Remove agents/ directory

**Priority 2 - Code Quality (Month 1):**
9. Remove duplicate code (6 instances)
10. Reorganize root directory (move 6 files)
11. Fix broken tests (3 files)
12. Merge community detection implementations

**Priority 3 - Performance (Quarter 1):**
13. Implement FAISS indexing (10-100x speedup)
14. Add result caching
15. Parallelize persona searches
16. Optimize reranker batching

**Priority 4 - Features (Quarter 2+):**
17. Add unit tests (60-70% coverage target)
18. Add monitoring/observability
19. Multi-user support
20. Async pipeline
21. Advanced features (MMR, explain API, etc.)

### Testing Before Deployment

```bash
# Fix missing dependencies first
pip install numpy torch requests pytest-asyncio

# Run unit tests
pytest tests/unit/ -v

# Run integration tests (need GPU/models)
pytest tests/integration/ -v --tb=short

# Check security issues
grep -r "allow_origins.*\*" api/
grep -r "except.*raise HTTPException.*str(e)" api/
```

### Architecture Quick Reference

```
UI (Gradio/FastAPI)
  → RAGPipeline
    → LangGraphOrchestrator
      → Search (HybridSearch + Personas + Consensus + Reranking)
    → GenerationEngine
      → PromptBuilder + LLMEngine + Citations
    → MemoryManager (or ConversationManager - needs unification!)
```

---

## Review Statistics

- **Files Analyzed:** 120+ Python files
- **Total LOC:** ~25,000-30,000 lines
- **Critical Issues:** 12
- **Security Issues:** 8
- **Code Quality Issues:** 15
- **Performance Issues:** 6
- **Testing Coverage:** 35-40% (60-65% missing)
- **Production Readiness:** 5.5/10 (NOT READY)

---

**Document Created:** December 19, 2025
**Last Updated:** December 19, 2025
**Reviewed By:** Senior Developer & Security Expert
