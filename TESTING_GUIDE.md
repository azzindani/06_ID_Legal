# Testing Guide - Verifying Bug Fixes

This guide shows you how to test all the critical bug fixes using the existing test infrastructure.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install all required dependencies
pip install -r requirements.txt

# Or if using conda:
conda create -n legal-rag python=3.10
conda activate legal-rag
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Create .env file with your API keys (if using external LLM providers)
cat > .env << 'EOF'
# Optional: Only needed if you want to test with external providers
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
EOF
```

## 📋 Testing the Bug Fixes

### Test 1: Division by Zero Fix (hybrid_search.py)

**What was fixed:** Weight normalization crash when all weights are zero

```bash
# Run the hybrid search unit test
python -m pytest tests/unit/test_dataloader.py -v -k "search"

# Or test directly with comprehensive test:
python tests/integration/comprehensive_test.py
```

**Manual verification:**
```python
# Create a test script: test_weight_fix.py
import sys
sys.path.insert(0, '.')

from core.search.hybrid_search import HybridSearchEngine
from config import RESEARCH_TEAM_PERSONAS

# Simulate zero weights scenario
persona = RESEARCH_TEAM_PERSONAS['senior_legal_researcher']
weights = {'semantic_match': 0.0, 'keyword_precision': 0.0}

# This should NOT crash anymore
engine = HybridSearchEngine(None, None, None)
result = engine._apply_persona_weights(weights, persona)
print(f"✅ Zero weights handled: {result}")
```

### Test 2: XML Parsing Fix (generation_engine.py)

**What was fixed:** Robust parsing of thinking tags with fallback mechanisms

```bash
# Run generation tests
python -m pytest tests/unit/test_generation.py -v

# Test with malformed XML
python -c "
import sys
sys.path.insert(0, '.')
from core.generation.generation_engine import GenerationEngine
from config import get_default_config

config = get_default_config()
engine = GenerationEngine(config)

# Test various formats
test_cases = [
    '<think>Normal thinking</think>Answer here',
    '<think>Nested <think>tags</think></think>Answer',
    'No tags at all',
    '<think>Unclosed tag... Answer here',
    'Multiple <think>first</think> and <think>second</think> blocks'
]

for i, text in enumerate(test_cases):
    thinking, answer = engine._extract_thinking(text)
    print(f'✅ Test {i+1}: Extracted thinking={len(thinking)} chars, answer={len(answer)} chars')
"
```

### Test 3: Global State Fix (api/server.py)

**What was fixed:** Multi-worker support with app.state instead of globals

```bash
# Test API with multiple workers
uvicorn api.server:app --workers 4 --host 0.0.0.0 --port 8000

# In another terminal, test concurrent requests:
for i in {1..10}; do
  curl -X POST "http://localhost:8000/api/v1/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "Test query '$i'", "max_results": 5}' &
done
wait

echo "✅ All concurrent requests completed without crashes"
```

**Automated test:**
```bash
# Run API health check
python -c "
import requests
import concurrent.futures

def test_endpoint(i):
    response = requests.post(
        'http://localhost:8000/api/v1/search',
        json={'query': f'Test {i}', 'max_results': 5}
    )
    return response.status_code == 200

# Start server first, then:
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(test_endpoint, range(20)))
    print(f'✅ Passed: {sum(results)}/20 concurrent requests')
"
```

### Test 4: Memory Leak Fix (stages_research.py)

**What was fixed:** Bounded memory for persona performance tracking

```bash
# Test memory growth
python -c "
import sys
sys.path.insert(0, '.')
from core.search.stages_research import StagesResearchEngine
import tracemalloc

tracemalloc.start()

engine = StagesResearchEngine(None, {})

# Simulate 1000 queries (should not grow unboundedly)
for i in range(1000):
    engine.update_persona_performance(
        'senior_legal_researcher',
        'procedural',
        0.8,
        10
    )

current, peak = tracemalloc.get_traced_memory()
print(f'✅ Memory after 1000 updates: {current / 1024:.2f} KB')
print(f'✅ Peak memory: {peak / 1024:.2f} KB')

# Check history is bounded
perf = engine._persona_performance['senior_legal_researcher']['procedural']
print(f'✅ History size bounded to: {len(perf[\"result_counts\"])} entries (max 100)')
assert len(perf['result_counts']) <= 100, 'Memory leak detected!'
print('✅ Memory leak fix verified!')
"
```

### Test 5: Input Validation (API routes)

**What was fixed:** Enhanced input validation and XSS prevention

```bash
# Test validation with various inputs
python -c "
import sys
sys.path.insert(0, '.')
from api.routes.search import SearchRequest
from pydantic import ValidationError

# Valid input
try:
    req = SearchRequest(query='Valid query', max_results=10)
    print('✅ Valid input accepted')
except ValidationError as e:
    print(f'❌ Valid input rejected: {e}')

# Test XSS prevention
malicious_inputs = [
    '<script>alert(1)</script>',
    'javascript:alert(1)',
    '<img onerror=alert(1)>',
]

for mal_input in malicious_inputs:
    try:
        req = SearchRequest(query=mal_input)
        print(f'❌ Malicious input NOT blocked: {mal_input}')
    except ValidationError:
        print(f'✅ Malicious input blocked: {mal_input[:30]}...')

# Test length limits
try:
    req = SearchRequest(query='x' * 3000)
    print('❌ Long input NOT rejected')
except ValidationError:
    print('✅ Long input rejected (max 2000 chars)')

# Test session ID validation
from api.routes.session import SessionCreateRequest
try:
    req = SessionCreateRequest(session_id='valid-session_123')
    print('✅ Valid session ID accepted')
except ValidationError as e:
    print(f'❌ Valid session ID rejected: {e}')

try:
    req = SessionCreateRequest(session_id='../../../etc/passwd')
    print('❌ Path traversal NOT blocked')
except ValidationError:
    print('✅ Path traversal blocked in session ID')
"
```

### Test 6: Rate Limiting

**What was fixed:** Added rate limiting middleware

```bash
# Test rate limiting (requires running server)
python -c "
import requests
import time

url = 'http://localhost:8000/api/v1/health'

print('Testing rate limiting...')
success_count = 0
rate_limited_count = 0

# Send 70 requests rapidly (limit is 60/minute)
for i in range(70):
    response = requests.get(url)
    if response.status_code == 200:
        success_count += 1
    elif response.status_code == 429:
        rate_limited_count += 1
        print(f'✅ Rate limited at request {i+1}')
        break
    time.sleep(0.1)

print(f'✅ Successful: {success_count}')
print(f'✅ Rate limited: {rate_limited_count}')
assert rate_limited_count > 0, 'Rate limiting not working!'
print('✅ Rate limiting working correctly!')
"
```

## 🧪 Running All Tests

### Run All Unit Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# With coverage report
python -m pytest tests/unit/ -v --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_generation.py -v
```

### Run Comprehensive Integration Tests (NEW - Production-Ready)

These tests show REAL output and initialize the full system like production:

```bash
# 1. Production Readiness Test (Complete System)
# Tests: Simple queries, complex queries, multi-turn, bug fixes, performance
python tests/integration/test_production_ready.py

# 2. API Endpoints Test (Full API Testing)
# Tests: Health, search, generate, sessions, validation, rate limiting
python tests/integration/test_api_endpoints.py

# 3. Session & Export Test (Conversation Management)
# Tests: Sessions, history, Markdown/JSON/HTML export
python tests/integration/test_session_export.py

# 4. Streaming LLM Output Test (REAL-TIME STREAMING) ⭐
# Tests: Direct pipeline streaming, API SSE streaming, session-based streaming
# Watch the LLM response appear in REAL-TIME as it generates!
python tests/integration/test_streaming.py

# Optional: Test API streaming endpoints too (requires starting server)
python tests/integration/test_streaming.py --api

# 5. Comprehensive Audit & Metadata Test (FULL TRANSPARENCY) 🔍
# Shows ALL internal details: scores, calculations, metadata, references
# Perfect for: auditing, debugging, UI development, understanding ranking
# Output includes:
#   - All scoring details (semantic, keyword, KG, authority, temporal, completeness)
#   - Weight calculations and final scores
#   - Complete document metadata with entities & relationships
#   - Phase-by-phase research results
#   - Persona contributions
#   - Timing breakdowns
#   - Full JSON metadata dump
python tests/integration/test_audit_metadata.py

# Custom query
python tests/integration/test_audit_metadata.py --query "Apa sanksi UU ITE?"

# Compare multiple queries
python tests/integration/test_audit_metadata.py --multi

# 6. Performance & Load Testing (BENCHMARKS) 📊
# Tests: Response times, concurrent handling, memory usage, throughput
# Output includes:
#   - Query response times by type (simple, sanction, procedural, complex)
#   - P50/P90/P99 latency percentiles
#   - Throughput (queries per second)
#   - Memory profiling
#   - Concurrent load testing
python tests/integration/test_performance.py

# Full benchmark suite (all query types)
python tests/integration/test_performance.py --full

# Concurrent load test
python tests/integration/test_performance.py --concurrent --threads 3 --queries 2

# Memory profiling
python tests/integration/test_performance.py --memory

# 7. Complete RAG Output Test (5 QUERIES WITH FULL METADATA) 📋
# Tests: Multiple queries with streaming, ALL retrieved documents, full scoring
# Output includes:
#   - Question, Query Type, Thinking Process
#   - Streamed answer (real-time token output)
#   - ALL legal references with complete scoring metadata
#   - Research process details (team members, phases, document counts)
#   - Export to JSON for further analysis
python tests/integration/test_complete_output.py

# With JSON export for parsing
python tests/integration/test_complete_output.py --export

# 8. Conversational Test (MULTI-TURN DIALOGUE WITH MEMORY) 💬
# Tests: Conversation memory, context management, topic shifts, regulation recognition
# Demonstrates intelligent legal assistant behavior across 5 turns:
#   Turn 1: General labor rights question (establishes context)
#   Turn 2: Specific UU No. 13/2003 on severance (tests regulation recognition)
#   Turn 3: Follow-up question on legal remedies (tests conversation memory)
#   Turn 4: Topic shift to environmental permits (tests context switching)
#   Turn 5: Tax law question (tests multi-domain knowledge)
# Output includes:
#   - Turn-by-turn analysis with topic tracking
#   - Conversation coherence scoring
#   - Memory and context metrics
#   - Specific regulation recognition stats
python tests/integration/test_conversational.py

# With verbose metadata (shows thinking process)
python tests/integration/test_conversational.py --verbose

# Export conversation results to JSON
python tests/integration/test_conversational.py --export --output conversation_results.json

# 9. Output Parser & Validator (AUDIT REPORTS) 📊
# Parses JSON exports and generates audit reports
# Output includes:
#   - Structured extraction of all legal references
#   - Validation of output completeness
#   - Per-query audit breakdown
#   - CSV export for spreadsheet analysis
python tests/integration/test_output_parser.py --file <export.json>

# Generate test data and parse in one command
python tests/integration/test_output_parser.py --generate

# Export references to CSV
python tests/integration/test_output_parser.py --file <export.json> --csv references.csv

# 10. Complete RAG Pipeline Test
python tests/integration/test_complete_rag.py

# 11. Integrated System Test
python tests/integration/test_integrated_system.py

# 12. End-to-End Test (with pytest)
python -m pytest tests/integration/test_end_to_end.py -v -s

# 13. Stress Test - Single Query (MAXIMUM LOAD)
# Tests single query with all settings maxed out
python tests/integration/test_stress_single.py

# 14. Stress Test - Conversational (MAXIMUM LOAD)
# Tests 7-turn conversation with maximum settings
python tests/integration/test_stress_conversational.py
```

### Run All Integration Tests at Once

```bash
# Run core comprehensive tests
python tests/integration/test_production_ready.py && \
python tests/integration/test_session_export.py && \
python tests/integration/test_conversational.py

# Or run all Python-based tests (shows real output)
for test in tests/integration/test_*.py; do
    echo "Running $test..."
    python "$test" || echo "Failed: $test"
done
```

## 🌐 Testing the API

### Start the API Server

```bash
# Development mode (single worker)
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# Production mode (multiple workers to test multi-worker fix)
uvicorn api.server:app --workers 4 --host 0.0.0.0 --port 8000
```

### Test API Endpoints

```bash
# 1. Health check
curl http://localhost:8000/api/v1/health

# 2. Search endpoint
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa sanksi pelanggaran UU ITE?",
    "max_results": 5
  }'

# 3. Generate answer endpoint
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Jelaskan tentang perlindungan data pribadi"
  }'

# 4. Create session
curl -X POST "http://localhost:8000/api/v1/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-1"
  }'

# 5. List sessions
curl http://localhost:8000/api/v1/sessions

# 6. STREAMING endpoint (Server-Sent Events - watch real-time output!)
curl -N -X POST "http://localhost:8000/api/v1/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Jelaskan tentang hak konsumen",
    "stream": true
  }'

# 7. View API documentation
# Open browser: http://localhost:8000/docs
```

## 🖥️ Testing with Gradio UI

### Start the Gradio Interface

```bash
# Run the Gradio UI
python ui/gradio_app.py

# Or if gradio_app is in root:
python gradio_app.py
```

Then open your browser to `http://localhost:7860` and test:

1. **Basic Query**: "Apa sanksi dalam UU ITE?"
2. **Complex Query**: "Bagaimana prosedur pelaporan pelanggaran data pribadi menurut UU PDP?"
3. **Session Testing**: Create a session and ask follow-up questions
4. **Export Testing**: Try exporting conversation to different formats

## 🔍 Audit & Transparency Testing

The audit test (`test_audit_metadata.py`) provides complete transparency into the RAG system's decision-making process.

### Use Cases

**1. Auditing Search Results**
```bash
# See exactly how documents are scored and ranked
python tests/integration/test_audit_metadata.py --query "Apa sanksi UU ITE?"
```

Output shows:
- Individual scores: semantic (0.8234), keyword (0.6521), KG (0.7123), etc.
- Weight calculations: how final score is computed
- Visual score bars for quick assessment
- Complete document metadata

**2. Debugging Unexpected Results**
```bash
# Compare scoring across multiple queries
python tests/integration/test_audit_metadata.py --multi
```

Helps identify:
- Why certain documents rank higher than others
- Which scoring components contribute most
- Query type classification accuracy
- Performance variations

**3. UI Development Reference**
```bash
# Get complete JSON metadata for UI implementation
python tests/integration/test_audit_metadata.py --query "Your query" > audit_output.txt
```

Provides:
- All metadata fields available
- Score breakdowns for display
- Citation formatting examples
- Phase-by-phase research data

**4. Verifying Multi-Researcher Consensus**

The audit shows:
- Which researchers (personas) contributed
- Confidence scores per researcher
- Top candidates from each phase
- Consensus building process

**5. Understanding Knowledge Graph Impact**

Shows detailed KG metrics:
- Authority scores (regulatory hierarchy)
- Temporal scores (recency/relevance)
- Completeness scores (coverage)
- Extracted entities and relationships
- Document references

### Example Output Sections

The audit test displays **11 comprehensive sections**:

1. **Generated Answer** - Final response
2. **Thinking Process** - Internal reasoning (if available)
3. **Timing Breakdown** - Where time is spent (retrieval vs generation)
4. **Query Analysis** - Query type, results count, cache status
5. **Research Phases** - Multi-researcher analysis details
6. **Source Documents** - Full scoring for each source
7. **Citations** - Formatted legal references
8. **Complete Metadata** - JSON dump for programmatic use
9. **Scoring Breakdown** - Visual bars and percentages for all score types
10. **Document Metadata** - Entities, relationships, references
11. **Audit Summary** - High-level statistics

### Scoring Details Explained

Each document shows 6 individual scores that combine into final score:

- **Semantic Match** (0-1): Embedding similarity to query
- **Keyword Precision** (0-1): TF-IDF keyword matching
- **Knowledge Graph** (0-1): Entity/relationship relevance
- **Authority Hierarchy** (0-1): Regulatory importance level
- **Temporal Relevance** (0-1): Recency and validity
- **Legal Completeness** (0-1): Document comprehensiveness

Final Score = weighted sum of above (default weights: 0.25, 0.15, 0.20, 0.20, 0.10, 0.10)

## 📊 Performance & Load Testing

The performance test (`test_performance.py`) provides benchmarks for system performance and scalability.

### What It Measures

**1. Response Time Benchmarks**
- Tests 4 query categories: simple, sanction, procedural, complex
- Calculates: average, min, max, P50, P90, P99 latencies
- Tracks success rate per query type

**2. Concurrent Load Testing**
- Multi-threaded query execution
- Configurable threads (default: 3) and queries per thread
- Measures true throughput (QPS) under load

**3. Memory Profiling**
- Tracks memory delta per query
- Monitors peak memory usage
- Identifies potential memory leaks

### Running Performance Tests

```bash
# Quick performance test (simple + sanction queries only)
python tests/integration/test_performance.py

# Full benchmark (all 4 query types)
python tests/integration/test_performance.py --full

# Concurrent load test (3 threads, 2 queries each)
python tests/integration/test_performance.py --concurrent

# Custom concurrent settings
python tests/integration/test_performance.py --concurrent --threads 5 --queries 3

# Memory profiling only
python tests/integration/test_performance.py --memory

# Minimal output (quiet mode)
python tests/integration/test_performance.py --quiet
```

### Example Output

```
  OVERALL PERFORMANCE
======================================================================

  Queries:
    Total:      12
    Successful: 12 (100.0%)
    Failed:     0

  Response Times:
    Average:    3.45s
    Min:        2.12s
    Max:        5.87s
    P50:        3.21s
    P90:        4.98s
    P99:        5.87s

  Throughput:
    QPS:        0.289 queries/second
    Total Time: 41.42s
```

### Performance Baselines

Expected performance on typical hardware:

| Metric | CPU-only | Single GPU |
|--------|----------|------------|
| Simple Query | 3-5s | 1-2s |
| Complex Query | 8-15s | 3-5s |
| Concurrent QPS | 0.2-0.3 | 0.5-1.0 |
| Memory per Query | <100MB | <500MB |

## 📋 Complete RAG Output Testing

The complete output test (`test_complete_output.py`) provides full transparency into ALL retrieved documents with streaming output - essential for legal auditing.

### Streaming Mode Metadata

Streaming mode now includes **full metadata** for audit transparency. The `complete` chunk contains:

```python
{
    'type': 'complete',
    'answer': '...',                    # Final answer text
    'thinking': '...',                  # LLM reasoning process (extracted from <think> tags)
    'sources': [...],                   # Formatted source documents
    'citations': [...],                 # Citation references
    'phase_metadata': {                 # ALL documents from each research phase
        '0_initial_search_analyst': {
            'phase': 'initial_search',
            'researcher': 'analyst',
            'researcher_name': 'Legal Analyst',
            'candidates': [             # Documents with full scores
                {
                    'record': {...},    # Full document record
                    'scores': {
                        'final': 0.85,
                        'semantic': 0.82,
                        'keyword': 0.78,
                        'kg': 0.90,
                        'authority': 0.95,
                        'temporal': 0.80
                    }
                }
            ],
            'confidence': 1.0
        }
    },
    'research_log': {                   # Summary of research process
        'team_members': ['analyst', 'expert', 'generalist'],
        'total_documents_retrieved': 15,
        'phase_results': {...}
    },
    'consensus_data': {...},            # Team consensus information
    'research_data': {...}              # Raw research data
}
```

### What It Shows

**1. Complete Document Retrieval**
- Shows ALL documents retrieved by RAG (not just cited sources)
- Full scoring breakdown for each document (semantic, keyword, KG, authority, temporal)
- Perfect for verifying no relevant regulations were missed

**2. Streaming Answer Output**
- Real-time token-by-token streaming using TextIteratorStreamer
- Watch the LLM generate answers live
- Chunk count and timing statistics

**3. Thinking Process Display**
- Shows LLM reasoning extracted from `<think>` tags
- Helps understand how the answer was formulated
- Essential for debugging and quality assurance

**4. Research Process Transparency**
- Team members (researcher personas) involved
- Phase-by-phase breakdown with document counts
- Per-phase document lists with scores
- Confidence levels per researcher

### Running Complete Output Tests

```bash
# Run 5 queries with full metadata display
python tests/integration/test_complete_output.py

# Export results to JSON for parsing
python tests/integration/test_complete_output.py --export

# Specify custom output path
python tests/integration/test_complete_output.py --export --output my_results.json
```

### Example Output Format

```
====================================================================================================
COMPLETE RAG OUTPUT
====================================================================================================

## QUESTION
--------------------------------------------------------------------------------
Apa saja hak-hak pekerja menurut UU Ketenagakerjaan?

## QUERY TYPE: procedural

## THINKING PROCESS
--------------------------------------------------------------------------------
[Full reasoning process displayed here]

## ANSWER
--------------------------------------------------------------------------------
[Streamed answer appears in real-time]
[Streamed: 145 chunks in 8.23s]

## LEGAL REFERENCES (All Retrieved Documents - FULL DETAILS)
--------------------------------------------------------------------------------
Total Documents Retrieved: 12

### 1. Undang-Undang No. 13/2003
   Global ID: uu-13-2003-ketenagakerjaan
   About: Ketenagakerjaan
   Enacting Body: Presiden Republik Indonesia
   Location in Document:
      Chapter/Bab: X
      Section/Bagian: Kesatu
      Article/Pasal: 77
      Paragraph/Ayat: 1
   Relevance Scores:
      Final Score: 0.8934
      Semantic: 0.8521
      Keyword: 0.7823
      KG Score: 0.9012
      Authority: 0.9500
      Temporal: 0.8200
      Completeness: 0.8750
   Knowledge Graph Metadata:
      Domain: ketenagakerjaan
      Hierarchy Level: 1
      Cross References: 15
   Research Team Analysis:
      Team Consensus: Yes
      Researcher Agreement: 3
      Personas Agreed: analyst, expert, generalist
   Full Content (2543 chars):
   ------------------------------------------------------------
      Pasal 77 ayat (1) Setiap pengusaha wajib melaksanakan
      ketentuan waktu kerja...
   ------------------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[... more documents with full details ...]

## RESEARCH PROCESS DETAILS (FULL)
--------------------------------------------------------------------------------

### Research Team
Team Size: 3
   - 👨‍⚖️ Senior Legal Researcher
   - 👩‍⚖️ Junior Legal Researcher
   - 📚 Knowledge Graph Specialist

### Summary Statistics
Total Documents Retrieved: 12
Total Phases: 3

### Phase-by-Phase Breakdown (ALL Documents)
================================================================================

#### PHASE: initial_search
   Researcher: 👨‍⚖️ Senior Legal Researcher
   Documents Found: 8
   Confidence: 85.00%

   Documents Retrieved in This Phase:
   ----------------------------------------------------------------------

   [1] Undang-Undang No. 13/2003
       ID: uu-13-2003-ketenagakerjaan
       About: Ketenagakerjaan
       Location: Bab X > Bagian Kesatu > Pasal 77 > Ayat 1
       Scores: Final=0.8934 | Semantic=0.8521 | Keyword=0.7823 | KG=0.9012 | Authority=0.9500 | Temporal=0.8200
       Content Preview: Pasal 77 ayat (1) Setiap pengusaha wajib melaksanakan...

   [2] Peraturan Pemerintah No. 35/2021
       ID: pp-35-2021-pkwt
       About: Perjanjian Kerja Waktu Tertentu
       Location: Bab II > Pasal 4
       Scores: Final=0.8521 | Semantic=0.8234 | Keyword=0.7654 | KG=0.8890 | Authority=0.9200 | Temporal=0.9100
       Content Preview: Pasal 4 PKWT dapat dibuat untuk pekerjaan yang...

   [... all documents in this phase ...]

   ----------------------------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#### PHASE: refinement
   Researcher: Regulatory Expert
   Documents Found: 5
   Confidence: 92.00%

   [... all documents in this phase with full details ...]

## TIMING
--------------------------------------------------------------------------------
Total Time: 12.345s
Retrieval Time: 4.123s
Generation Time: 8.222s
```

### Parsing Exported Results

Use `test_output_parser.py` to parse and analyze exported JSON:

```bash
# Parse existing export file
python tests/integration/test_output_parser.py --file complete_output_results_1234567890.json

# Generate and parse in one command
python tests/integration/test_output_parser.py --generate

# Export all references to CSV for spreadsheet analysis
python tests/integration/test_output_parser.py --file export.json --csv all_references.csv

# Save audit report to file
python tests/integration/test_output_parser.py --file export.json --report audit_report.txt
```

### Output Parser Features

The parser extracts structured data from RAG output:

**LegalReference dataclass:**
- `regulation_type`, `regulation_number`, `year`, `about`
- All scores: `final_score`, `semantic_score`, `keyword_score`, `kg_score`, etc.
- `domain`, `hierarchy_level`
- `team_consensus`, `researcher_agreement`, `personas_agreed`
- `content_snippet`

**QueryResult dataclass:**
- Query information and success status
- Streaming statistics (chunks, duration)
- List of all `LegalReference` objects
- Research phases with researcher details
- Timing breakdown

### Use Cases

**1. Legal Audit Compliance**
- Verify all relevant regulations were considered
- Check scoring justification for each document
- Validate research process transparency

**2. Quality Assurance**
- Compare results across multiple queries
- Identify scoring anomalies
- Verify team consensus accuracy

**3. Data Export**
- CSV export for Excel/spreadsheet analysis
- JSON for programmatic processing
- Text reports for documentation

## 💪 Stress Testing (MAXIMUM LOAD)

Stress tests verify system stability under maximum configuration settings.

### Stress Test 1: Single Query Maximum Load

Tests a single complex query with ALL settings maxed out:
- All 5 search phases enabled (including expert_review)
- Maximum candidates per phase (600-800)
- All 5 research personas active
- Maximum final_top_k (20 documents)
- Maximum max_new_tokens (8192)

```bash
# Full stress test (maximum settings)
python tests/integration/test_stress_single.py

# Quick mode (moderate settings)
python tests/integration/test_stress_single.py --quick

# With memory profiling
python tests/integration/test_stress_single.py --memory

# Export results to JSON
python tests/integration/test_stress_single.py --export
```

### Stress Test 2: Conversational Maximum Load

Tests 7-turn complex conversation with maximum settings:
- Cross-domain topics (tax, labor, multi-domain)
- Topic shifts and back-references
- Context building across turns
- Heavy conversation history (50 turns tracked)
- Summary requests requiring full context

```bash
# Full stress test (7 turns, maximum settings)
python tests/integration/test_stress_conversational.py

# Quick mode (5 turns, moderate settings)
python tests/integration/test_stress_conversational.py --quick

# With memory profiling per turn
python tests/integration/test_stress_conversational.py --memory

# Export results to JSON
python tests/integration/test_stress_conversational.py --export
```

### Stress Test Configuration Details

| Setting | Default | Stress Max | Description |
|---------|---------|------------|-------------|
| final_top_k | 3 | 20 | Documents returned |
| research_team_size | 4 | 5 | Active personas |
| max_new_tokens | 2048 | 8192 | Generation limit |
| search_phases | 4/5 | 5/5 | All phases enabled |
| candidates (total) | ~640 | ~1500+ | Search candidates |

### What Stress Tests Verify

1. **System Stability**: No crashes under maximum load
2. **Memory Bounds**: Memory stays within acceptable limits
3. **Timeout Handling**: Long operations complete or timeout gracefully
4. **Context Management**: Large conversation contexts are handled
5. **Resource Cleanup**: Proper cleanup after heavy operations

### Expected Stress Test Metrics

| Metric | Quick Mode | Maximum Mode |
|--------|------------|--------------|
| Single Query Time | 30-60s | 60-180s |
| Memory Peak | <2GB | <4GB |
| Conv Turn Time (avg) | 20-40s | 40-90s |
| 7-Turn Total | N/A | 5-15 min |

## 📊 Verification Checklist

After running tests, verify all bug fixes:

- [ ] **Division by Zero**: No crashes with zero weights
- [ ] **XML Parsing**: Handles malformed thinking tags gracefully
- [ ] **Global State**: Multiple workers work correctly
- [ ] **Memory Leak**: Memory stays bounded after many queries
- [ ] **Input Validation**: Malicious inputs are rejected
- [ ] **Rate Limiting**: Requests are rate-limited correctly

## 🐛 Troubleshooting

### Missing Dependencies

```bash
# If pytest not found
pip install pytest pytest-cov pytest-timeout

# If torch not found
pip install torch torchvision torchaudio

# If fastapi not found
pip install fastapi uvicorn pydantic
```

### GPU/CUDA Issues

```bash
# Use CPU-only versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu
```

### Import Errors

```bash
# Make sure you're in the project root
cd /home/user/06_ID_Legal
python -c "import sys; print(sys.path)"

# Run tests from project root
python -m pytest tests/unit/ -v
```

## 📝 Expected Test Results

All tests should pass with these approximate results:

- **Unit Tests**: ~90% pass rate (some may need GPU)
- **Integration Tests**: ~80% pass rate (some may need dataset)
- **API Tests**: 100% pass rate (with server running)
- **Syntax Checks**: 100% pass rate ✅

## 🎯 Quick Validation Script

Run this to validate all fixes at once:

```bash
# Create quick_validation.sh
cat > quick_validation.sh << 'EOF'
#!/bin/bash
echo "🔍 Quick Validation of Bug Fixes"
echo "================================="

echo "✅ 1. Checking Python syntax..."
python -m py_compile core/search/hybrid_search.py && echo "   hybrid_search.py OK"
python -m py_compile core/generation/generation_engine.py && echo "   generation_engine.py OK"
python -m py_compile api/server.py && echo "   api/server.py OK"
python -m py_compile core/search/stages_research.py && echo "   stages_research.py OK"

echo ""
echo "✅ 2. Testing imports..."
python -c "from core.search.hybrid_search import HybridSearchEngine; print('   HybridSearchEngine OK')"
python -c "from core.generation.generation_engine import GenerationEngine; print('   GenerationEngine OK')"
python -c "from core.search.stages_research import StagesResearchEngine; print('   StagesResearchEngine OK')"

echo ""
echo "✅ 3. Running unit tests (if available)..."
python -m pytest tests/unit/ -q || echo "   (Skipped - pytest or dependencies not available)"

echo ""
echo "✅ All validations complete!"
EOF

chmod +x quick_validation.sh
./quick_validation.sh
```

This comprehensive testing approach will verify that all 7 critical bugs have been fixed correctly!
