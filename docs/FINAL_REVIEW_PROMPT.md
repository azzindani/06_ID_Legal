# Comprehensive Final Review Prompt

Use this prompt in a new conversation session to perform a thorough final review of the Indonesian Legal RAG System.

---

## PROMPT TO COPY:

```
I need you to perform a comprehensive final review of my Indonesian Legal RAG System project. This is a production-grade RAG (Retrieval-Augmented Generation) system for Indonesian legal documents.

## PROJECT LOCATION
The project is located at: d:\Antigravity\06_ID_Legal

## SYSTEM ARCHITECTURE
- **Backend**: FastAPI + Uvicorn (RESTful API with SSE streaming)
- **RAG Pipeline**: LangGraph-based multi-agent orchestrator with consensus mechanism
- **Models**: Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, Local LLM (Deepseek)
- **LLM Providers**: Local GPU + OpenRouter (valve architecture for instant switching)
- **UI**: Gradio-based unified interface
- **Dataset**: Indonesian legal regulations (HuggingFace: Azzindani/ID_REG_DB_2510)

## KEY FEATURES IMPLEMENTED
1. Multi-phase retrieval with research team personas
2. Skip Retrieval for conversational queries (greetings, thanks)
3. Query Rewriting (colloquial → formal legal terms)
4. Document upload and context injection
5. Streaming with thinking mode (<think> tags)
6. Valve-style LLM provider switching (Local ↔ OpenRouter)
7. Session-based conversation management
8. Export functionality (JSON, Markdown, HTML)

## REVIEW SCOPE

### 1. CODE QUALITY REVIEW
Please review:
- `config.py` - Configuration and environment handling
- `pipeline/rag_pipeline.py` - Main RAG pipeline
- `core/search/query_detection.py` - Query analysis and rewriting
- `core/search/langgraph_orchestrator.py` - Multi-agent orchestration
- `api/routes/llm.py` - LLM provider switching
- `api/routes/rag_enhanced.py` - Chat and research endpoints
- `conversation/conversational_service.py` - Conversational handling
- `core/legal_vocab.py` - Legal vocabulary and synonyms
- `ui/unified_app_api.py` - Gradio UI

Focus on:
- Code organization and modularity
- Error handling completeness
- Logging consistency
- Type hints usage
- Docstring quality
- DRY violations
- Potential bugs

### 2. SECURITY REVIEW
Check for:
- Input validation vulnerabilities
- API authentication gaps
- Injection risks (prompt injection, SQL injection)
- Sensitive data exposure in logs
- Rate limiting implementation
- CORS configuration

### 3. PERFORMANCE REVIEW
Analyze:
- Memory management (GPU/CPU)
- Caching effectiveness
- Query optimization
- Streaming efficiency
- Resource cleanup

### 4. DOCUMENTATION REVIEW
Review:
- `README.md` - Main documentation
- `docs/PRODUCTION_DEPLOYMENT.md` - Deployment guide
- API endpoint documentation
- Code comments and docstrings

### 5. TESTING GAPS
Identify:
- Missing unit tests
- Missing integration tests
- Edge cases not covered
- Error scenarios not tested

### 6. PRODUCTION READINESS
Evaluate:
- Logging and monitoring hooks
- Health check endpoints
- Graceful shutdown handling
- Configuration management
- Environment separation (dev/staging/prod)

## DELIVERABLES REQUESTED

1. **Executive Summary**: Overall assessment (1-10 scale) with key findings
2. **Critical Issues**: Must-fix before production deployment
3. **Improvements**: Nice-to-have enhancements
4. **Security Findings**: Any vulnerabilities found
5. **Performance Recommendations**: Optimization opportunities
6. **Code Quality Report**: With specific file/line references
7. **Action Items**: Prioritized list of next steps

Please start by exploring the codebase structure, then perform each review category systematically.
```

---

## USAGE INSTRUCTIONS

1. Start a new conversation session
2. Copy the entire prompt above (from "I need you to perform..." to the end)
3. Paste it as your first message
4. The AI will systematically review all components
5. You'll receive a comprehensive report with actionable items

## EXPECTED OUTPUT

The review should produce:
- Overall quality score (1-10)
- List of critical bugs/issues
- Security vulnerability assessment
- Performance optimization suggestions
- Prioritized action items for production readiness
