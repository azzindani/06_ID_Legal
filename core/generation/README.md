# Generation Module

LLM-based response generation for the Indonesian Legal RAG System.

## Components

| File | Description |
|------|-------------|
| `llm_engine.py` | LLM model loading and inference |
| `generation_engine.py` | Complete generation pipeline |
| `prompt_builder.py` | Context-aware prompt construction |
| `citation_formatter.py` | Legal citation formatting |
| `response_validator.py` | Response quality validation |

## Usage

```python
from core.generation import (
    LLMEngine,
    GenerationEngine,
    PromptBuilder,
    CitationFormatter
)

# Initialize LLM
llm = LLMEngine()
llm.load_model('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')

# Build prompt
builder = PromptBuilder()
prompt = builder.build(
    query="Apa sanksi pelanggaran?",
    context=retrieved_docs,
    query_type="sanctions"
)

# Generate response
engine = GenerationEngine(llm)
response = engine.generate(prompt, max_tokens=512)

# Format citations
formatter = CitationFormatter()
formatted = formatter.format(response, sources)
```

## Prompt Templates

The system uses specialized prompts for different query types:
- Definitions emphasize clarity
- Procedures emphasize steps
- Sanctions emphasize specific penalties

## Configuration

```python
config = {
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    'repetition_penalty': 1.1,
}
```
