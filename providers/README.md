# LLM Providers Module

Plug-and-play LLM backends for the Indonesian Legal RAG System.

## Supported Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| `local` | HuggingFace models | Privacy, offline, full control |
| `openai` | GPT-4, GPT-3.5 | High quality, fast |
| `anthropic` | Claude 3.5, Claude 3 | Long context, reasoning |
| `google` | Gemini 1.5 | Multimodal, long context |
| `openrouter` | Multiple providers | Flexibility, cost optimization |

## Configuration

### Environment Variables

```bash
# Select provider
LLM_PROVIDER=local  # local, openai, anthropic, google, openrouter

# API Keys (only for cloud providers)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# Cloud model names
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-sonnet-4-20250514
GOOGLE_MODEL=gemini-1.5-pro
OPENROUTER_MODEL=anthropic/claude-sonnet-4
```

### Local Inference (Privacy-Focused)

```bash
# CPU/GPU split for resource efficiency
EMBEDDING_DEVICE=cpu    # Small model, CPU is fine
RERANKER_DEVICE=cpu     # Small model, CPU is fine
LLM_DEVICE=cuda         # Large model needs GPU

# Quantization (reduces VRAM from 16GB to ~6GB)
LLM_LOAD_IN_4BIT=true
LLM_LOAD_IN_8BIT=false
```

## Usage

### Basic Usage

```python
from providers import get_provider

# Use configured provider
provider = get_provider()
response = provider.generate("Apa sanksi pelanggaran UU Ketenagakerjaan?")
print(response)
```

### Switching Providers

```python
from providers import switch_provider

# Switch to Claude
provider = switch_provider('anthropic', {
    'api_key': 'sk-ant-...',
    'model': 'claude-sonnet-4-20250514'
})

response = provider.generate("Explain Indonesian labor law")
```

### Streaming

```python
from providers import get_provider

provider = get_provider()
for chunk in provider.generate_stream("Apa itu hukum pidana?"):
    print(chunk, end="", flush=True)
```

### Chat with History

```python
messages = [
    {"role": "system", "content": "You are a legal assistant."},
    {"role": "user", "content": "Apa itu UU Ketenagakerjaan?"},
    {"role": "assistant", "content": "UU Ketenagakerjaan adalah..."},
    {"role": "user", "content": "Apa sanksinya?"}
]

response = provider.chat(messages)
```

## Hybrid Mode

Best for privacy: Local embeddings + Cloud LLM

```bash
# Keep embeddings local (data stays private)
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu

# Use cloud for generation (no raw data sent)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

This way:
- Your documents never leave your machine
- Only the query and retrieved context go to the cloud
- Full control over what data is shared

## Custom Providers

```python
from providers.base import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    def initialize(self) -> bool:
        # Setup your provider
        return True

    def generate(self, prompt, max_tokens=2048, temperature=0.7, **kwargs):
        # Generate response
        return "Response"

    def generate_stream(self, prompt, max_tokens=2048, temperature=0.7, **kwargs):
        # Stream response
        yield "Response"

# Register
from providers.factory import PROVIDERS
PROVIDERS['myprovider'] = MyProvider
```

## Installation

For cloud providers, install the required packages:

```bash
pip install openai        # For OpenAI and OpenRouter
pip install anthropic     # For Anthropic
pip install google-generativeai  # For Google
pip install bitsandbytes  # For local quantization
```
