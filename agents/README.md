# Agents Module

Agentic workflows with tool-based execution for complex legal tasks.

## Components

| File | Description |
|------|-------------|
| `tool_registry.py` | Tool registration and management |
| `agent_executor.py` | Multi-step agent execution |
| `tools/search_tool.py` | Document search tool |
| `tools/citation_tool.py` | Legal citation lookup |
| `tools/summary_tool.py` | Topic summarization |

## Usage

```python
from agents import AgentExecutor
from pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()
pipeline.initialize()

# Create agent executor
executor = AgentExecutor(pipeline)

# Execute complex task
result = executor.execute(
    "Research all sanctions related to labor law violations and summarize them"
)

print(result['answer'])
print(f"Tools used: {len(result['tool_results'])}")
```

## Available Tools

### Search Tool
```python
# Searches legal documents
{"name": "search", "query": "sanksi ketenagakerjaan", "max_results": 5}
```

### Citation Tool
```python
# Looks up specific regulations
{"name": "citation", "regulation_type": "UU", "regulation_number": "13", "year": "2003"}
```

### Summary Tool
```python
# Summarizes legal topics
{"name": "summary", "topic": "hak karyawan kontrak", "max_length": 200}
```

## Creating Custom Tools

```python
from agents.tool_registry import BaseTool

class CustomTool(BaseTool):
    def __init__(self, pipeline):
        super().__init__(
            name="custom",
            description="My custom tool"
        )
        self.pipeline = pipeline

    def execute(self, **kwargs):
        # Tool logic here
        return {"result": "..."}

    def _get_parameters(self):
        return {
            "type": "object",
            "required": ["param1"],
            "properties": {
                "param1": {"type": "string"}
            }
        }

# Register tool
executor.registry.register(CustomTool(pipeline))
```

## Configuration

```python
config = {
    'max_iterations': 5,  # Maximum agent iterations
    'max_workers': 4,     # Parallel tool execution
}
```
