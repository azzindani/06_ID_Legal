# Conversation Module

Session management and export functionality for the Indonesian Legal RAG system.

## Purpose

The Conversation module provides:
- Session creation and management
- Conversation history tracking with metadata
- Context retrieval for follow-up questions
- Export to multiple formats (Markdown, JSON, HTML)

## Components

| File | Description |
|------|-------------|
| `manager.py` | ConversationManager class |
| `export/base_exporter.py` | Abstract base exporter |
| `export/markdown_exporter.py` | Markdown export |
| `export/json_exporter.py` | JSON export |
| `export/html_exporter.py` | HTML export |

## Usage

### Basic Session Management

```python
from conversation import ConversationManager

# Create manager
manager = ConversationManager()

# Start session
session_id = manager.start_session()

# Add conversation turn
manager.add_turn(
    session_id=session_id,
    query="Apa sanksi pelanggaran UU Ketenagakerjaan?",
    answer="Sanksi pelanggaran meliputi...",
    metadata={
        'total_time': 5.2,
        'tokens_generated': 150,
        'query_type': 'sanctions',
        'citations': [
            {
                'regulation_type': 'Undang-Undang',
                'regulation_number': '13',
                'year': '2003',
                'about': 'Ketenagakerjaan'
            }
        ]
    }
)

# Get history for RAG context
context = manager.get_context_for_query(session_id)

# Get session summary
summary = manager.get_session_summary(session_id)

# End session
final_data = manager.end_session(session_id)
```

### With RAG Pipeline

```python
from pipeline import RAGPipeline
from conversation import ConversationManager

pipeline = RAGPipeline()
pipeline.initialize()

manager = ConversationManager()
session_id = manager.start_session()

# First query
result1 = pipeline.query("Apa itu UU Ketenagakerjaan?")
manager.add_turn(session_id, result1['metadata']['question'], result1['answer'], result1['metadata'])

# Follow-up with context
context = manager.get_context_for_query(session_id)
result2 = pipeline.query("Apa sanksinya?", conversation_history=context)
manager.add_turn(session_id, result2['metadata']['question'], result2['answer'], result2['metadata'])

pipeline.shutdown()
```

### Export Conversations

```python
from conversation import ConversationManager, MarkdownExporter, JSONExporter, HTMLExporter

manager = ConversationManager()
session_id = manager.start_session()

# ... add turns ...

# Get session data
session_data = manager.get_session(session_id)

# Export to Markdown
md_exporter = MarkdownExporter()
md_content = md_exporter.export(session_data)
md_exporter.save_to_file(md_content, 'my_conversation.md')

# Export to JSON
json_exporter = JSONExporter({'pretty_print': True})
json_path = json_exporter.export_and_save(session_data, directory='exports')

# Export to HTML
html_exporter = HTMLExporter()
html_path = html_exporter.export_and_save(session_data)
```

### Export Configuration

```python
# Configure what to include in exports
config = {
    'include_metadata': True,    # Include timing, scores
    'include_sources': True,     # Include citations
    'include_timing': True,      # Include timing details
    'include_thinking': False,   # Include LLM thinking (if any)
    'pretty_print': True,        # For JSON: format nicely
    'indent': 2                  # For JSON: indentation
}

exporter = MarkdownExporter(config)
```

## API Reference

### ConversationManager

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `start_session(session_id)` | Start new session | `str` (session_id) |
| `add_turn(session_id, query, answer, metadata)` | Add conversation turn | `int` (turn_number) |
| `get_history(session_id, max_turns)` | Get full history | `List[Dict]` |
| `get_context_for_query(session_id, max_turns)` | Get RAG context | `List[Dict]` |
| `get_session(session_id)` | Get complete session | `Dict` |
| `get_session_summary(session_id)` | Get session stats | `Dict` |
| `end_session(session_id)` | End and remove session | `Dict` |
| `list_sessions()` | List all sessions | `List[Dict]` |
| `clear_all_sessions()` | Clear all sessions | `None` |
| `get_last_turn(session_id)` | Get last turn | `Dict` |
| `search_history(session_id, keyword)` | Search history | `List[Dict]` |

#### Session Data Structure

```python
{
    'id': 'uuid-string',
    'created_at': 'ISO-timestamp',
    'updated_at': 'ISO-timestamp',
    'turns': [
        {
            'turn_number': 1,
            'timestamp': 'ISO-timestamp',
            'query': 'user question',
            'answer': 'assistant answer',
            'metadata': {
                'total_time': 5.2,
                'retrieval_time': 2.1,
                'generation_time': 3.1,
                'tokens_generated': 150,
                'query_type': 'sanctions',
                'results_count': 3,
                'citations': [...],
                'sources': [...]
            }
        }
    ],
    'metadata': {
        'total_queries': 1,
        'total_tokens': 150,
        'total_time': 5.2,
        'regulations_cited': ['UU 13/2003']
    }
}
```

### Exporters

#### Common Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `export(session_data)` | Export to string | `str` |
| `save_to_file(content, filename, directory)` | Save to file | `Path` |
| `export_and_save(session_data, filename, directory)` | Export and save | `Path` |
| `get_file_extension()` | Get file extension | `str` |

#### Export Formats

| Format | Extension | Features |
|--------|-----------|----------|
| Markdown | `.md` | Collapsible details, tables |
| JSON | `.json` | Complete metadata, structured |
| HTML | `.html` | Styled, responsive, printable |

## Configuration Options

### ConversationManager

| Option | Default | Description |
|--------|---------|-------------|
| `max_history_turns` | 50 | Maximum turns to keep |
| `max_context_turns` | 5 | Default context turns for RAG |

### Exporters

| Option | Default | Description |
|--------|---------|-------------|
| `include_metadata` | True | Include turn metadata |
| `include_sources` | True | Include citations |
| `include_timing` | True | Include timing details |
| `include_thinking` | False | Include LLM thinking |
| `pretty_print` | True | JSON: format nicely |
| `indent` | 2 | JSON: indentation level |

## Testing

```bash
# Run all conversation tests
pytest conversation/tests/ -v

# Run specific tests
pytest conversation/tests/test_manager.py -v
pytest conversation/tests/test_exporters.py -v

# Run with coverage
pytest conversation/tests/ --cov=conversation --cov-report=html
```

## Export Examples

### Markdown Output

```markdown
# Konsultasi Hukum Indonesia

**Session ID:** `abc-123`
**Tanggal:** 2024-01-15 10:30:00

## Ringkasan Sesi

- **Total Pertanyaan:** 2
- **Total Token:** 300
- **Total Waktu:** 10.5s

## Percakapan

### Pertanyaan 1

> Apa sanksi pelanggaran UU Ketenagakerjaan?

### Jawaban 1

Sanksi pelanggaran meliputi...

<details>
<summary>Detail (2024-01-15 10:30:00)</summary>
- **Waktu Total:** 5.2s
- **Token:** 150
</details>
```

### JSON Output

```json
{
  "export_info": {
    "format": "json",
    "version": "1.0",
    "exported_at": "2024-01-15T10:35:00"
  },
  "session": {
    "id": "abc-123",
    "summary": {
      "total_turns": 2,
      "total_tokens": 300
    },
    "turns": [...]
  }
}
```

## Dependencies

- `logger_utils` - Logging
- `uuid` - Session ID generation
- `datetime` - Timestamps
- `pathlib` - File operations
- `json` - JSON export

## Future Enhancements

- [ ] PDF export
- [ ] Session persistence (SQLite/Redis)
- [ ] Session sharing/import
- [ ] Compression for large sessions
- [ ] Async export for large files
