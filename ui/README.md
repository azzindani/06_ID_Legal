# UI Module

Gradio-based web interface for the Indonesian Legal RAG System.

## Quick Start

```bash
python ui/gradio_app.py
```

Then open http://localhost:7860 in your browser.

## Features

- Chat interface for legal Q&A
- Conversation history tracking
- Session export (Markdown, JSON, HTML)
- Example questions
- Real-time response streaming

## Usage

### Standalone

```python
from ui import launch_app

# Launch with defaults
launch_app()

# Custom configuration
launch_app(
    share=True,      # Create public link
    server_port=7860
)
```

### With Custom Pipeline

```python
import gradio as gr
from ui.gradio_app import create_demo

demo = create_demo()
demo.launch(server_name="0.0.0.0", server_port=7860)
```

## Interface Commands

- Type questions in Indonesian or English
- `/export [md|json|html]` - Export conversation
- `/history` - View conversation history
- `/clear` - Start new session

## Docker

```bash
docker-compose --profile ui up
```

This starts both the API and UI services.

## Configuration

Environment variables:
- `API_URL` - Backend API URL (default: http://localhost:8000)

## Screenshots

The interface includes:
- Main chat panel (left)
- Action buttons (right)
- Export options
- Session info
- Example questions
