"""
Search Engine UI - Indonesian Legal RAG System

Web-based search interface for legal document search using RAG.
This UI focuses on document retrieval without conversational features.
"""

import gradio as gr
import sys
import os
from typing import List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RAGPipeline
from providers import list_providers
from config import LLM_PROVIDER, EMBEDDING_DEVICE, LLM_DEVICE
from logger_utils import get_logger

logger = get_logger(__name__)

# Global instances
pipeline: Optional[RAGPipeline] = None
current_provider: str = LLM_PROVIDER

# Custom CSS for search interface
SEARCH_CSS = """
/* Base responsive sizing */
html { font-size: 16px; }

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header styling */
.header-title {
    text-align: center;
    font-size: 2em;
    font-weight: 700;
    color: #1e3a5f;
    margin-bottom: 0.5em;
}

.header-subtitle {
    text-align: center;
    font-size: 1em;
    color: #666;
    margin-bottom: 1.5em;
}

/* Search box styling */
.search-box textarea {
    font-size: 1.1em !important;
    padding: 1em !important;
    border-radius: 0.5em !important;
    border: 2px solid #e0e0e0 !important;
}

.search-box textarea:focus {
    border-color: #1e3a5f !important;
    box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.1) !important;
}

/* Results styling */
.results-container {
    margin-top: 1.5em;
}

.result-card {
    background: #f8f9fa;
    border-radius: 0.5em;
    padding: 1em;
    margin-bottom: 1em;
    border-left: 4px solid #1e3a5f;
}

/* Metadata styling */
.metadata-box {
    background: #f0f4f8;
    border-radius: 0.5em;
    padding: 1em;
    font-family: 'Fira Code', monospace;
    font-size: 0.85em;
    max-height: 400px;
    overflow-y: auto;
}

/* Button styling */
.search-btn {
    background: #1e3a5f !important;
    color: white !important;
    font-size: 1em !important;
    padding: 0.8em 2em !important;
    border-radius: 0.5em !important;
}

.search-btn:hover {
    background: #2c5282 !important;
}

/* Responsive breakpoints */
@media (max-width: 768px) {
    html { font-size: 14px; }
    .gradio-container { padding: 1em !important; }
}

@media (min-width: 1200px) {
    html { font-size: 18px; }
}
"""


def initialize_search_system(provider_type: str = None):
    """Initialize the RAG system for search"""
    global pipeline, current_provider

    if provider_type:
        current_provider = provider_type

    if pipeline is None:
        logger.info(f"Initializing search system with provider: {current_provider}")
        pipeline = RAGPipeline({'llm_provider': current_provider})
        if not pipeline.initialize():
            return "Failed to initialize pipeline"
        logger.info("Search pipeline initialized")

    device_info = f"Embedding: {EMBEDDING_DEVICE}, LLM: {LLM_DEVICE}"
    return f"✓ Initialized with {current_provider}. {device_info}"


def search_documents(query: str, num_results: int = 10, show_scores: bool = True) -> tuple:
    """
    Search for legal documents based on query.
    Returns formatted results and metadata.
    """
    global pipeline

    if not query.strip():
        return "Please enter a search query.", ""

    try:
        if pipeline is None:
            initialize_search_system()

        # Perform search using pipeline's search component
        logger.info(f"Searching for: {query}")

        # Use pipeline query but focus on search results
        result = pipeline.query(query, stream=False)

        # Format search results
        results_md = format_search_results(result, show_scores)
        metadata_md = format_search_metadata(result)

        return results_md, metadata_md

    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error during search: {str(e)}", ""


def format_search_results(result: dict, show_scores: bool = True) -> str:
    """Format search results for display"""
    parts = []

    # Header with answer summary
    if result.get('answer'):
        parts.append("## 📋 Ringkasan Hasil\n")
        parts.append(result['answer'])
        parts.append("\n\n---\n")

    # Document sources
    sources = result.get('sources', [])
    if sources:
        parts.append("## 📚 Dokumen Ditemukan\n\n")

        for i, source in enumerate(sources, 1):
            title = source.get('title', f'Dokumen {i}')
            content = source.get('content', source.get('text', ''))[:500]
            score = source.get('score', source.get('relevance_score', 0))
            doc_type = source.get('type', source.get('jenis_peraturan', 'Peraturan'))

            parts.append(f"### {i}. {title}\n")
            if show_scores:
                parts.append(f"**Skor Relevansi:** {score:.3f}\n")
            parts.append(f"**Jenis:** {doc_type}\n\n")
            parts.append(f"{content}{'...' if len(content) >= 500 else ''}\n\n")
            parts.append("---\n\n")
    else:
        parts.append("*Tidak ditemukan dokumen yang relevan.*\n")

    return "".join(parts)


def format_search_metadata(result: dict) -> str:
    """Format search metadata for display"""
    parts = []

    # Search phases info
    search_phases = result.get('search_phases', result.get('metadata', {}).get('search_phases', []))
    if search_phases:
        parts.append("### Search Phases\n")
        for phase in search_phases:
            phase_name = phase.get('phase', 'unknown')
            docs_found = phase.get('documents_found', 0)
            parts.append(f"- {phase_name}: {docs_found} docs\n")
        parts.append("\n")

    # Team analysis if available
    team_analysis = result.get('team_analysis', result.get('metadata', {}).get('team_analysis', {}))
    if team_analysis:
        parts.append("### Research Team Analysis\n")
        consensus = team_analysis.get('consensus_reached', False)
        confidence = team_analysis.get('confidence', 0)
        parts.append(f"- Consensus: {'Yes' if consensus else 'No'}\n")
        parts.append(f"- Confidence: {confidence:.2f}\n")

        perspectives = team_analysis.get('perspectives', [])
        if perspectives:
            parts.append("- Perspectives:\n")
            for p in perspectives[:3]:
                role = p.get('role', 'analyst')
                parts.append(f"  - {role}\n")
        parts.append("\n")

    # Performance metrics
    metadata = result.get('metadata', {})
    if metadata:
        parts.append("### Performance\n")
        total_time = metadata.get('total_time', 0)
        total_docs = metadata.get('total_documents_retrieved', 0)
        parts.append(f"- Total time: {total_time:.2f}s\n")
        parts.append(f"- Documents retrieved: {total_docs}\n")

    return "".join(parts) if parts else "No metadata available"


def create_search_demo():
    """Create the search engine Gradio interface"""

    with gr.Blocks(css=SEARCH_CSS, title="Indonesian Legal Search Engine") as demo:
        # Header
        gr.HTML("""
            <div class="header-title">🔍 Indonesian Legal Search Engine</div>
            <div class="header-subtitle">Search Indonesian regulations and legal documents</div>
        """)

        # Status
        status = gr.Textbox(
            label="System Status",
            value="Ready to initialize...",
            interactive=False
        )

        # Search input
        with gr.Row():
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter your legal search query in Indonesian or English...",
                lines=3,
                elem_classes=["search-box"]
            )

        # Search options
        with gr.Row():
            num_results = gr.Slider(
                minimum=5,
                maximum=50,
                value=10,
                step=5,
                label="Number of Results"
            )
            show_scores = gr.Checkbox(
                value=True,
                label="Show Relevance Scores"
            )

        # Search button
        search_btn = gr.Button(
            "🔍 Search",
            variant="primary",
            elem_classes=["search-btn"]
        )

        # Results
        with gr.Row():
            with gr.Column(scale=2):
                results_output = gr.Markdown(
                    label="Search Results",
                    elem_classes=["results-container"]
                )
            with gr.Column(scale=1):
                metadata_output = gr.Markdown(
                    label="Search Metadata",
                    elem_classes=["metadata-box"]
                )

        # Example queries
        gr.Examples(
            examples=[
                ["Apa saja syarat pendirian PT menurut UU Perseroan Terbatas?"],
                ["Bagaimana prosedur pendaftaran merek dagang di Indonesia?"],
                ["Apa sanksi pelanggaran UU Perlindungan Data Pribadi?"],
                ["Ketentuan cuti karyawan menurut UU Ketenagakerjaan"],
                ["Syarat dan prosedur perceraian menurut hukum Indonesia"],
                ["Regulasi tentang investasi asing di Indonesia"],
                ["Ketentuan pajak penghasilan untuk UMKM"],
                ["Prosedur penyelesaian sengketa konsumen"],
            ],
            inputs=query_input,
            label="Example Queries"
        )

        # Event handlers
        search_btn.click(
            fn=search_documents,
            inputs=[query_input, num_results, show_scores],
            outputs=[results_output, metadata_output]
        )

        query_input.submit(
            fn=search_documents,
            inputs=[query_input, num_results, show_scores],
            outputs=[results_output, metadata_output]
        )

        # Initialize on load
        demo.load(
            fn=lambda: initialize_search_system(),
            outputs=status
        )

    return demo


def launch_search_app(share: bool = False, server_port: int = 7861):
    """Launch search engine app with pre-initialization"""
    global pipeline

    # Pre-initialize system
    logger.info("Pre-initializing search system before UI launch...")

    if pipeline is None:
        logger.info(f"Initializing RAG pipeline with provider: {current_provider}")
        pipeline = RAGPipeline({'llm_provider': current_provider})
        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            raise RuntimeError("Pipeline initialization failed")
        logger.info("Pipeline initialized successfully")

    logger.info("Search system ready, launching UI...")

    # Create and launch demo
    demo = create_search_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    launch_search_app(share=False)
