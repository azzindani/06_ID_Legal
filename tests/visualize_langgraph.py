"""
Visualization Utility for LangGraph RAG Workflow
Generates ASCII and Mermaid diagrams of the search orchestrator.
"""

import os
import sys
from unittest.mock import MagicMock

# Add project root to sys.path to allow imports from core, utils, etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def visualize_graph():
    print("Initializing LangGraph Orchestrator (with mocks)...")
    
    # Mock dependencies to avoid loading heavy models or needing a dataset
    mock_loader = MagicMock()
    mock_loader.get_all_records_count.return_value = 0
    mock_loader.processed_data = [] # To avoid BM25 division by zero
    
    mock_emb = MagicMock()
    
    mock_rerank = MagicMock()
    mock_rerank.device = "cpu" # Fix for torch.device(str(MagicMock)) error

    
    try:
        from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator
        from config import get_default_config
        
        config = get_default_config()
        
        # Instantiate orchestrator
        orchestrator = LangGraphRAGOrchestrator(
            data_loader=mock_loader,
            embedding_model=mock_emb,
            reranker_model=mock_rerank,
            config=config
        )
        
        # Extract the graph
        # Note: LangGraph compiled apps provide get_graph() for visualization
        graph = orchestrator.app.get_graph()
        
        print("\n" + "="*50)
        print(" LANGGRAPH WORKFLOW VISUALIZATION (ASCII)")
        print("="*50)
        try:
            print(graph.print_ascii())
        except Exception as e:
            print(f"ASCII representation not available: {e}")
            
        print("\n" + "="*50)
        print(" MERMAID GRAPH DEFINITION")
        print("="*50)
        mermaid_code = graph.draw_mermaid()
        print(mermaid_code)
        
        # Save Mermaid definition to a file
        output_file = "tests/langgraph_workflow.md"
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("# LangGraph RAG Workflow Diagram\n\n")
            f.write("This diagram represents the internal state machine flow for the Legal RAG Orchestrator.\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_code)
            f.write("\n```\n")
        
        print(f"\n✅ Created Mermaid diagram in: {output_file}")
        
    except ImportError as e:
        print(f"❌ Error: Could not import core components. Make sure you are in the project root. {e}")
    except Exception as e:
        print(f"❌ Unexpected error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize_graph()
