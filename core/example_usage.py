"""
Simple Usage Example - Complete RAG Pipeline
Demonstrates how to use the complete Indonesian Legal RAG system

File: example_usage.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

from utils.logger_utils import initialize_logging, get_logger
from config import get_default_config, DATASET_NAME, EMBEDDING_DIM
from loader.dataloader import EnhancedKGDatasetLoader
from core.model_manager import get_model_manager
from core.search.langgraph_orchestrator import LangGraphRAGOrchestrator
from core.generation import GenerationEngine


def simple_rag_query(query: str):
    """
    Simple function to query the RAG system
    
    Args:
        query: Legal question in Indonesian
        
    Returns:
        Answer with citations
    """
    logger = get_logger("Example")
    
    logger.info(f"Query: {query}")
    
    # Initialize (do this once at startup)
    logger.info("Loading system...")
    
    # 1. Load configuration
    config = get_default_config()
    
    # 2. Load dataset
    data_loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)
    if not data_loader.load_from_huggingface(lambda msg: logger.info(f"  {msg}")):
        logger.error("Failed to load dataset")
        return None
    
    # 3. Load models
    model_manager = get_model_manager()
    embedding_model = model_manager.load_embedding_model()
    reranker_model = model_manager.load_reranker_model(use_mock=True)
    
    # 4. Create search orchestrator
    search_orchestrator = LangGraphRAGOrchestrator(
        data_loader=data_loader,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        config=config
    )
    
    # 5. Create generation engine
    generation_engine = GenerationEngine(config)
    generation_engine.initialize()
    
    logger.info("System ready!")
    
    # Search for relevant documents
    logger.info("Searching...")
    search_result = search_orchestrator.run(query)
    
    if search_result.get('errors'):
        logger.error(f"Search failed: {search_result['errors']}")
        return None
    
    final_results = search_result.get('final_results', [])
    logger.info(f"Found {len(final_results)} relevant documents")
    
    if len(final_results) == 0:
        return {
            'success': True,
            'answer': "Maaf, tidak ditemukan dokumen yang relevan.",
            'citations': [],
            'metadata': {'generation_time': 0, 'tokens_generated': 0}
        }
    
    # Generate answer
    logger.info("Generating answer...")
    answer_result = generation_engine.generate_answer(
        query=query,
        retrieved_results=final_results,
        query_analysis=search_result.get('query_analysis'),
        stream=False
    )
    
    if not answer_result['success']:
        logger.error(f"Generation failed: {answer_result.get('error')}")
        return None
    
    # Cleanup
    generation_engine.shutdown()
    model_manager.unload_models()
    
    return answer_result


def main():
    """Main example"""
    
    # Initialize logging
    initialize_logging(
        enable_file_logging=True,
        log_dir="logs",
        log_filename="example_usage.log"
    )
    
    print("=" * 80)
    print("Indonesian Legal RAG System - Simple Example")
    print("=" * 80)
    
    # Example queries
    queries = [
        "Apa sanksi pidana dalam UU ITE?",
        "Bagaimana prosedur pengadaan barang pemerintah?",
        "Apa definisi pelanggaran administratif?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}/{len(queries)}")
        print(f"{'='*80}")
        print(f"Query: {query}\n")
        
        result = simple_rag_query(query)
        
        if result:
            print("\n--- ANSWER ---")
            print(result['answer'])
            
            print("\n--- CITATIONS ---")
            for citation in result.get('citations', [])[:3]:
                print(f"{citation['id']}. {citation['citation_text']}")
            
            print("\n--- METADATA ---")
            metadata = result['metadata']
            print(f"Generation Time: {metadata.get('generation_time', 0):.2f}s")
            print(f"Tokens Generated: {metadata.get('tokens_generated', 0)}")
            print(f"Quality Score: {metadata.get('validation', {}).get('quality_score', 0):.2f}")
        else:
            print("Failed to generate answer")
        
        print(f"\n{'='*80}\n")


def interactive_mode():
    """Interactive query mode"""
    
    initialize_logging(
        enable_file_logging=True,
        log_dir="logs",
        log_filename="interactive.log"
    )
    
    logger = get_logger("Interactive")
    
    print("=" * 80)
    print("Indonesian Legal RAG System - Interactive Mode")
    print("=" * 80)
    print("\nInitializing system...")
    
    # Initialize system once
    config = get_default_config()
    
    logger.info("Loading dataset...")
    data_loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)
    if not data_loader.load_from_huggingface(lambda msg: print(f"  {msg}")):
        print("Failed to load dataset")
        return
    
    logger.info("Loading models...")
    model_manager = get_model_manager()
    embedding_model = model_manager.load_embedding_model()
    reranker_model = model_manager.load_reranker_model(use_mock=True)
    
    search_orchestrator = LangGraphRAGOrchestrator(
        data_loader=data_loader,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        config=config
    )
    
    generation_engine = GenerationEngine(config)
    if not generation_engine.initialize():
        print("Warning: LLM not available, will use fallback")
    
    print("\nSystem ready! Type 'exit' to quit.\n")
    
    conversation_history = []
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            print("\nSearching...")
            
            # Search
            search_result = search_orchestrator.run(query)
            final_results = search_result.get('final_results', [])
            
            if len(final_results) == 0:
                print("\nAssistant: Maaf, tidak ditemukan dokumen yang relevan.\n")
                continue
            
            print(f"Found {len(final_results)} documents. Generating answer...\n")
            
            # Generate
            answer_result = generation_engine.generate_answer(
                query=query,
                retrieved_results=final_results,
                query_analysis=search_result.get('query_analysis'),
                conversation_history=conversation_history,
                stream=False
            )
            
            if answer_result['success']:
                answer = answer_result['answer']
                print(f"Assistant: {answer}\n")
                
                # Update conversation history
                conversation_history.append({'role': 'user', 'content': query})
                conversation_history.append({'role': 'assistant', 'content': answer})
                
                # Keep only last 5 turns
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
            else:
                print(f"Error: {answer_result.get('error')}\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            logger.error(f"Error in interactive mode: {e}")
    
    # Cleanup
    generation_engine.shutdown()
    model_manager.unload_models()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Indonesian Legal RAG System Example')
    parser.add_argument(
        '--mode',
        choices=['batch', 'interactive'],
        default='batch',
        help='Run mode: batch examples or interactive'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_mode()
    else:
        main()