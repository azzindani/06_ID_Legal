"""
Demo 01: Basic RAG Query

This demonstrates the core end-to-end RAG pipeline:
- Pipeline initialization
- Query execution
- Answer generation
- Source retrieval
- Metadata tracking

Run: python demos/01_basic_rag_query.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 80)
    print(" " * 20 + "INDONESIAN LEGAL ASSISTANT")
    print(" " * 25 + "Basic RAG Demo")
    print("=" * 80)
    print()

    from pipeline import RAGPipeline

    # Step 1: Initialize Pipeline
    print("[1/5] Initializing pipeline...")
    print("  ⏳ Loading models (this may take 30-90 seconds on first run)...")
    start_init = time.time()

    pipeline = RAGPipeline()

    try:
        success = pipeline.initialize()
        init_time = time.time() - start_init

        if not success:
            print("  ❌ Failed to initialize pipeline")
            return 1

        print(f"  ✓ Models loaded ({init_time:.1f}s)")
        print(f"  ✓ Dataset loaded")
        print(f"  ✓ Pipeline ready")
        print()

    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return 1

    # Step 2: Run Query
    test_query = "Apa itu UU Ketenagakerjaan?"
    print(f'[2/5] Running query: "{test_query}"')
    print("  ⏳ Processing...")
    start_query = time.time()

    try:
        result = pipeline.query(test_query)
        query_time = time.time() - start_query

        if not result or not result.get('success', True):
            print(f"  ❌ Query failed: {result.get('error', 'Unknown error')}")
            pipeline.shutdown()
            return 1

        print(f"  ✓ Query executed in {query_time:.1f}s")
        print()

    except Exception as e:
        print(f"  ❌ Query failed: {e}")
        pipeline.shutdown()
        return 1

    # Step 3: Display Answer
    print("[3/5] Answer:")
    print("-" * 80)
    answer = result.get('answer', 'No answer generated')

    # Truncate long answers for demo
    if len(answer) > 500:
        print(answer[:500] + "...")
        print(f"\n[Answer truncated - full length: {len(answer)} characters]")
    else:
        print(answer)
    print("-" * 80)
    print()

    # Step 4: Display Sources
    sources = result.get('sources', [])
    print(f"[4/5] Sources Retrieved: {len(sources)}")
    if sources:
        for i, source in enumerate(sources[:5], 1):  # Show top 5
            reg_type = source.get('regulation_type', 'Unknown')
            reg_num = source.get('regulation_number', '?')
            year = source.get('year', '?')
            about = source.get('about', 'Unknown')
            score = source.get('relevance_score', 0.0)

            print(f"  {i}. {reg_type} {reg_num}/{year} - {about}")
            print(f"     Relevance: {score:.3f}")

        if len(sources) > 5:
            print(f"  ... and {len(sources) - 5} more sources")
    else:
        print("  ⚠️  No sources retrieved")
    print()

    # Step 5: Display Metadata
    metadata = result.get('metadata', {})
    print("[5/5] Performance Metadata:")
    print(f"  Total time: {metadata.get('total_time', 0):.2f}s")
    print(f"  Retrieval time: {metadata.get('retrieval_time', 0):.2f}s")
    print(f"  Generation time: {metadata.get('generation_time', 0):.2f}s")
    print(f"  Tokens generated: {metadata.get('tokens_generated', 0)}")
    print(f"  Query type: {metadata.get('query_type', 'unknown')}")
    print(f"  Results count: {metadata.get('results_count', 0)}")
    print()

    # Cleanup
    print("[Cleanup] Shutting down pipeline...")
    pipeline.shutdown()
    print("  ✓ Resources released")
    print()

    # Summary
    print("=" * 80)
    print("✅ BASIC RAG DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Initialization: {init_time:.1f}s")
    print(f"  - Query execution: {query_time:.1f}s")
    print(f"  - Answer length: {len(answer)} characters")
    print(f"  - Sources found: {len(sources)}")
    print(f"  - Status: SUCCESS")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
