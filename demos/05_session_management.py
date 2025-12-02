"""
Demo 05: Session Management

Demonstrates:
- Creating and managing conversation sessions
- Multi-turn conversations with context
- History tracking
- Export to different formats (Markdown, JSON, HTML)

Run: python demos/05_session_management.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 80)
    print(" " * 20 + "SESSION MANAGEMENT DEMO")
    print(" " * 20 + "Indonesian Legal Assistant")
    print("=" * 80)
    print()

    from conversation import (
        ConversationManager,
        MarkdownExporter,
        JSONExporter,
        HTMLExporter
    )

    # Step 1: Create manager and start session
    print("[1/6] Creating conversation manager...")
    manager = ConversationManager()
    print("  ✓ Manager created")
    print()

    print("[2/6] Starting new session...")
    session_id = manager.start_session()
    print(f"  ✓ Session created: {session_id}")
    print()

    # Step 2: Simulate multi-turn conversation
    print("[3/6] Simulating multi-turn conversation...")

    conversation = [
        {
            'query': 'Apa itu UU Ketenagakerjaan?',
            'answer': 'UU Ketenagakerjaan adalah Undang-Undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan yang mengatur hubungan kerja antara pekerja dan pengusaha di Indonesia.',
            'metadata': {
                'total_time': 8.5,
                'retrieval_time': 2.1,
                'generation_time': 6.4,
                'tokens_generated': 45,
                'query_type': 'definitional',
                'results_count': 3,
                'sources': [
                    {
                        'regulation_type': 'Undang-Undang',
                        'regulation_number': '13',
                        'year': '2003',
                        'about': 'Ketenagakerjaan',
                        'relevance_score': 0.95
                    }
                ]
            }
        },
        {
            'query': 'Apa hak-hak pekerja dalam undang-undang tersebut?',
            'answer': 'Hak-hak pekerja meliputi: (1) hak atas pekerjaan dan penghasilan yang layak, (2) hak atas jaminan sosial ketenagakerjaan, (3) hak atas keselamatan dan kesehatan kerja, (4) hak untuk berserikat dan berunding bersama.',
            'metadata': {
                'total_time': 7.2,
                'retrieval_time': 1.8,
                'generation_time': 5.4,
                'tokens_generated': 52,
                'query_type': 'general',
                'results_count': 5,
                'sources': [
                    {
                        'regulation_type': 'Undang-Undang',
                        'regulation_number': '13',
                        'year': '2003',
                        'about': 'Ketenagakerjaan',
                        'relevance_score': 0.92
                    }
                ]
            }
        },
        {
            'query': 'Apa sanksi bagi pengusaha yang melanggar?',
            'answer': 'Sanksi bagi pengusaha yang melanggar ketentuan UU Ketenagakerjaan dapat berupa: (1) sanksi administratif seperti teguran, peringatan, atau pembekuan kegiatan, (2) sanksi pidana penjara paling singkat 1 tahun dan paling lama 4 tahun, dan/atau (3) denda paling sedikit Rp 100 juta dan paling banyak Rp 400 juta.',
            'metadata': {
                'total_time': 9.1,
                'retrieval_time': 2.5,
                'generation_time': 6.6,
                'tokens_generated': 68,
                'query_type': 'sanctions',
                'results_count': 4,
                'sources': [
                    {
                        'regulation_type': 'Undang-Undang',
                        'regulation_number': '13',
                        'year': '2003',
                        'about': 'Ketenagakerjaan',
                        'relevance_score': 0.89
                    }
                ]
            }
        }
    ]

    for i, turn in enumerate(conversation, 1):
        print(f"\n  Turn {i}:")
        print(f"    Q: {turn['query']}")
        turn_num = manager.add_turn(
            session_id=session_id,
            query=turn['query'],
            answer=turn['answer'],
            metadata=turn['metadata']
        )
        print(f"    ✓ Added to session (turn #{turn_num})")

    print(f"\n  ✓ Added {len(conversation)} conversation turns")
    print()

    # Step 3: Retrieve history
    print("[4/6] Retrieving conversation history...")
    history = manager.get_history(session_id)
    print(f"  ✓ Retrieved {len(history)} turns")

    # Display history
    for turn in history:
        print(f"\n  Turn {turn['turn_number']}:")
        print(f"    Q: {turn['query'][:60]}...")
        print(f"    A: {turn['answer'][:60]}...")
        print(f"    Time: {turn['metadata'].get('total_time', 0):.1f}s")

    print()

    # Step 4: Get session summary
    print("[5/6] Getting session summary...")
    summary = manager.get_session_summary(session_id)

    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Total time: {summary['total_time']:.1f}s")
    print(f"  Average time per query: {summary['total_time']/summary['total_queries']:.1f}s")
    print()

    # Step 5: Export to different formats
    print("[6/6] Exporting session to different formats...")

    # Get session data
    session_data = manager.get_session(session_id)

    # Create exports directory
    export_dir = Path(__file__).parent / "exports"
    export_dir.mkdir(exist_ok=True)

    # Export to Markdown
    md_exporter = MarkdownExporter()
    md_path = export_dir / f"session_{session_id}.md"
    md_exporter.save_to_file(
        md_exporter.export(session_data),
        md_path
    )
    print(f"  ✓ Markdown: {md_path}")

    # Export to JSON
    json_exporter = JSONExporter({'pretty_print': True})
    json_path = export_dir / f"session_{session_id}.json"
    json_exporter.save_to_file(
        json_exporter.export(session_data),
        json_path
    )
    print(f"  ✓ JSON: {json_path}")

    # Export to HTML
    html_exporter = HTMLExporter()
    html_path = export_dir / f"session_{session_id}.html"
    html_exporter.save_to_file(
        html_exporter.export(session_data),
        html_path
    )
    print(f"  ✓ HTML: {html_path}")

    print()
    print(f"  All exports saved to: {export_dir}/")
    print()

    # End session
    print("[Cleanup] Ending session...")
    final_data = manager.end_session(session_id)
    print(f"  ✓ Session ended")
    print()

    # Summary
    print("=" * 80)
    print("✅ SESSION MANAGEMENT DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Demonstrated features:")
    print("  ✓ Session creation and management")
    print("  ✓ Multi-turn conversation tracking")
    print("  ✓ Conversation history retrieval")
    print("  ✓ Session metadata and statistics")
    print("  ✓ Export to Markdown, JSON, and HTML")
    print()
    print(f"Exported files: {export_dir}/")
    print("  - session_*.md  (human-readable)")
    print("  - session_*.json (machine-readable)")
    print("  - session_*.html (web-viewable)")
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
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
