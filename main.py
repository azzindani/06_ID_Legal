"""
Indonesian Legal RAG System - Main Entry Point

Command-line interface for the Indonesian Legal RAG system.

Usage:
    python main.py                           # Interactive mode
    python main.py --query "your question"   # Single query
    python main.py --export SESSION_ID       # Export session
"""

import argparse
import sys
from typing import Optional

from config import Config
from pipeline import RAGPipeline
from conversation import ConversationManager, MarkdownExporter, JSONExporter, HTMLExporter
from logger_utils import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Indonesian Legal RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Start interactive session
  python main.py --query "Apa itu UU Ketenagakerjaan?"
  python main.py --export abc123 --format md
  python main.py --list-sessions
        """
    )

    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to execute'
    )

    parser.add_argument(
        '--session', '-s',
        type=str,
        help='Session ID to use (creates new if not exists)'
    )

    parser.add_argument(
        '--export', '-e',
        type=str,
        metavar='SESSION_ID',
        help='Export session to file'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['md', 'json', 'html'],
        default='md',
        help='Export format (default: md)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for exports (default: exports)'
    )

    parser.add_argument(
        '--list-sessions',
        action='store_true',
        help='List all active sessions'
    )

    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Disable streaming output'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def print_welcome():
    """Print welcome message"""
    print("\n" + "="*60)
    print("  Indonesian Legal RAG System")
    print("  Sistem Konsultasi Hukum Indonesia")
    print("="*60)
    print("\nKetik pertanyaan Anda atau gunakan perintah:")
    print("  /help     - Tampilkan bantuan")
    print("  /export   - Ekspor percakapan")
    print("  /clear    - Mulai sesi baru")
    print("  /quit     - Keluar")
    print("-"*60 + "\n")


def print_help():
    """Print help message"""
    print("\nPerintah yang tersedia:")
    print("  /help              - Tampilkan bantuan ini")
    print("  /export [format]   - Ekspor sesi (md/json/html)")
    print("  /history           - Tampilkan riwayat percakapan")
    print("  /clear             - Mulai sesi baru")
    print("  /session           - Tampilkan info sesi")
    print("  /quit atau /exit   - Keluar dari program")
    print("\nContoh pertanyaan:")
    print("  - Apa itu UU Ketenagakerjaan?")
    print("  - Apa sanksi pelanggaran UU Perlindungan Konsumen?")
    print("  - Bagaimana prosedur PHK menurut hukum?")
    print()


def export_session(
    manager: ConversationManager,
    session_id: str,
    format_type: str = 'md',
    output_dir: str = 'exports'
) -> Optional[str]:
    """Export session to file"""
    session_data = manager.get_session(session_id)
    if not session_data:
        print(f"Session '{session_id}' tidak ditemukan")
        return None

    exporters = {
        'md': MarkdownExporter,
        'json': JSONExporter,
        'html': HTMLExporter
    }

    exporter_class = exporters.get(format_type, MarkdownExporter)
    exporter = exporter_class()

    path = exporter.export_and_save(session_data, directory=output_dir)
    return str(path)


def run_single_query(args):
    """Run a single query and exit"""
    print("Initializing system...")

    with RAGPipeline() as pipeline:
        if not pipeline.initialize():
            print("Failed to initialize pipeline")
            return 1

        print(f"\nQuery: {args.query}\n")
        print("-" * 40)

        if not args.no_stream:
            # Streaming mode - iterate over generator
            print("\nAnswer: ", end="", flush=True)
            result = None
            for chunk in pipeline.query(args.query, stream=True):
                if chunk.get('type') == 'token':
                    print(chunk.get('token', ''), end='', flush=True)
                elif chunk.get('type') == 'complete':
                    result = chunk
                elif chunk.get('type') == 'error':
                    print(f"\nError: {chunk.get('error')}")
                    return 1
            print()  # New line after streaming
        else:
            # Non-streaming mode
            result = pipeline.query(args.query, stream=False)
            print(f"\n{result['answer']}")

        if args.verbose and result and result.get('metadata'):
            meta = result['metadata']
            print(f"\n[Time: {meta.get('total_time', 0):.2f}s, "
                  f"Tokens: {meta.get('tokens_generated', 0)}]")

    return 0


def run_interactive(args):
    """Run interactive session"""
    print_welcome()

    print("Initializing system...")

    manager = ConversationManager()
    session_id = args.session or manager.start_session()

    print(f"Session ID: {session_id}\n")

    with RAGPipeline() as pipeline:
        if not pipeline.initialize():
            print("Failed to initialize pipeline")
            return 1

        print("System ready!\n")

        while True:
            try:
                user_input = input("Anda: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    cmd_parts = user_input[1:].split()
                    cmd = cmd_parts[0].lower()

                    if cmd in ['quit', 'exit', 'q']:
                        print("\nTerima kasih! Sampai jumpa.")
                        break

                    elif cmd == 'help':
                        print_help()
                        continue

                    elif cmd == 'export':
                        fmt = cmd_parts[1] if len(cmd_parts) > 1 else 'md'
                        output_dir = args.output or 'exports'
                        path = export_session(manager, session_id, fmt, output_dir)
                        if path:
                            print(f"Exported to: {path}")
                        continue

                    elif cmd == 'history':
                        history = manager.get_history(session_id)
                        if not history:
                            print("Belum ada riwayat percakapan")
                        else:
                            for turn in history:
                                print(f"\n[{turn['turn_number']}] Q: {turn['query']}")
                                print(f"    A: {turn['answer'][:100]}...")
                        continue

                    elif cmd == 'clear':
                        session_id = manager.start_session()
                        print(f"Sesi baru dimulai: {session_id}")
                        continue

                    elif cmd == 'session':
                        summary = manager.get_session_summary(session_id)
                        if summary:
                            print(f"\nSession: {summary['session_id']}")
                            print(f"Turns: {summary['total_turns']}")
                            print(f"Tokens: {summary['total_tokens']}")
                            print(f"Time: {summary['total_time']:.2f}s")
                        continue

                    else:
                        print(f"Perintah tidak dikenal: {cmd}")
                        print("Ketik /help untuk bantuan")
                        continue

                # Process query
                print("\nAssistant: ", end="", flush=True)

                # Get conversation context
                context = manager.get_context_for_query(session_id)

                # Execute query
                if not args.no_stream:
                    # Streaming mode - iterate over generator
                    result = None
                    answer_text = ""
                    for chunk in pipeline.query(
                        user_input,
                        conversation_history=context,
                        stream=True
                    ):
                        if chunk.get('type') == 'token':
                            token = chunk.get('token', '')
                            print(token, end='', flush=True)
                            answer_text += token
                        elif chunk.get('type') == 'complete':
                            result = chunk
                            if not answer_text:
                                answer_text = chunk.get('answer', '')
                        elif chunk.get('type') == 'error':
                            print(f"\nError: {chunk.get('error')}")
                            continue
                    print()  # New line after streaming

                    # Use the accumulated answer or from result
                    final_answer = answer_text if answer_text else (result.get('answer', '') if result else '')
                else:
                    # Non-streaming mode
                    result = pipeline.query(
                        user_input,
                        conversation_history=context,
                        stream=False
                    )
                    print(result['answer'])
                    final_answer = result['answer']

                # Save to history
                manager.add_turn(
                    session_id=session_id,
                    query=user_input,
                    answer=final_answer,
                    metadata=result.get('metadata') if result else None
                )

                if args.verbose and result and result.get('metadata'):
                    meta = result['metadata']
                    print(f"\n[Time: {meta.get('total_time', 0):.2f}s, "
                          f"Tokens: {meta.get('tokens_generated', 0)}]")

                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Ketik /quit untuk keluar.")
                continue
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}")
                continue

        # Export on exit if there was conversation
        if manager.get_history(session_id):
            try:
                output_dir = args.output or 'exports'
                path = export_session(manager, session_id, 'json', output_dir)
                if path:
                    print(f"Session saved to: {path}")
            except Exception as e:
                logger.error(f"Failed to save session: {e}")

    return 0


def list_sessions(manager: ConversationManager):
    """List all active sessions"""
    sessions = manager.list_sessions()

    if not sessions:
        print("No active sessions")
        return

    print("\nActive Sessions:")
    print("-" * 60)

    for session in sessions:
        print(f"  {session['id']}")
        print(f"    Created: {session['created_at']}")
        print(f"    Turns: {session['total_turns']}")
        print()


def main():
    """Main entry point"""
    args = parse_arguments()

    try:
        # Handle list sessions
        if args.list_sessions:
            manager = ConversationManager()
            list_sessions(manager)
            return 0

        # Handle export
        if args.export:
            manager = ConversationManager()
            output_dir = args.output or 'exports'
            path = export_session(manager, args.export, args.format, output_dir)
            if path:
                print(f"Exported to: {path}")
                return 0
            return 1

        # Handle single query
        if args.query:
            return run_single_query(args)

        # Run interactive mode
        return run_interactive(args)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
