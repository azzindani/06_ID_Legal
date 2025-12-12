"""
Conversation Context Auditing - Utilities for Auditing Conversation Memory and Context

Provides functions to print and analyze conversation context, memory, and history
for debugging and verification purposes.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


def format_conversation_context(
    session_data: Dict[str, Any],
    show_full_content: bool = False,
    max_turns: Optional[int] = None
) -> str:
    """
    Format conversation context for auditing and debugging

    Shows:
    - Session metadata
    - All conversation turns with Q&A
    - Context used for each turn
    - Memory/history tracking

    Args:
        session_data: Session data from ConversationManager
        show_full_content: Show full answer content (default: False, show truncated)
        max_turns: Maximum number of turns to show (default: None, show all)

    Returns:
        Formatted string with conversation context audit
    """
    lines = []

    lines.append("\n" + "=" * 100)
    lines.append("CONVERSATION CONTEXT AUDIT")
    lines.append("=" * 100)
    lines.append("")

    # Session Metadata
    lines.append("### 📋 Session Metadata")
    lines.append("-" * 80)
    lines.append(f"Session ID: {session_data.get('id', 'N/A')}")
    lines.append(f"Created: {session_data.get('created_at', 'N/A')}")
    lines.append(f"Updated: {session_data.get('updated_at', 'N/A')}")

    metadata = session_data.get('metadata', {})
    lines.append(f"Total Queries: {metadata.get('total_queries', 0)}")
    lines.append(f"Total Tokens: {metadata.get('total_tokens', 0)}")
    lines.append(f"Total Time: {metadata.get('total_time', 0):.2f}s")
    lines.append("")

    # Conversation Turns
    turns = session_data.get('turns', [])
    total_turns = len(turns)

    if max_turns:
        turns_to_show = turns[:max_turns]
        remaining = total_turns - max_turns
    else:
        turns_to_show = turns
        remaining = 0

    lines.append(f"### 💬 Conversation Turns ({len(turns_to_show)}/{total_turns})")
    lines.append("-" * 80)
    lines.append("")

    for idx, turn in enumerate(turns_to_show, 1):
        lines.append(f"#### Turn {idx}")
        lines.append("")

        # User Question (ConversationManager stores as 'query', not 'user_message')
        user_msg = turn.get('query', turn.get('user_message', ''))
        lines.append(f"**👤 User Question:**")
        lines.append(f"```")
        lines.append(user_msg)
        lines.append(f"```")
        lines.append("")

        # Context Used (if available)
        turn_metadata = turn.get('metadata', {})
        context_info = turn_metadata.get('context_info', {})

        if context_info:
            lines.append(f"**🧠 Context Used:**")
            lines.append(f"   - Previous Turns Referenced: {context_info.get('previous_turns', 0)}")
            lines.append(f"   - Context Window Size: {context_info.get('context_window', 0)}")

            if context_info.get('is_followup'):
                lines.append(f"   - ✅ Detected as Follow-up Question")

            if context_info.get('topic_shift'):
                lines.append(f"   - 🔄 Topic Shift Detected: {context_info.get('new_topic', 'Unknown')}")

            lines.append("")

        # Query Analysis (if available)
        query_analysis = turn_metadata.get('query_analysis', {})
        if query_analysis:
            lines.append(f"**🔍 Query Analysis:**")
            lines.append(f"   - Query Type: {query_analysis.get('query_type', 'general')}")
            lines.append(f"   - Complexity: {query_analysis.get('complexity_score', 0):.2f}")

            if query_analysis.get('specific_regulation'):
                lines.append(f"   - Specific Regulation: {query_analysis.get('specific_regulation')}")

            entities = query_analysis.get('entities', [])
            if entities:
                lines.append(f"   - Entities: {', '.join(entities[:5])}")

            lines.append("")

        # Assistant Answer (ConversationManager stores as 'answer', not 'assistant_message')
        assistant_msg = turn.get('answer', turn.get('assistant_message', ''))

        if show_full_content:
            lines.append(f"**🤖 Assistant Answer:**")
            lines.append(f"```")
            lines.append(assistant_msg)
            lines.append(f"```")
        else:
            # Truncated version
            truncated = assistant_msg[:300] + "..." if len(assistant_msg) > 300 else assistant_msg
            lines.append(f"**🤖 Assistant Answer (truncated):**")
            lines.append(f"```")
            lines.append(truncated)
            lines.append(f"```")

        lines.append("")

        # Documents Retrieved
        docs_count = turn_metadata.get('results_count', 0)
        lines.append(f"**📚 Documents Retrieved:** {docs_count}")

        # Timing
        if turn_metadata.get('total_time'):
            lines.append(f"**⏱️  Processing Time:** {turn_metadata.get('total_time', 0):.2f}s")

        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    if remaining > 0:
        lines.append(f"... and {remaining} more turns (use max_turns=None to show all)")
        lines.append("")

    lines.append("=" * 100)
    lines.append("")

    return '\n'.join(lines)


def format_context_window(
    conversation_history: List[Dict[str, str]],
    current_query: str
) -> str:
    """
    Format the context window being used for current query

    Shows what conversation history is being passed to the pipeline

    Args:
        conversation_history: List of previous turns
        current_query: Current user query

    Returns:
        Formatted string showing context window
    """
    lines = []

    lines.append("\n### 🪟 Context Window")
    lines.append("-" * 80)
    lines.append(f"Previous Turns in Context: {len(conversation_history)}")
    lines.append("")

    for idx, turn in enumerate(conversation_history, 1):
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')

        # Truncate for readability
        if len(content) > 150:
            content = content[:150] + "..."

        lines.append(f"{idx}. [{role.upper()}]: {content}")

    lines.append("")
    lines.append(f"Current Query: {current_query}")
    lines.append("-" * 80)

    return '\n'.join(lines)


def print_conversation_memory_summary(session_data: Dict[str, Any]) -> None:
    """
    Print quick summary of conversation memory for debugging

    Args:
        session_data: Session data from ConversationManager
    """
    print("\n" + "=" * 80)
    print("CONVERSATION MEMORY SUMMARY")
    print("=" * 80)

    turns = session_data.get('turns', [])
    metadata = session_data.get('metadata', {})

    print(f"Session ID: {session_data.get('id', 'N/A')}")
    print(f"Total Turns: {len(turns)}")
    print(f"Total Queries: {metadata.get('total_queries', 0)}")
    print(f"Total Time: {metadata.get('total_time', 0):.2f}s")
    print("")

    if turns:
        print("Recent Turns:")
        for idx, turn in enumerate(turns[-3:], len(turns) - 2):
            user_msg = turn.get('user_message', '')[:80]
            print(f"  {idx}. {user_msg}...")

    print("=" * 80)
    print("")


def export_conversation_audit(
    session_data: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Export full conversation audit to file

    Args:
        session_data: Session data from ConversationManager
        output_path: Path to save audit report

    Returns:
        True if successful, False otherwise
    """
    try:
        audit_content = format_conversation_context(
            session_data,
            show_full_content=True,
            max_turns=None
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(audit_content)

        return True
    except Exception as e:
        print(f"Error exporting conversation audit: {e}")
        return False
