"""
Context Transfer - Smart Provider Switching

Handles context preservation when switching between
LLM providers mid-conversation.

File: core/llm_providers/context_transfer.py
"""

from typing import Dict, Any, List, Optional

# Import logger
try:
    from utils.logger_utils import get_logger
    logger = get_logger("ContextTransfer")
except ImportError:
    import logging
    logger = logging.getLogger("ContextTransfer")


# Context window sizes for common models
MODEL_CONTEXT_WINDOWS = {
    # Free models
    "nvidia/nemotron-3-nano-30b-a3b:free": 32768,
    "google/gemini-2.0-flash-exp:free": 1048576,  # 1M
    "openai/gpt-oss-120b:free": 131072,
    
    # Premium models
    "anthropic/claude-sonnet-4": 200000,
    "openai/gpt-4o": 128000,
    "openai/gpt-4o-mini": 128000,
    "deepseek/deepseek-r1": 65536,
    
    # Local models
    "local": 32768,
}


class ContextTransfer:
    """
    Handle context preservation when switching providers.
    
    Features:
    - Context window validation
    - Conversation summarization for context fit
    - Format adaptation between providers
    - Warning generation for potential issues
    
    Usage:
        transfer = ContextTransfer()
        
        # Check compatibility
        warnings = transfer.check_compatibility(
            from_model="claude-sonnet-4",  # 200K
            to_model="gpt-4o-mini",         # 128K
            conversation_tokens=150000
        )
        
        # Prepare context for new provider
        context = transfer.prepare_context(
            conversation=history,
            to_model="gpt-4o-mini"
        )
    """
    
    def __init__(self):
        self.context_windows = MODEL_CONTEXT_WINDOWS.copy()
    
    def get_context_window(self, model: str) -> int:
        """
        Get context window size for a model.
        
        Args:
            model: Model identifier
            
        Returns:
            Context window in tokens (default 32768 if unknown)
        """
        return self.context_windows.get(model, 32768)
    
    def check_compatibility(
        self,
        from_model: str,
        to_model: str,
        conversation_tokens: int = 0
    ) -> List[str]:
        """
        Check compatibility between providers.
        
        Args:
            from_model: Source model
            to_model: Target model
            conversation_tokens: Estimated tokens in conversation
            
        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        
        from_window = self.get_context_window(from_model)
        to_window = self.get_context_window(to_model)
        
        # Context window reduction warning
        if to_window < from_window:
            reduction = ((from_window - to_window) / from_window) * 100
            warnings.append(
                f"⚠️ Context window reduced: {from_model} ({from_window:,} tokens) → "
                f"{to_model} ({to_window:,} tokens) [-{reduction:.0f}%]"
            )
        
        # Check if conversation fits
        if conversation_tokens > to_window * 0.8:  # 80% threshold
            warnings.append(
                f"⚠️ Conversation ({conversation_tokens:,} tokens) may not fully fit "
                f"in {to_model}'s context ({to_window:,} tokens). "
                f"Earlier messages may be truncated."
            )
        
        return warnings
    
    def prepare_context(
        self,
        conversation: List[Dict[str, str]],
        to_model: str,
        max_tokens: int = None,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare conversation context for a new provider.
        
        Args:
            conversation: List of {role, content} messages
            to_model: Target model
            max_tokens: Override max tokens (default: 80% of context window)
            include_summary: Whether to add summary prefix
            
        Returns:
            Dict with:
            - messages: Prepared messages
            - summary: Optional conversation summary
            - truncated: Whether truncation occurred
            - warnings: Any warnings
        """
        if not conversation:
            return {
                "messages": [],
                "summary": None,
                "truncated": False,
                "warnings": []
            }
        
        to_window = self.get_context_window(to_model)
        max_tokens = max_tokens or int(to_window * 0.8)
        
        # Simple token estimation (4 chars per token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        # Calculate conversation size
        total_tokens = sum(
            estimate_tokens(m.get("content", ""))
            for m in conversation
        )
        
        result = {
            "messages": [],
            "summary": None,
            "truncated": False,
            "warnings": []
        }
        
        # If fits, return as-is
        if total_tokens <= max_tokens:
            result["messages"] = conversation
            return result
        
        # Need to truncate - keep most recent messages
        result["truncated"] = True
        result["warnings"].append(
            f"Conversation truncated to fit {to_model}'s context window"
        )
        
        # Calculate how many messages to keep (from end)
        kept_tokens = 0
        kept_messages = []
        
        for msg in reversed(conversation):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if kept_tokens + msg_tokens > max_tokens * 0.9:  # Leave room for summary
                break
            kept_tokens += msg_tokens
            kept_messages.insert(0, msg)
        
        # Generate summary of truncated messages
        if include_summary and len(conversation) > len(kept_messages):
            truncated_count = len(conversation) - len(kept_messages)
            result["summary"] = (
                f"[Konteks sebelumnya: {truncated_count} pesan telah diringkas. "
                f"Topik yang dibahas meliputi aspek hukum yang ditanyakan pengguna.]"
            )
        
        result["messages"] = kept_messages
        
        logger.info(f"Context prepared for {to_model}", {
            "original_messages": len(conversation),
            "kept_messages": len(kept_messages),
            "truncated": result["truncated"]
        })
        
        return result
    
    def create_switch_summary(
        self,
        conversation: List[Dict[str, str]],
        max_length: int = 500
    ) -> str:
        """
        Create a brief summary of conversation for context transfer.
        
        Args:
            conversation: Conversation history
            max_length: Maximum summary length in characters
            
        Returns:
            Summary string
        """
        if not conversation:
            return ""
        
        # Extract key points from last few exchanges
        recent = conversation[-6:]  # Last 3 exchanges
        
        summary_parts = []
        
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]  # First 200 chars
            
            if role == "user":
                summary_parts.append(f"Q: {content}")
            elif role == "assistant":
                # Skip thinking tags
                if "<think>" in content:
                    # Extract answer after thinking
                    if "</think>" in content:
                        content = content.split("</think>")[-1].strip()[:200]
                summary_parts.append(f"A: {content}")
        
        summary = "\n".join(summary_parts)
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary


# Global instance
_context_transfer: Optional[ContextTransfer] = None


def get_context_transfer() -> ContextTransfer:
    """Get global context transfer instance"""
    global _context_transfer
    if _context_transfer is None:
        _context_transfer = ContextTransfer()
    return _context_transfer
