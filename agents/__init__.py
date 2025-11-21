"""
Agents Module - Agentic Workflows

Provides tool-based agent execution for complex legal tasks.
"""

from .tool_registry import ToolRegistry
from .agent_executor import AgentExecutor

__all__ = ['ToolRegistry', 'AgentExecutor']
