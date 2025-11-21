"""
Tool Registry - Agent Tool Management

Manages registration and execution of agent tools.
"""

from typing import Dict, Any, List, Callable, Optional
from abc import ABC, abstractmethod
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger_utils import get_logger

logger = get_logger(__name__)


class BaseTool(ABC):
    """Abstract base class for agent tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }

    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema"""
        pass


class ToolRegistry:
    """
    Registry for managing agent tools

    Provides tool registration, lookup, and execution.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool"""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools"""
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name"""
        tool = self.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}

    def register_default_tools(self, pipeline) -> None:
        """Register default tools with pipeline"""
        from .tools import SearchTool, CitationTool, SummaryTool

        self.register(SearchTool(pipeline))
        self.register(CitationTool(pipeline))
        self.register(SummaryTool(pipeline))

        logger.info(f"Registered {len(self._tools)} default tools")
