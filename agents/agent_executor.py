"""
Agent Executor - Agentic Workflow Execution

Executes multi-step agent workflows with tool usage.
"""

from typing import Dict, Any, List, Optional
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .tool_registry import ToolRegistry
from logger_utils import get_logger

logger = get_logger(__name__)


class AgentExecutor:
    """
    Executes agentic workflows with tool usage

    Coordinates between LLM reasoning and tool execution.
    """

    def __init__(self, pipeline, config: Optional[Dict[str, Any]] = None):
        self.pipeline = pipeline
        self.config = config or {}
        self.max_iterations = self.config.get('max_iterations', 5)

        # Initialize tool registry
        self.registry = ToolRegistry()
        self.registry.register_default_tools(pipeline)

    def execute(self, task: str, context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Execute an agentic task

        Args:
            task: Task description
            context: Optional conversation context

        Returns:
            Execution result with final answer
        """
        logger.info(f"Executing agent task: {task[:100]}...")

        # Build agent prompt
        tools_description = self._format_tools_description()

        agent_prompt = f"""You are a legal research agent with access to the following tools:

{tools_description}

To use a tool, respond with:
TOOL: tool_name
ARGS: {{"param": "value"}}

When you have enough information, respond with:
ANSWER: Your final answer here

Task: {task}

Think step by step and use tools as needed."""

        # Execute agent loop
        messages = context.copy() if context else []
        messages.append({"role": "user", "content": agent_prompt})

        iterations = 0
        tool_results = []

        while iterations < self.max_iterations:
            iterations += 1

            # Get LLM response
            result = self.pipeline.query(
                messages[-1]["content"] if messages else task,
                conversation_history=messages[:-1] if len(messages) > 1 else None,
                stream=False
            )

            response = result.get('answer', '')

            # Check for final answer
            if 'ANSWER:' in response:
                final_answer = response.split('ANSWER:')[-1].strip()
                return {
                    "answer": final_answer,
                    "iterations": iterations,
                    "tool_results": tool_results,
                    "success": True
                }

            # Check for tool usage
            if 'TOOL:' in response and 'ARGS:' in response:
                tool_result = self._execute_tool_from_response(response)
                tool_results.append(tool_result)

                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {json.dumps(tool_result, ensure_ascii=False)}"
                })
            else:
                # No tool call, treat as thinking
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Continue with your analysis. Use tools or provide ANSWER:"
                })

        # Max iterations reached
        return {
            "answer": "Unable to complete task within iteration limit",
            "iterations": iterations,
            "tool_results": tool_results,
            "success": False
        }

    def _format_tools_description(self) -> str:
        """Format tools description for prompt"""
        descriptions = []
        for schema in self.registry.get_all_schemas():
            params = schema.get('parameters', {})
            param_str = ", ".join(params.get('required', []))
            descriptions.append(f"- {schema['name']}: {schema['description']}")
            if param_str:
                descriptions.append(f"  Parameters: {param_str}")

        return "\n".join(descriptions)

    def _execute_tool_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and execute tool from LLM response"""
        try:
            # Extract tool name
            tool_line = [l for l in response.split('\n') if 'TOOL:' in l][0]
            tool_name = tool_line.split('TOOL:')[-1].strip()

            # Extract arguments
            args_line = [l for l in response.split('\n') if 'ARGS:' in l][0]
            args_str = args_line.split('ARGS:')[-1].strip()
            args = json.loads(args_str)

            # Execute tool
            result = self.registry.execute(tool_name, **args)
            return {
                "tool": tool_name,
                "args": args,
                "result": result
            }

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "tool": "unknown",
                "error": str(e)
            }

    def list_available_tools(self) -> List[str]:
        """List available tools"""
        return self.registry.list_tools()
