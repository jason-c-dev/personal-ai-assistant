"""
Personal AI Assistant Agent Module

This module contains the core agent implementation using the Strands framework,
integrating with memory systems and MCP servers for enhanced capabilities.
"""

from agent.core_agent import PersonalAssistantAgent
from agent.agent_config import AgentConfig
from agent.mcp_integration import MCPIntegration

__all__ = [
    'PersonalAssistantAgent',
    'AgentConfig', 
    'MCPIntegration'
] 