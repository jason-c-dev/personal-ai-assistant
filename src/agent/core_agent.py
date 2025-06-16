"""
Core Agent Implementation

Main Personal AI Assistant agent built with the Strands framework,
integrating memory systems and MCP servers for enhanced capabilities.
"""

from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator, current_time

from memory.memory_manager import MemoryManager, MemoryEntry
from memory.file_operations import MemoryFileOperations
from agent.agent_config import AgentConfig
from agent.mcp_integration import MCPIntegration


logger = logging.getLogger(__name__)


class PersonalAssistantAgent:
    """
    Personal AI Assistant with persistent memory and MCP integration.
    
    This agent provides:
    - Persistent memory across conversations
    - File system access through MCP servers
    - Memory search and retrieval capabilities
    - Personalized responses based on interaction history
    - Integration with external tools via MCP
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Personal AI Assistant agent.
        
        Args:
            config: Agent configuration. If None, uses default configuration.
        """
        self.config = config or AgentConfig()
        self.memory_manager: Optional[MemoryManager] = None
        self.file_ops: Optional[MemoryFileOperations] = None
        self.mcp_integration: Optional[MCPIntegration] = None
        self.agent: Optional[Agent] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_initialized = False
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for the agent."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    async def initialize(self) -> bool:
        """
        Initialize the agent and all its components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info(f"Initializing {self.config.agent_name} v{self.config.agent_version}")
            
            # Initialize memory system
            if self.config.memory.enabled:
                await self._initialize_memory_system()
            
            # Initialize MCP integration
            if self.config.mcp.enabled:
                await self._initialize_mcp_integration()
            
            # Initialize the Strands agent
            await self._initialize_strands_agent()
            
            self.is_initialized = True
            logger.info("Personal AI Assistant initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            return False
    
    async def _initialize_memory_system(self) -> None:
        """Initialize the memory management system."""
        logger.info("Initializing memory system...")
        
        # Ensure memory directory exists
        memory_path = Path(self.config.memory.memory_base_path)
        memory_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize file operations (static class, no instantiation needed)
        self.file_ops = MemoryFileOperations
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(base_path=str(memory_path))
        
        logger.info("Memory system initialized successfully")
    
    async def _initialize_mcp_integration(self) -> None:
        """Initialize MCP server connections."""
        logger.info("Initializing MCP integration...")
        
        self.mcp_integration = MCPIntegration(self.config.mcp)
        success = await self.mcp_integration.initialize()
        
        if not success:
            logger.warning("MCP integration failed - continuing without MCP tools")
        else:
            logger.info("MCP integration initialized successfully")
    
    async def _initialize_strands_agent(self) -> None:
        """Initialize the Strands agent with model and tools."""
        logger.info("Initializing Strands agent...")
        
        # Configure model
        model = self._create_model()
        
        # Collect all tools
        tools = await self._collect_tools()
        
        # Create the Strands agent
        self.agent = Agent(
            model=model,
            tools=tools,
            system_prompt=self.config.system_prompt
        )
        
        logger.info(f"Strands agent initialized with {len(tools)} tools")
    
    def _create_model(self) -> BedrockModel:
        """Create and configure the AI model."""
        return BedrockModel(
            model_id=self.config.model.model_id,
            region_name=self.config.model.region_name,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
            streaming=self.config.model.streaming
        )
    
    async def _collect_tools(self) -> List[Any]:
        """Collect all available tools for the agent."""
        tools = []
        
        # Add built-in Strands tools
        if self.config.enable_builtin_tools:
            tools.extend([calculator, current_time])
        
        # Add memory-related custom tools
        if self.memory_manager:
            tools.extend(self._create_memory_tools())
        
        # Add MCP tools
        if self.mcp_integration:
            mcp_tools = self.mcp_integration.get_available_tools()
            tools.extend(mcp_tools)
            logger.info(f"Added {len(mcp_tools)} MCP tools")
        
        return tools
    
    def _create_memory_tools(self) -> List[Any]:
        """Create custom tools for memory operations."""
        memory_tools = []
        
        @tool
        def search_memories(query: str, limit: int = 5) -> str:
            """
            Search through stored memories for relevant information.
            
            Args:
                query: Search query to find relevant memories
                limit: Maximum number of memories to return
                
            Returns:
                String containing relevant memories found
            """
            try:
                if not self.memory_manager:
                    return "Memory system not available"
                
                results = self.memory_manager.search_memories(query, limit=limit)
                
                if not results:
                    return f"No memories found for query: {query}"
                
                formatted_results = []
                for result in results:
                    formatted_results.append(
                        f"Memory (Score: {result.relevance_score:.2f}): {result.content_snippet}"
                    )
                
                return "\n".join(formatted_results)
                
            except Exception as e:
                logger.error(f"Error searching memories: {e}")
                return f"Error searching memories: {str(e)}"
        
        @tool
        def store_memory(content: str, importance: float = 0.5) -> str:
            """
            Store important information in memory for future reference.
            
            Args:
                content: Information to store in memory
                importance: Importance score (0.0 to 1.0)
                
            Returns:
                Confirmation message
            """
            try:
                if not self.memory_manager:
                    return "Memory system not available"
                
                entry = MemoryEntry(
                    content=content,
                    importance_score=int(importance * 10),  # Convert 0-1 to 0-10 scale
                    category="manual_storage"
                )
                memory_id = self.memory_manager.create_interaction_memory(entry)
                
                return f"Memory stored successfully with ID: {memory_id}"
                
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
                return f"Error storing memory: {str(e)}"
        
        @tool
        def get_user_profile() -> str:
            """
            Retrieve the current user profile information.
            
            Returns:
                User profile information as a string
            """
            try:
                if not self.memory_manager:
                    return "Memory system not available"
                
                frontmatter, profile = self.memory_manager.get_core_memory("user_profile")
                
                if not profile:
                    return "No user profile found"
                
                return f"User Profile:\n{profile}"
                
            except Exception as e:
                logger.error(f"Error retrieving user profile: {e}")
                return f"Error retrieving user profile: {str(e)}"
        
        memory_tools.extend([search_memories, store_memory, get_user_profile])
        return memory_tools
    
    async def process_message(self, message: str, user_id: str = "default") -> str:
        """
        Process a user message and generate a response.
        
        Args:
            message: User's input message
            user_id: Identifier for the user (for personalization)
            
        Returns:
            Assistant's response
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Add conversation context from memory
            enhanced_message = await self._enhance_message_with_context(message, user_id)
            
            # Process with Strands agent
            response = self.agent(enhanced_message)
            
            # Extract text from response
            if isinstance(response, dict):
                # Handle dictionary response format
                if 'content' in response and isinstance(response['content'], list):
                    if len(response['content']) > 0 and 'text' in response['content'][0]:
                        response_text = response['content'][0]['text']
                    else:
                        response_text = str(response)
                elif 'message' in response:
                    response_text = response['message']
                else:
                    response_text = str(response)
            elif hasattr(response, 'message'):
                response_text = response.message
            elif hasattr(response, 'content'):
                if isinstance(response.content, list) and len(response.content) > 0:
                    # Handle list of content items
                    content_item = response.content[0]
                    if isinstance(content_item, dict) and 'text' in content_item:
                        response_text = content_item['text']
                    else:
                        response_text = str(content_item)
                else:
                    response_text = str(response.content)
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            # Store interaction in memory
            if self.memory_manager and self.config.enable_conversation_logging:
                await self._store_interaction(message, response_text, user_id)
            
            # Update conversation history
            self._update_conversation_history(message, response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    async def _enhance_message_with_context(self, message: str, user_id: str) -> str:
        """Enhance the user message with relevant context from memory."""
        if not self.memory_manager or not self.config.memory.enabled:
            return message
        
        try:
            # Search for relevant memories
            relevant_memories = self.memory_manager.search_memories(
                message, 
                limit=self.config.memory.max_context_memories
            )
            
            if not relevant_memories:
                return message
            
            # Filter by importance threshold
            important_memories = [
                mem for mem in relevant_memories 
                if mem.relevance_score >= self.config.memory.importance_threshold
            ]
            
            if not important_memories:
                return message
            
            # Build context
            context_parts = ["Previous relevant context:"]
            for memory in important_memories:
                context_parts.append(f"- {memory.content_snippet}")
            
            context_parts.append(f"\nCurrent message: {message}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Failed to enhance message with context: {e}")
            return message
    
    async def _store_interaction(self, user_input: str, assistant_response: str, user_id: str) -> None:
        """Store the interaction in memory."""
        try:
            if self.memory_manager:
                entry = MemoryEntry(
                    content=f"User: {user_input}\n\nAssistant: {assistant_response}",
                    importance_score=5,  # Default importance
                    category="conversation",
                    metadata={"user_id": user_id}
                )
                self.memory_manager.create_interaction_memory(entry)
        except Exception as e:
            logger.warning(f"Failed to store interaction in memory: {e}")
    
    def _update_conversation_history(self, user_message: str, assistant_response: str) -> None:
        """Update the in-memory conversation history."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response
        })
        
        # Trim history if it exceeds maximum length
        if len(self.conversation_history) > self.config.max_conversation_history:
            self.conversation_history = self.conversation_history[-self.config.max_conversation_history:]
    
    async def stream_response(self, message: str, user_id: str = "default") -> AsyncGenerator[str, None]:
        """
        Stream a response to a user message.
        
        Args:
            message: User's input message
            user_id: Identifier for the user
            
        Yields:
            Chunks of the assistant's response
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Add conversation context from memory
            enhanced_message = await self._enhance_message_with_context(message, user_id)
            
            # Stream response from Strands agent
            full_response = ""
            async for chunk in self.agent.stream_async(enhanced_message):
                if "data" in chunk:
                    text_chunk = chunk["data"]
                    full_response += text_chunk
                    yield text_chunk
                elif isinstance(chunk, str):
                    full_response += chunk
                    yield chunk
            
            # Store interaction in memory after streaming is complete
            if self.memory_manager and self.config.enable_conversation_logging:
                await self._store_interaction(message, full_response, user_id)
            
            # Update conversation history
            self._update_conversation_history(message, full_response)
            
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the agent.
        
        Returns:
            Dictionary containing agent status information
        """
        status = {
            "agent_name": self.config.agent_name,
            "agent_version": self.config.agent_version,
            "initialized": self.is_initialized,
            "memory_enabled": self.config.memory.enabled,
            "mcp_enabled": self.config.mcp.enabled,
            "conversation_turns": len(self.conversation_history)
        }
        
        # Add memory system status
        if self.memory_manager:
            try:
                memory_stats = self.memory_manager.get_memory_statistics()
                status["memory_stats"] = memory_stats
            except Exception as e:
                status["memory_error"] = str(e)
        
        # Add MCP status
        if self.mcp_integration:
            try:
                mcp_health = await self.mcp_integration.health_check()
                status["mcp_status"] = mcp_health
            except Exception as e:
                status["mcp_error"] = str(e)
        
        return status
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent and all its components."""
        logger.info("Shutting down Personal AI Assistant...")
        
        try:
            # Shutdown MCP integration
            if self.mcp_integration:
                await self.mcp_integration.shutdown()
            
            # Clear conversation history
            self.conversation_history.clear()
            
            self.is_initialized = False
            logger.info("Personal AI Assistant shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history.copy() 