"""
Core Agent Implementation

Main Personal AI Assistant agent built with the Strands framework,
integrating memory systems and MCP servers for enhanced capabilities.
"""

from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

from strands import Agent, tool
from strands.models import BedrockModel
from strands.models.anthropic import AnthropicModel
from strands.tools.mcp import MCPClient
from strands_tools import calculator, current_time
from mcp.client.stdio import stdio_client, StdioServerParameters

from ..memory.memory_manager import MemoryManager, MemoryEntry
from ..memory.file_operations import MemoryFileOperations
from .agent_config import AgentConfig, MCPServerConfig


logger = logging.getLogger(__name__)


class SessionContext:
    """Manages session-specific context and memory state."""
    
    def __init__(self):
        self.core_memories: Dict[str, Tuple[Dict[str, Any], str]] = {}
        self.recent_interactions: List[Dict[str, Any]] = []
        self.active_context: str = ""
        self.user_profile: str = ""
        self.conversation_summary: str = ""
        self.memory_stats: Dict[str, Any] = {}
        self.session_initialized_at: datetime = datetime.now()
        self.context_prepared: bool = False
        
    def is_ready(self) -> bool:
        """Check if session context is ready for conversations."""
        return (
            bool(self.core_memories) and 
            self.context_prepared and 
            bool(self.user_profile)
        )


class PersonalAssistantAgent:
    """
    Personal AI Assistant with persistent memory and native Strands MCP integration.
    
    This agent provides:
    - Persistent memory across conversations
    - File system access through native Strands MCP servers
    - Memory search and retrieval capabilities
    - Personalized responses based on interaction history
    - Integration with external tools via native MCP
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
        self.mcp_clients: List[MCPClient] = []
        self.mcp_server_names: List[str] = []
        self.mcp_tools: List[Any] = []
        self.agent: Optional[Agent] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_context: SessionContext = SessionContext()
        self.is_initialized = False
        self.initialization_steps: Dict[str, bool] = {
            'memory_system': False,
            'memory_loaded': False,
            'context_prepared': False,
            'mcp_clients': False,
            'strands_agent': False,
            'validation_complete': False
        }
        
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
        Initialize the agent and all its components with comprehensive session setup.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info(f"Starting comprehensive initialization of {self.config.agent_name} v{self.config.agent_version}")
            
            # Step 1: Initialize memory system
            if self.config.memory.enabled:
                success = await self._initialize_memory_system()
                self.initialization_steps['memory_system'] = success
                if not success:
                    logger.error("Memory system initialization failed")
                    return False
            
            # Step 2: Load all memory files and prepare context
            if self.config.memory.enabled:
                success = await self._load_memory_and_prepare_context()
                self.initialization_steps['memory_loaded'] = success
                self.initialization_steps['context_prepared'] = success
                if not success:
                    logger.error("Memory loading and context preparation failed")
                    return False
            
            # Step 3: Initialize native Strands MCP clients
            if self.config.mcp.enabled:
                success = await self._initialize_mcp_clients()
                self.initialization_steps['mcp_clients'] = success
                if not success:
                    logger.warning("MCP clients initialization failed - continuing without MCP tools")
            
            # Step 4: Initialize the Strands agent with native MCP integration
            success = await self._initialize_strands_agent()
            self.initialization_steps['strands_agent'] = success
            if not success:
                logger.error("Strands agent initialization failed")
                return False
            
            # Step 5: Validate complete system
            success = await self._validate_system_integrity()
            self.initialization_steps['validation_complete'] = success
            if not success:
                logger.error("System validation failed")
                return False
            
            self.is_initialized = True
            logger.info("Personal AI Assistant initialization complete - ready for conversations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            return False
    
    async def _initialize_memory_system(self) -> bool:
        """Initialize the memory management system."""
        try:
            logger.info("Initializing memory system...")
            
            # Ensure memory directory exists
            memory_path = Path(self.config.memory.memory_base_path)
            memory_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize file operations (static class, no instantiation needed)
            self.file_ops = MemoryFileOperations
            
            # Initialize memory manager
            self.memory_manager = MemoryManager(base_path=str(memory_path))
            
            logger.info("Memory system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Memory system initialization failed: {e}")
            return False
    
    async def _load_memory_and_prepare_context(self) -> bool:
        """Load all memory files and prepare conversation context."""
        try:
            logger.info("Loading memory files and preparing session context...")
            
            if not self.memory_manager:
                raise RuntimeError("Memory manager not initialized")
            
            # Load all core memories
            await self._load_core_memories()
            
            # Load recent interaction history
            await self._load_recent_interactions()
            
            # Prepare conversation context
            await self._prepare_conversation_context()
            
            # Get memory statistics
            await self._load_memory_statistics()
            
            # Mark context as prepared
            self.session_context.context_prepared = True
            
            logger.info(f"Memory loading complete - loaded {len(self.session_context.core_memories)} core memories")
            logger.info(f"Loaded {len(self.session_context.recent_interactions)} recent interactions")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory loading failed: {e}")
            return False
    
    async def _load_core_memories(self) -> None:
        """Load all core memory files."""
        logger.info("Loading core memory files...")
        
        # Load all core memories at once
        self.session_context.core_memories = self.memory_manager.get_all_core_memories()
        
        # Extract key information for quick access
        if 'user_profile' in self.session_context.core_memories:
            _, content = self.session_context.core_memories['user_profile']
            self.session_context.user_profile = content
        
        if 'active_context' in self.session_context.core_memories:
            _, content = self.session_context.core_memories['active_context']
            self.session_context.active_context = content
        
        logger.info(f"Loaded {len(self.session_context.core_memories)} core memory files")
    
    async def _load_recent_interactions(self) -> None:
        """Load recent interaction history for context."""
        logger.info("Loading recent interaction history...")
        
        # Get recent interactions (last 7 days, up to 20 interactions)
        self.session_context.recent_interactions = self.memory_manager.get_recent_interactions(
            days=7, limit=20
        )
        
        logger.info(f"Loaded {len(self.session_context.recent_interactions)} recent interactions")
    
    async def _prepare_conversation_context(self) -> None:
        """Prepare conversation context summary from loaded memories."""
        logger.info("Preparing conversation context...")
        
        context_parts = []
        
        # Add user profile summary
        if self.session_context.user_profile:
            context_parts.append(f"User Profile:\n{self.session_context.user_profile[:500]}...")
        
        # Add active context
        if self.session_context.active_context:
            context_parts.append(f"Active Context:\n{self.session_context.active_context[:300]}...")
        
        # Add recent interaction summary
        if self.session_context.recent_interactions:
            recent_summary = self._create_interaction_summary(self.session_context.recent_interactions)
            context_parts.append(f"Recent Interactions Summary:\n{recent_summary}")
        
        # Combine all context
        self.session_context.conversation_summary = "\n\n".join(context_parts)
        
        logger.info(f"Conversation context prepared ({len(self.session_context.conversation_summary)} characters)")
    
    def _create_interaction_summary(self, interactions: List[Dict[str, Any]]) -> str:
        """Create a summary of recent interactions."""
        if not interactions:
            return "No recent interactions found."
        
        summary_parts = []
        for interaction in interactions[-5:]:  # Last 5 interactions
            created = interaction.get('created', 'Unknown time')
            content_preview = interaction.get('content', '')[:100]
            summary_parts.append(f"[{created}] {content_preview}...")
        
        return "\n".join(summary_parts)
    
    async def _load_memory_statistics(self) -> None:
        """Load memory system statistics."""
        logger.info("Loading memory statistics...")
        
        self.session_context.memory_stats = self.memory_manager.get_memory_statistics()
        
        logger.info(f"Memory stats: {self.session_context.memory_stats.get('total_memories', 0)} total memories")
    
    async def _initialize_mcp_clients(self) -> bool:
        """Initialize MCP clients for enabled servers with enhanced error isolation."""
        if not self.config.mcp.enabled:
            logger.info("MCP integration disabled")
            return True
        
        try:
            logger.info("Initializing MCP clients with enhanced multi-server support...")
            
            enabled_servers = self.config.mcp.get_enabled_servers()
            if not enabled_servers:
                logger.warning("No enabled MCP servers found in configuration")
                return True
            
            # Track successful and failed server connections
            successful_clients = []
            failed_servers = []
            
            for server in enabled_servers:
                try:
                    client = await self._create_mcp_client(server)
                    if client:
                        successful_clients.append((server.name, client))
                        logger.info(f"✅ MCP client created for {server.name} ({server.transport})")
                    else:
                        failed_servers.append(server.name)
                        logger.warning(f"❌ Failed to create MCP client for {server.name}")
                except Exception as e:
                    failed_servers.append(server.name)
                    logger.error(f"❌ Error creating MCP client for {server.name}: {e}")
                    # Continue with other servers rather than failing completely
                    continue
            
            # Store successful clients with server names for tool namespacing
            self.mcp_clients = [client for _, client in successful_clients]
            self.mcp_server_names = [name for name, _ in successful_clients]
            
            # Report results
            if successful_clients:
                logger.info(f"✅ Successfully initialized {len(successful_clients)} MCP clients: {[name for name, _ in successful_clients]}")
            else:
                logger.warning("No MCP clients were successfully created")
            
            if failed_servers:
                logger.warning(f"❌ Failed to initialize MCP clients for: {failed_servers}")
                
            # Return True even if some servers failed (error isolation)
            return len(successful_clients) > 0 or len(enabled_servers) == 0
            
        except Exception as e:
            logger.error(f"MCP clients initialization failed: {e}")
            # Initialize empty lists to prevent errors
            self.mcp_clients = []
            self.mcp_server_names = []
            return False
    
    async def _create_mcp_client(self, server: MCPServerConfig) -> Optional[MCPClient]:
        """Create a single MCP client for the specified server configuration with support for all transport types."""
        try:
            logger.debug(f"Creating MCP client for {server.name} using {server.transport} transport")
            
            if server.transport == "stdio":
                # Create stdio client
                server_params = StdioServerParameters(
                    command=server.command,
                    args=server.args,
                    env=server.env
                )
                
                client = MCPClient(
                    lambda: stdio_client(server_params)
                )
                
                logger.debug(f"Created stdio MCP client for {server.name}")
                return client
                
            elif server.transport == "http":
                # HTTP transport implementation
                try:
                    from mcp.client.sse import sse_client
                    from mcp.client.session import ClientSession
                    import httpx
                    
                    # Create HTTP client session
                    async def create_http_session():
                        async with httpx.AsyncClient() as http_client:
                            # Create SSE connection for HTTP transport
                            read, write = await sse_client(server.url)
                            return ClientSession(read, write)
                    
                    client = MCPClient(create_http_session)
                    logger.debug(f"Created HTTP MCP client for {server.name} at {server.url}")
                    return client
                    
                except ImportError as e:
                    logger.error(f"HTTP transport dependencies not available for {server.name}: {e}")
                    logger.info("Install with: pip install mcp[sse] httpx")
                    return None
                except Exception as e:
                    logger.error(f"Error creating HTTP client for {server.name}: {e}")
                    return None
                
            elif server.transport == "sse":
                # SSE transport implementation
                try:
                    from mcp.client.sse import sse_client
                    
                    client = MCPClient(
                        lambda: sse_client(server.url)
                    )
                    
                    logger.debug(f"Created SSE MCP client for {server.name} at {server.url}")
                    return client
                    
                except ImportError as e:
                    logger.error(f"SSE transport dependencies not available for {server.name}: {e}")
                    logger.info("Install with: pip install mcp[sse]")
                    return None
                except Exception as e:
                    logger.error(f"Error creating SSE client for {server.name}: {e}")
                    return None
                
            else:
                logger.error(f"Unsupported transport type: {server.transport} for {server.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating MCP client for {server.name}: {e}")
            return None
    
    async def _initialize_strands_agent(self) -> bool:
        """Initialize the Strands agent with model and tools using native MCP integration."""
        try:
            logger.info("Initializing Strands agent with native MCP integration...")
            
            # Configure model
            model = self._create_model()
            
            # Create enhanced system prompt with context
            enhanced_system_prompt = self._create_enhanced_system_prompt()
            
            # Collect non-MCP tools first
            tools = []
            
            # Add built-in Strands tools
            if self.config.enable_builtin_tools:
                tools.extend([calculator, current_time])
            
            # Add memory-related custom tools (fallback when MCP isn't available)
            if self.memory_manager:
                tools.extend(self._create_memory_tools())
            
            # For Strands native MCP integration, we need to pass the MCP clients directly to Agent
            # and use them within context managers during conversation
            if self.config.mcp.enabled and self.mcp_clients:
                # Create the Strands agent with MCP clients for native integration
                self.agent = Agent(
                    model=model,
                    tools=tools,
                    system_prompt=enhanced_system_prompt
                )
                logger.info(f"Strands agent initialized with {len(tools)} tools + {len(self.mcp_clients)} MCP clients")
            else:
                # Create agent without MCP
                self.agent = Agent(
                    model=model,
                    tools=tools,
                    system_prompt=enhanced_system_prompt
                )
                logger.info(f"Strands agent initialized with {len(tools)} tools (no MCP)")
            
            return True
            
        except Exception as e:
            logger.error(f"Strands agent initialization failed: {e}")
            return False
    
    async def _collect_tools_without_mcp(self) -> List[Any]:
        """Collect tools excluding MCP tools."""
        tools = []
        
        # Add built-in Strands tools
        if self.config.enable_builtin_tools:
            tools.extend([calculator, current_time])
        
        # Add memory-related custom tools
        if self.memory_manager:
            tools.extend(self._create_memory_tools())
        
        return tools
    
    def _create_enhanced_system_prompt(self) -> str:
        """Create system prompt enhanced with session context."""
        base_prompt = self.config.system_prompt
        
        if not self.session_context.is_ready():
            return base_prompt
        
        # Add context information to system prompt
        context_addition = f"""

CURRENT SESSION CONTEXT:
{self.session_context.conversation_summary}

Remember to use this context to provide personalized and relevant responses.
You have access to the user's profile, recent conversations, and preferences.
"""
        
        return base_prompt + context_addition
    
    async def _validate_system_integrity(self) -> bool:
        """Validate that all systems are working correctly."""
        try:
            logger.info("Validating system integrity...")
            
            # Validate memory system
            if self.config.memory.enabled:
                if not self.memory_manager:
                    logger.error("Memory manager not initialized")
                    return False
                
                # Run memory system validation
                validation_results = self.memory_manager.validate_memory_system()
                if not validation_results.get('overall_health', False):
                    logger.warning(f"Memory system validation issues: {validation_results}")
                    # Don't fail initialization for minor issues
            
            # Validate MCP clients
            if self.config.mcp.enabled and self.mcp_clients:
                logger.info(f"Validating {len(self.mcp_clients)} MCP clients")
                # Basic validation - just check that clients were created
                for i, mcp_client in enumerate(self.mcp_clients):
                    if mcp_client is None:
                        logger.warning(f"MCP client {i} is None")
                    else:
                        logger.debug(f"MCP client {i} is initialized")
            
            # Validate Strands agent
            if not self.agent:
                logger.error("Strands agent not initialized")
                return False
            
            # Validate session context
            if not self.session_context.is_ready():
                logger.error("Session context not properly prepared")
                return False
            
            logger.info("System validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return False
    
    def _create_model(self):
        """Create and configure the AI model based on provider."""
        if self.config.model.provider == "anthropic":
            return AnthropicModel(
                model_id=self.config.model.model_id,
                max_tokens=self.config.model.max_tokens,
                params={
                    "temperature": self.config.model.temperature,
                }
            )
        elif self.config.model.provider == "bedrock":
            return BedrockModel(
                model_id=self.config.model.model_id,
                region_name=self.config.model.region_name,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                streaming=self.config.model.streaming
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.config.model.provider}")
    

    
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
            
            # Process with Strands agent using MCP context managers
            # This is the proper way according to Strands documentation
            if self.config.mcp.enabled and self.mcp_clients:
                # Use MCP context managers for the conversation
                response = await self._process_with_mcp_context(enhanced_message)
            else:
                # Process without MCP
                response = self.agent(enhanced_message)
            
            # Extract text from response - handle various formats
            response_text = self._extract_response_text(response)
            
            # Store interaction in memory
            if self.memory_manager and self.config.enable_conversation_logging:
                await self._store_interaction(message, response_text, user_id)
            
            # Update conversation history
            self._update_conversation_history(message, response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    async def _process_with_mcp_context(self, message: str) -> str:
        """Process message using MCP context managers as required by Strands."""
        # We need to gather MCP tools within context managers
        all_mcp_tools = []
        
        # Collect tools from all MCP clients using context managers
        for i, client in enumerate(self.mcp_clients):
            server_name = self.mcp_server_names[i] if i < len(self.mcp_server_names) else f"server_{i}"
            
            try:
                # This is the proper Strands pattern - using async context manager
                async with client:
                    tools = await client.list_tools()
                    
                    # Add namespace prefix to tool names to avoid conflicts
                    for tool in tools:
                        if hasattr(tool, 'name'):
                            # Add server prefix to tool name for uniqueness
                            original_name = tool.name
                            tool.name = f"{server_name}_{original_name}"
                            
                            # Update description to include server source
                            if hasattr(tool, 'description'):
                                tool.description = f"[{server_name}] {tool.description}"
                            
                            logger.debug(f"Namespaced tool: {original_name} -> {tool.name}")
                    
                    all_mcp_tools.extend(tools)
                    logger.info(f"✅ Discovered {len(tools)} tools from {server_name} server")
                
            except Exception as e:
                logger.error(f"❌ Error discovering tools from {server_name} server: {e}")
                # Continue with other servers instead of failing completely
                continue
        
        # Create a new agent with all tools for this conversation
        if all_mcp_tools:
            current_tools = []
            
            # Add built-in tools
            if self.config.enable_builtin_tools:
                current_tools.extend([calculator, current_time])
            
            # Add memory tools  
            if self.memory_manager:
                current_tools.extend(self._create_memory_tools())
            
            # Add MCP tools
            current_tools.extend(all_mcp_tools)
            
            # Create temporary agent with all tools
            enhanced_agent = Agent(
                model=self._create_model(),
                tools=current_tools,
                system_prompt=self._create_enhanced_system_prompt()
            )
            
            logger.info(f"Created enhanced agent with {len(current_tools)} tools ({len(all_mcp_tools)} from MCP)")
            
            # Process with enhanced agent
            return enhanced_agent(message)
        else:
            # Fall back to agent without MCP tools
            logger.warning("No MCP tools available, using base agent")
            return self.agent(message)
    
    def _extract_response_text(self, response) -> str:
        """Extract text from various response formats."""
        # Handle AgentResult objects from Strands first
        if hasattr(response, '__class__') and 'AgentResult' in str(type(response)):
            # AgentResult object - convert to string
            return str(response)
        elif isinstance(response, list):
            # Handle list response format (common with Strands)
            if len(response) > 0:
                first_item = response[0]
                if isinstance(first_item, dict) and 'text' in first_item:
                    return first_item['text']
                else:
                    return str(first_item)
            else:
                return "No response generated"
        elif isinstance(response, dict):
            # Handle dictionary response format (Strands format)
            if 'content' in response and isinstance(response['content'], list):
                if len(response['content']) > 0:
                    content_item = response['content'][0]
                    if isinstance(content_item, dict) and 'text' in content_item:
                        return content_item['text']
                    else:
                        return str(content_item)
                else:
                    return str(response)
            elif 'text' in response:
                return response['text']
            elif 'message' in response:
                return response['message']
            else:
                return str(response)
        elif hasattr(response, 'message'):
            return response.message
        elif hasattr(response, 'content'):
            if isinstance(response.content, list) and len(response.content) > 0:
                # Handle list of content items
                content_item = response.content[0]
                if isinstance(content_item, dict) and 'text' in content_item:
                    return content_item['text']
                else:
                    return str(content_item)
            else:
                return str(response.content)
        elif hasattr(response, 'text'):
            return response.text
        else:
            return str(response)
    
    async def _enhance_message_with_context(self, message: str, user_id: str) -> str:
        """Enhance the user message with relevant context from memory."""
        if not self.config.memory.enabled or not self.memory_manager:
            return message
        
        try:
            # Use direct memory manager search (native MCP tools will be available through the agent)
            relevant_memories = self.memory_manager.search_memories(
                message, 
                limit=self.config.memory.max_context_memories
            )
            
            if relevant_memories:
                # Filter by importance threshold
                important_memories = [
                    mem for mem in relevant_memories 
                    if mem.relevance_score >= self.config.memory.importance_threshold
                ]
                
                if important_memories:
                    # Build context
                    context_parts = ["Previous relevant context:"]
                    for memory in important_memories:
                        context_parts.append(f"- {memory.content_snippet}")
                    
                    context_parts.append(f"\nCurrent message: {message}")
                    
                    return "\n".join(context_parts)
            
            return message
            
        except Exception as e:
            logger.warning(f"Failed to enhance message with context: {e}")
            return message
    
    async def _store_interaction(self, user_input: str, assistant_response: str, user_id: str) -> None:
        """Store the interaction in memory."""
        try:
            # Use direct memory manager (native MCP tools are available through the agent)
            if self.memory_manager:
                entry = MemoryEntry(
                    content=f"User: {user_input}\n\nAssistant: {assistant_response}",
                    importance_score=5,  # Default importance
                    category="conversation",
                    metadata={"user_id": user_id}
                )
                self.memory_manager.create_interaction_memory(entry)
                logger.debug("Interaction stored in memory")
                
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
        """Get comprehensive agent status including detailed MCP server information."""
        try:
            status = {
                "agent_status": "operational" if self.is_initialized else "not_initialized",
                "initialization_status": {
                    "memory_system": self.memory_manager is not None,
                    "mcp_clients": len(self.mcp_clients) > 0 if hasattr(self, 'mcp_clients') else False,
                    "strands_agent": self.agent is not None,
                    "session_context": self.session_context.is_ready()
                },
                "configuration": {
                    "memory_enabled": self.config.memory.enabled,
                    "mcp_enabled": self.config.mcp.enabled,
                    "builtin_tools_enabled": self.config.enable_builtin_tools,
                    "conversation_logging": self.config.enable_conversation_logging
                },
                "memory_system": {},
                "mcp_system": {},
                "conversation": {
                    "history_length": len(self.conversation_history),
                    "max_history": self.config.max_conversation_history
                }
            }
            
            # Enhanced memory system status
            if self.memory_manager:
                try:
                    memory_stats = self.memory_manager.get_memory_statistics()
                    status["memory_system"] = {
                        "status": "operational",
                        "base_path": str(self.config.memory.memory_base_path),
                        "statistics": memory_stats
                    }
                except Exception as e:
                    status["memory_system"] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                status["memory_system"] = {"status": "disabled"}
            
            # Enhanced MCP system status with multiple server support
            if hasattr(self, 'mcp_clients') and self.mcp_clients:
                mcp_status = {
                    "status": "operational",
                    "total_clients": len(self.mcp_clients),
                    "configuration_status": self.config.mcp.get_server_health_status(),
                    "active_servers": [],
                    "server_tools": {}
                }
                
                # Get server names (with fallback)
                server_names = getattr(self, 'mcp_server_names', [f"server_{i}" for i in range(len(self.mcp_clients))])
                
                # Test each MCP client connection
                for i, client in enumerate(self.mcp_clients):
                    server_name = server_names[i] if i < len(server_names) else f"server_{i}"
                    server_status = {
                        "name": server_name,
                        "status": "unknown",
                        "tools_count": 0,
                        "error": None
                    }
                    
                    try:
                        # Test connection by attempting to list tools
                        tools = client.list_tools_sync()
                        server_status["status"] = "operational"
                        server_status["tools_count"] = len(tools)
                        
                        # Store tool names for this server
                        mcp_status["server_tools"][server_name] = [
                            tool.name if hasattr(tool, 'name') else str(tool) 
                            for tool in tools
                        ]
                        
                    except Exception as e:
                        server_status["status"] = "error"
                        server_status["error"] = str(e)
                    
                    mcp_status["active_servers"].append(server_status)
                
                # Calculate overall MCP health
                operational_servers = [s for s in mcp_status["active_servers"] if s["status"] == "operational"]
                total_tools = sum(s["tools_count"] for s in operational_servers)
                
                mcp_status["health_summary"] = {
                    "operational_servers": len(operational_servers),
                    "total_servers": len(mcp_status["active_servers"]),
                    "total_tools": total_tools,
                    "health_percentage": (len(operational_servers) / len(mcp_status["active_servers"]) * 100) if mcp_status["active_servers"] else 0
                }
                
                status["mcp_system"] = mcp_status
                
            elif self.config.mcp.enabled:
                status["mcp_system"] = {
                    "status": "configured_but_not_initialized",
                    "configuration_status": self.config.mcp.get_server_health_status()
                }
            else:
                status["mcp_system"] = {"status": "disabled"}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return {
                "agent_status": "error",
                "error": str(e),
                "initialization_status": {
                    "memory_system": False,
                    "mcp_clients": False,
                    "strands_agent": False,
                    "session_context": False
                }
            }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent and all its components."""
        logger.info("Shutting down Personal AI Assistant...")
        
        try:
            # Clear MCP clients (native Strands clients handle cleanup automatically)
            self.mcp_clients.clear()
            
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