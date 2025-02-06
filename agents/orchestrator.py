from typing import Dict, List, Union
import logging
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from models.agent_state import AgentState
from tools.notion_tools import NotionSearchTool, NotionPageReaderTool, NotionKnowledgeGraphTool
from agents.base import create_notion_chat_agent
from agents.search import create_search_agent

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class NotionAgentOrchestrator:
    """
    Orchestrates different specialized agents for interacting with the Notion knowledge base.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        search_tool: NotionSearchTool,
        reader_tool: NotionPageReaderTool,
        graph_tool: NotionKnowledgeGraphTool
    ):
        """Initialize the orchestrator with necessary tools and models."""
        logger.info("Initializing NotionAgentOrchestrator...")
        self.llm = llm
        self.search_tool = search_tool
        self.reader_tool = reader_tool
        self.graph_tool = graph_tool
        self.tools = [search_tool, reader_tool, graph_tool]
        
        try:
            # Initialize specialized agents
            logger.info("Creating chat agent...")
            self.chat_agent = create_notion_chat_agent(
                llm=llm,
                tools=self.tools
            )
            
            logger.info("Creating search agent...")
            self.search_agent = create_search_agent(
                llm=llm,
                search_tool=search_tool,
                reader_tool=reader_tool
            )
            
            # Initialize memory
            self.memory = MemorySaver()
            logger.info("NotionAgentOrchestrator initialization complete.")
            
        except Exception as e:
            logger.error(f"Error during orchestrator initialization: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
    def chat(self, message: str, thread_id: str = "default") -> str:
        """
        Process a user message and return a response.
        
        Args:
            message: The user's message
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            str: The agent's response
        """
        logger.info(f"Processing chat message in thread {thread_id}")
        logger.info(f"Message content: {message}")
        
        # Create config with thread ID
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initialize message state
        state = {
            "messages": [HumanMessage(content=message)],
            "current_query": message,
            "search_results": [],
            "notion_context": {},
            "status": "READY",
            "intermediate_steps": [],
            "next": None  # Add next field for routing
        }
        
        try:
            # First, try to handle with search agent for information retrieval
            logger.info("Invoking search agent...")
            search_result = self.search_agent.invoke(state, config)
            
            # Update state with search results
            if search_result:
                logger.debug(f"Search results received: {search_result}")
                state.update({
                    "search_results": search_result.get("search_results", []),
                    "notion_context": search_result.get("notion_context", {}),
                    "status": search_result.get("status", "READY")
                })
            
            # Then, process with main chat agent
            logger.info("Invoking chat agent...")
            chat_result = self.chat_agent.invoke(state, config)
            
            # Extract the final response
            if chat_result and "messages" in chat_result:
                final_messages = chat_result["messages"]
                if final_messages:
                    final_message = final_messages[-1]
                    if hasattr(final_message, 'content') and final_message.content:
                        logger.info("Successfully generated response")
                        return final_message.content
            
            logger.warning("No valid response generated from chat agent")
            return "I apologize, but I couldn't process your request properly."
            
        except Exception as e:
            logger.error(f"Error during chat processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Current state: {state}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def reset_thread(self, thread_id: str) -> None:
        """
        Reset the conversation history for a specific thread.
        
        Args:
            thread_id: The ID of the thread to reset
        """
        logger.info(f"Resetting thread {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        try:
            self.memory.delete(config)
            logger.info(f"Successfully reset thread {thread_id}")
        except Exception as e:
            logger.error(f"Error resetting thread {thread_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise 