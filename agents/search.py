from typing import Dict, List, Optional
import logging
import traceback
import json

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, Graph, END
from langgraph.prebuilt import ToolNode

from models.agent_state import AgentState
from tools.notion_tools import NotionSearchTool, NotionPageReaderTool

# Set up logging
logger = logging.getLogger(__name__)


def create_search_agent(
    llm: ChatOpenAI,
    search_tool: NotionSearchTool,
    reader_tool: NotionPageReaderTool
) -> Graph:
    """
    Create a specialized agent for searching and retrieving information from Notion.
    """
    logger.info("Creating search agent...")
    
    # Create the graph
    logger.info("Creating state graph...")
    workflow = StateGraph(AgentState)
    
    # Define the search node
    def search_node(state: AgentState) -> Dict:
        """Execute search and update state with results."""
        logger.debug(f"Processing search node with query: {state.get('current_query')}")
        try:
            # Execute search
            search_results_json = search_tool.run(state["current_query"])
            search_results = json.loads(search_results_json) if search_results_json else []
            logger.debug(f"Search results: {search_results}")
            
            # Read pages if search returned results
            notion_context = {}
            if search_results:
                for result in search_results[:3]:  # Process top 3 results
                    page_id = result.get("page_id")
                    if page_id:
                        try:
                            page_content = reader_tool.run(page_id)
                            notion_context[page_id] = json.loads(page_content) if page_content else {}
                        except Exception as e:
                            logger.error(f"Error reading page {page_id}: {str(e)}")
            
            logger.debug(f"Notion context gathered: {notion_context}")
            return {
                "search_results": search_results,
                "notion_context": notion_context,
                "status": "COMPLETE"
            }
        except Exception as e:
            logger.error(f"Error in search node: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "search_results": [],
                "notion_context": {},
                "status": "ERROR",
                "error": str(e)
            }
    
    # Add nodes to graph
    logger.info("Adding nodes to graph...")
    workflow.add_node("search", search_node)
    
    # Set entry point and end condition
    workflow.set_entry_point("search")
    workflow.add_edge("search", END)
    
    logger.info("Successfully created search agent")
    return workflow.compile() 