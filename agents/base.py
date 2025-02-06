from typing import Annotated, Dict, List, Sequence, Union
import logging
import traceback

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, Graph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from models.agent_state import AgentState
from tools.notion_tools import NotionSearchTool, NotionPageReaderTool, NotionKnowledgeGraphTool

# Set up logging
logger = logging.getLogger(__name__)


def create_notion_chat_agent(
    llm: ChatOpenAI,
    tools: List[Union[NotionSearchTool, NotionPageReaderTool, NotionKnowledgeGraphTool]]
) -> Graph:
    """
    Create a chat agent that can interact with Notion knowledge base.
    """
    logger.info("Creating notion chat agent...")
    try:
        # Create the agent prompt
        logger.info("Creating agent prompt...")
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant with access to a Notion knowledge base. "
                "Use the provided tools to search and retrieve information from Notion "
                "to answer user queries accurately. Always cite the specific Notion pages "
                "you reference.\n\n"
                "Available tools: {tool_names}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Bind tools to the LLM
        logger.info(f"Binding tools to LLM: {[tool.name for tool in tools]}")
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        llm_with_tools = llm.bind_tools(tools)

        # Create the graph
        logger.info("Creating state graph...")
        workflow = StateGraph(AgentState)
        
        # Define the main chat node
        def chat_node(state: AgentState) -> Dict:
            """Process messages and generate responses."""
            logger.debug(f"Processing chat node with state: {state}")
            response = llm_with_tools.invoke(state["messages"])
            logger.debug(f"Generated response: {response}")
            
            # Check if the response contains tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"Tool calls detected: {response.tool_calls}")
                return {"messages": state["messages"] + [response], "next": "tool_executor"}
            
            return {"messages": state["messages"] + [response], "next": END}
        
        # Define the tool execution node
        logger.info("Creating tool node...")
        tool_executor = ToolNode(tools=tools)
        
        # Add nodes to graph
        logger.info("Adding nodes to graph...")
        workflow.add_node("chat", chat_node)
        workflow.add_node("tool_executor", tool_executor)
        
        # Add edges
        logger.info("Adding edges to graph...")
        workflow.add_edge("tool_executor", "chat")
        
        # Add conditional edges for the chat node
        def route_next(state):
            return state["next"]
            
        workflow.add_conditional_edges(
            "chat",
            route_next,
            {
                "tool_executor": "tool_executor",
                END: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("chat")
        
        # Add memory for conversation persistence
        logger.info("Adding memory saver...")
        memory = MemorySaver()
        
        logger.info("Successfully created notion chat agent")
        return workflow.compile(checkpointer=memory)
        
    except Exception as e:
        logger.error(f"Error creating notion chat agent: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 