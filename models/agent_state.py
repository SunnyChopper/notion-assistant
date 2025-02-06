from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolExecutor
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """State for the Notion knowledge base interaction agent."""
    
    # Messages being accumulated in conversation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Current status of the agent
    status: str
    
    # Store for intermediate steps/results
    intermediate_steps: List[tuple]
    
    # Knowledge context from Notion
    notion_context: Dict[str, Any]
    
    # Current query or task being processed
    current_query: Optional[str]
    
    # Search results and relevant pages
    search_results: List[Dict[str, Any]]
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the messages list."""
        self.messages.append(HumanMessage(content=message))
        self.current_query = message
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the messages list."""
        self.messages.append(AIMessage(content=message))
    
    def add_notion_context(self, context: Dict[str, Any]) -> None:
        """Update the Notion context with new information."""
        self.notion_context.update(context)
    
    def add_search_result(self, result: Dict[str, Any]) -> None:
        """Add a search result to the search results list."""
        self.search_results.append(result)
    
    def clear_search_results(self) -> None:
        """Clear the search results list."""
        self.search_results = []
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the full conversation history."""
        return self.messages
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True 