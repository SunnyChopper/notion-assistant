from typing import Dict, List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
import json

from services.notion_reader import NotionReader, NotionPage
from services.notion_indexer import NotionIndexer


class NotionSearchInput(BaseModel):
    """Input for searching Notion pages."""
    query: str = Field(..., description="The search query to find relevant Notion pages")
    max_results: int = Field(default=3, description="Maximum number of results to return")


class NotionPageInput(BaseModel):
    """Input for retrieving a specific Notion page."""
    page_id: str = Field(..., description="The ID of the Notion page to retrieve")


class NotionSearchTool(BaseTool):
    """Tool for semantic search over Notion pages."""
    
    name: str = "notion_search"
    description: str = "Search through your Notion knowledge base using semantic search"
    args_schema: Type[BaseModel] = NotionSearchInput
    indexer: NotionIndexer = None
    
    def __init__(self, indexer: NotionIndexer):
        """Initialize the tool with a NotionIndexer instance."""
        super().__init__()
        self.indexer = indexer

    def _run(
        self, 
        query: str,
        max_results: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the search tool."""
        try:
            # Use the vector store from the indexer to perform semantic search
            search_results = self.indexer.vector_store.similarity_search(
                query,
                k=max_results
            )
            
            # Format results
            formatted_results = []
            for doc in search_results:
                result = {
                    "page_id": doc.metadata["page_id"],
                    "title": doc.metadata["title"],
                    "content_preview": doc.page_content[:200] + "..."
                }
                formatted_results.append(result)
            
            return json.dumps(formatted_results, indent=2)
            
        except Exception as e:
            return f"Error searching Notion: {str(e)}"


class NotionPageReaderTool(BaseTool):
    """Tool for reading specific Notion pages."""
    
    name: str = "notion_page_reader"
    description: str = "Retrieve and read the content of a specific Notion page"
    args_schema: Type[BaseModel] = NotionPageInput
    
    def _run(
        self,
        page_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the page reader tool."""
        try:
            notion_page: NotionPage = NotionReader.get_page_content(page_id)
            
            # Format the page content
            result = {
                "page_id": notion_page.page_id,
                "properties": notion_page.content,
                "content": notion_page.full_content,
                "child_pages": [
                    {"title": child.title, "page_id": child.page_id}
                    for child in notion_page.child_pages
                ]
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error reading Notion page: {str(e)}"


class NotionKnowledgeGraphTool(BaseTool):
    """Tool for exploring the Notion knowledge graph."""
    
    name: str = "notion_knowledge_graph"
    description: str = "Explore the relationships between Notion pages in your knowledge base"
    indexer: NotionIndexer = None
    
    def __init__(self, indexer: NotionIndexer):
        """Initialize the tool with a NotionIndexer instance."""
        super().__init__()
        self.indexer = indexer
    
    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the knowledge graph tool."""
        try:
            graph = self.indexer.knowledge_graph
            
            # Get basic graph statistics
            stats = {
                "total_pages": graph.number_of_nodes(),
                "total_connections": graph.number_of_edges(),
                "root_page_id": self.indexer.root_page_id
            }
            
            # Get immediate children of root page
            root_children = [
                {
                    "page_id": neighbor,
                    "title": graph.nodes[neighbor].get("title", "Untitled")
                }
                for neighbor in graph.neighbors(self.indexer.root_page_id)
            ]
            
            result = {
                "statistics": stats,
                "root_children": root_children
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error exploring knowledge graph: {str(e)}" 