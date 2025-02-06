import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from services.notion_indexer import NotionIndexer
from tools.notion_tools import NotionSearchTool, NotionPageReaderTool, NotionKnowledgeGraphTool
from agents.orchestrator import NotionAgentOrchestrator


def setup_notion_agent(
    vector_store_path: str = "notion_store",
    use_mini: bool = False
) -> NotionAgentOrchestrator:
    """
    Set up the Notion agent with all necessary components.
    
    Args:
        vector_store_path: Path to store/load the vector database
        use_mini: Whether to use the smaller GPT-4 model for cost savings
        
    Returns:
        NotionAgentOrchestrator: The configured agent orchestrator
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Initialize or load vector store
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embeddings
    )
    
    # Initialize indexer
    indexer = NotionIndexer(vector_store=vector_store)
    
    # Initialize tools
    search_tool = NotionSearchTool(indexer=indexer)
    reader_tool = NotionPageReaderTool()
    graph_tool = NotionKnowledgeGraphTool(indexer=indexer)
    
    # Initialize language models
    # Use gpt-4o-mini for cost efficiency or gpt-4o for best performance
    model_name = "gpt-4o-mini" if use_mini else "gpt-4o"
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        streaming=True  # Enable streaming for better UX
    )
    
    # Create and return orchestrator
    return NotionAgentOrchestrator(
        llm=llm,
        search_tool=search_tool,
        reader_tool=reader_tool,
        graph_tool=graph_tool
    )


def chat_loop(
    orchestrator: NotionAgentOrchestrator,
    thread_id: Optional[str] = None
) -> None:
    """
    Run an interactive chat loop with the Notion agent.
    
    Args:
        orchestrator: The configured NotionAgentOrchestrator
        thread_id: Optional thread ID for conversation persistence
    """
    print("Welcome to your Notion Knowledge Base Assistant!")
    print("Type 'quit', 'exit', or press Ctrl+C to end the conversation.")
    print("Type 'index' to refresh the Notion index.")
    print("Type 'clear' to clear the conversation history.")
    print("-" * 50)
    
    # Generate a thread ID if none provided
    if not thread_id:
        import uuid
        thread_id = str(uuid.uuid4())
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
                
            # Check for special commands
            if user_input.lower() == 'index':
                print("\nIndexing Notion knowledge base...")
                orchestrator.tools[0].indexer.run()  # Run the indexer
                print("Indexing complete!")
                continue
                
            if user_input.lower() == 'clear':
                orchestrator.reset_thread(thread_id)
                print("\nConversation history cleared!")
                continue
            
            # Process normal chat input
            try:
                response = orchestrator.chat(user_input, thread_id=thread_id)
                print("\nAssistant:", response)
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again or type 'clear' to reset the conversation.")
                
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    # Set up the agent
    agent = setup_notion_agent(use_mini=True)  # Use mini model by default for cost efficiency
    
    # Start the chat loop
    chat_loop(agent) 