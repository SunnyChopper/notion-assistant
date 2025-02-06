# Notion Knowledge Base Assistant

The Notion Knowledge Base Assistant is an interactive tool designed to integrate with your Notion workspace. It leverages the Notion API, vector store indexing, and OpenAI-powered language models to enable semantic search, page retrieval, and relationship exploration of your Notion pages. Using a combination of specialized agents and tools, you can query your Notion content and receive detailed, context-rich responses.

**Description**: Custom help from a chatbot that uses your Notion as a knowledge base for answering questions. 

---

## Features

- **Notion Page Reading**  
  Retrieve and process Notion pages with detailed content extraction including properties and child pages.

- **Semantic Search**  
  Perform semantic searches over your indexed Notion content using vector embeddings.

- **Knowledge Graph Exploration**  
  Visualize and explore the relationships between pages in your Notion workspace via a knowledge graph.

- **Interactive Chat Interface**  
  Ask questions and interact with your Notion content in a conversational format using the integrated chat agent.

- **Indexing**  
  Automatically index Notion pages with semantic embeddings and maintain a processed pages log to avoid duplicates.

---

## Architecture Overview

The codebase is organized into several modules:

- **services/**  
  - **notion_reader.py**: Handles retrieving page content and blocks from Notion’s API.  
  - **notion_indexer.py**: Indexes pages recursively, generates embeddings using a vector store (Chroma), and builds a knowledge graph.

- **tools/**  
  - **notion_tools.py**: Contains the tools for semantic search (`NotionSearchTool`), page reading (`NotionPageReaderTool`), and knowledge graph exploration (`NotionKnowledgeGraphTool`).

- **agents/**  
  - **base.py**: Defines the base Notion chat agent state graph that processes messages and routes tool calls.  
  - **orchestrator.py**: Orchestrates the specialized agents (chat and search) and manages conversation threads.  
  - **search.py**: Sets up a specialized search agent for querying and retrieving relevant Notion pages.

- **models/**  
  - **agent_state.py**: Defines the conversational state model used by the agents.

- **run.py**  
  The entry point for setting up the agent and launching the interactive chat loop.

- **Configuration Files**  
  - `.gitignore`: Excludes cached files, virtual environments, and sensitive data (e.g., `.env`).
  - `requirements.txt`: Lists the Python dependencies for the project.

---

## Getting Started

### Prerequisites

- **Python 3.7+**  
- **Notion API Credentials**:  
  - A valid `NOTION_TOKEN`
  - A `ROOT_PAGE_ID` corresponding to the top-level page in your Notion workspace
- **OpenAI API Key**:  
  - A valid `OPENAI_API_KEY`

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/SunnyChopper/notion-assistant
   cd notion-assistant
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the root directory with the following content:

   ```env
   NOTION_TOKEN=your_notion_token
   ROOT_PAGE_ID=your_root_page_id
   OPENAI_API_KEY=your_openai_api_key
   ```

   Replace `your_notion_token` and `your_root_page_id` with your actual Notion API token and page ID.

---

## Running the Assistant

Execute the following command to launch the interactive chat loop:
```bash
python run.py
```

Once running, you will be presented with a command-line interface that supports:

- **General Queries**: Ask questions to retrieve context-rich answers from your Notion workspace.
- **Indexing Command**: Type `index` to refresh or build the Notion index.
- **Conversation Reset**: Type `clear` to clear the conversation history.
- **Exit**: Type `quit` or `exit` (or press Ctrl+C) to end the session.

---

## How It Works

1. **Notion Page Processing**:  
   The `NotionReader` fetches page content and blocks from Notion. The `NotionIndexer` recursively processes pages starting from the root page, generating embeddings with the Chroma vector store and building a knowledge graph to capture page relationships.

2. **Semantic Search & Retrieval**:  
   The `NotionSearchTool` leverages the vector store to perform semantic queries. In conjunction with the `NotionPageReaderTool`, it retrieves relevant pages and provides content previews.

3. **Interactive Chat**:  
   The orchestrator in `agents/orchestrator.py` combines the outputs of a specialized search agent and chat agent. This chain of processing ensures that queries are enriched with context from the indexed Notion content before generating the final response.

---

## Code Structure Reference

- **services/notion_reader.py**  
  Implements classes to read Notion pages and process their content.

- **services/notion_indexer.py**  
  Indexes Notion content for semantic search and constructs a knowledge graph.

- **agents/base.py, orchestrator.py, search.py**  
  Define the conversation flow and behaviors of the chat and search agents.

- **tools/notion_tools.py**  
  Contains tools for searching, reading, and graph exploration of Notion pages.

- **models/agent_state.py**  
  Defines the state model used throughout the chat and search process.

- **run.py**  
  Serves as the entry point, setting up the Notion agent and starting the chat loop.

---

## Troubleshooting & Logging

- **Error Handling**:  
  Detailed error information is logged in the console. Common issues include invalid API credentials or network errors.

- **Logging**:  
  The system uses Python’s logging module. Check the console output for debugging messages or errors during execution.

- **Before Re-indexing**:  
  If encountering duplicate or outdated information, consider clearing the processed pages file (`processed_pages.pkl`) and the hash store (`hash_store.pkl`).

---

## Acknowledgements

- **Notion API** – For providing access to Notion pages.
- **LangChain and LangGraph** – For enabling modular agent workflows.
- **Chroma** – For handling vector store operations and semantic searches.
- **OpenAI** – For powering the language model responses.

Enjoy exploring and interacting with your Notion knowledge base!