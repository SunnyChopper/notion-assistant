from typing import Optional, Dict
import concurrent.futures
import hashlib
import logging
import pickle
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import networkx as nx
import pickle

from services.notion_reader import NotionReader, NotionPage

load_dotenv()

logger = logging.getLogger(__name__)

class NotionIndexer:
    '''
    Index the content of a page from Notion to a vector database. This allows for semantic search over the content,
    which can be used for retrieval augmented generation (RAG).
    '''
    DEFAULT_HASH_STORE_PATH: str = 'hash_store.pkl'
    DEFAULT_GRAPH_STORE_PATH: str = 'graph_store.gpickle'

    def __init__(self, vector_store: Chroma, hash_store_path: str = DEFAULT_HASH_STORE_PATH, knowledge_graph_path: str = DEFAULT_GRAPH_STORE_PATH):
        self.root_page_id: str = os.getenv("ROOT_PAGE_ID")
        self.vector_store: Chroma = vector_store
        self.hash_store_path: str = hash_store_path
        self.knowledge_graph_path: str = knowledge_graph_path
        self.total_pages_found: int = 0
        self.pages_processed: int = 0
        self.processed_pages: set = set()
        
        # Load hash store if it exists
        self.hash_store: Dict = {}
        if os.path.exists(hash_store_path):
            with open(hash_store_path, 'rb') as f:
                self.hash_store = pickle.load(f)
                
        # Load knowledge graph if it exists
        if os.path.exists(knowledge_graph_path):
            with open(knowledge_graph_path, 'rb') as f:
                self.knowledge_graph = pickle.load(f)
        else:
            self.knowledge_graph = nx.DiGraph()
            
        self.text_splitter: CharacterTextSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def save_hash_store(self):
        '''
        Save the hash store to a file, which is used to prevent duplicate pages from being indexed.
        '''
        with open(self.hash_store_path, 'wb') as f:
            pickle.dump(self.hash_store, f)

    def save_knowledge_graph(self):
        '''
        Save the knowledge graph to a file for retrieval later.
        '''
        with open(self.knowledge_graph_path, 'wb') as f:
            pickle.dump(self.knowledge_graph, f)

    def save_processed_pages(self):
        '''
        Save the set of processed pages to a file.
        '''
        with open('processed_pages.pkl', 'wb') as f:
            pickle.dump(self.processed_pages, f)

    def load_processed_pages(self):
        '''
        Load the set of processed pages from a file.
        '''
        if os.path.exists('processed_pages.pkl'):
            with open('processed_pages.pkl', 'rb') as f:
                self.processed_pages = pickle.load(f)

    def process_page(self, page_id: str, parent_id: Optional[str] = None):
        '''
        Process a page and its child pages recursively.
        '''
        notion_page: NotionPage = NotionReader.get_page_content(page_id)
        content: str = notion_page.full_content
        current_hash: str = hashlib.md5(content.encode('utf-8')).hexdigest()
        self.total_pages_found += len(notion_page.child_pages)

        # Get the page title for logging
        title_list = notion_page.content.get('title', ['Untitled'])
        title = title_list[0] if title_list else 'Untitled'

        # Check if page needs processing
        is_new_page: bool = page_id not in self.hash_store
        is_page_modified: bool = current_hash != self.hash_store.get(page_id)
        is_already_processed: bool = page_id in self.processed_pages

        if is_already_processed:
            logger.info(f"Page {self.pages_processed + 1}/{self.total_pages_found + 1}: {title} already processed. Checking child pages...")
        elif is_new_page or is_page_modified:
            logger.info(f"Processing page {self.pages_processed + 1}/{self.total_pages_found + 1}: {title}")
            
            chunks = self.text_splitter.split_text(content)
            if not chunks:
                logger.warning(f"No content to index for page {page_id}. Skipping...")
            else:
                try:
                    self.vector_store._collection.delete(
                        where={"page_id": page_id}
                    )
                except Exception as e:
                    logger.warning(f"Error deleting page {page_id} from vector store: {e}")

                # Parallelize embedding generation
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_chunk = {}
                    for chunk in chunks:
                        logger.info(f"Submitting chunk for page {page_id} to executor.")
                        future = executor.submit(self.vector_store.add_texts, [chunk], [{"page_id": page_id, "title": title}])
                        future_to_chunk[future] = chunk

                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk = future_to_chunk[future]
                        try:
                            future.result()
                            logger.info(f"Successfully processed chunk for page {page_id}.")
                        except Exception as e:
                            logger.error(f"Error processing chunk for page {page_id}: {e}")

                logger.info(f"Completed processing of page {page_id}.")
                self.hash_store[page_id] = current_hash
                self.save_hash_store()
                self.processed_pages.add(page_id)
                self.save_processed_pages()
        else:
            logger.info(f"Skipping page {self.pages_processed + 1}/{self.total_pages_found + 1}: {title} (already indexed)")

        self.pages_processed += 1
        self.knowledge_graph.add_node(page_id, title=title)

        if parent_id:
            self.knowledge_graph.add_edge(parent_id, page_id)
            
        # Always process child pages, regardless of parent's status
        if notion_page.child_pages:
            logger.info(f"Found {len(notion_page.child_pages)} child pages for {title}")
            for child_page in notion_page.child_pages:
                child_id = child_page.page_id
                self.process_page(page_id=child_id, parent_id=page_id)

    def run(self):
        '''
        Run the indexer on Notion starting from the root page.
        '''
        logger.info(f"Starting indexing from root page...")
        self.load_processed_pages()
        self.total_pages_found = 1  # Start with root page
        self.pages_processed = 0
        self.process_page(page_id=self.root_page_id)
        self.save_knowledge_graph()
        logger.info(f"Indexing complete. Processed {self.pages_processed} pages total.")
        return self.knowledge_graph

