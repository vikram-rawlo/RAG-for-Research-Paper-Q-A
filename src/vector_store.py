import chromadb
from typing import List
from langchain.docstore.document import Document
from langchain_chroma import Chroma

from config.settings import CHROMA_DB_DIRECTORY, COLLECTION_NAME, FORCE_REBUILD, SEARCH_RESULTS_COUNT
from utils.helpers import log_message, calculate_document_hash


class VectorStore:
    """
    Manages a persistent ChromaDB vector store.
    """
    def __init__(self, embeddings_model, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIRECTORY)
        self.collection_name = collection_name
        self.embeddings_model = embeddings_model

        try:
            # If FORCE_REBUILD is True, drop existing collection
            if FORCE_REBUILD:
                try:
                    self.client.delete_collection(self.collection_name)
                    log_message(f"Collection '{self.collection_name}' cleared (FORCE_REBUILD=True).")
                except Exception:
                    log_message(f"Collection '{self.collection_name}' did not exist, nothing to delete.", level="warning")

            # Initialize persistent vector store
            self.vector_store = Chroma(
                client=self.client,
                embedding_function=self.embeddings_model,
                collection_name=self.collection_name
            )
            log_message(f"VectorStore initialized with collection: {self.collection_name}")
        except Exception as e:
            log_message(f"Failed to initialize VectorStore: {e}", level="error")
            raise

    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store with unique IDs (deduplication)."""
        if not documents:
            log_message("No documents to add to the vector store.", level="info")
            return

        try:
            ids = [calculate_document_hash(doc.page_content) for doc in documents]
            self.vector_store.add_documents(documents=documents, ids=ids)
            log_message(f"Added {len(documents)} documents to the vector store.")
        except Exception as e:
            log_message(f"Error adding documents to the vector store: {e}", level="error")
            raise

    def get_retriever(self, k: int = SEARCH_RESULTS_COUNT):
        """Return a retriever instance for similarity search."""
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            log_message(f"Retriever created with k={k}.")
            return retriever
        except Exception as e:
            log_message(f"Error getting retriever: {e}", level="error")
            raise