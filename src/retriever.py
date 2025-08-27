from langchain.vectorstores.base import VectorStoreRetriever

from config.settings import SEARCH_RESULTS_COUNT
from utils.helpers import log_message
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore  # reuse your existing implementation


# ----------------------------------------------------------------------
# 🔹 Module-level helper functions for QA chains & scripts
# ----------------------------------------------------------------------

def create_retriever(k: int = SEARCH_RESULTS_COUNT) -> VectorStoreRetriever:
    """
    Factory function to return a retriever object for QA chains.
    Uses the shared VectorStore implementation (with persistence & deduplication).
    """
    try:
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore(embedding_manager.get_embeddings_model())
        retriever = vector_store.get_retriever(k=k)
        log_message(f"Retriever created with k={k}.", level="info")
        return retriever
    except Exception as e:
        log_message(f"Error creating retriever: {e}", level="error")
        raise


def retrieve_documents(query: str, k: int = SEARCH_RESULTS_COUNT):
    """
    Convenience function: directly query documents without manually handling the store.
    """
    try:
        retriever = create_retriever(k=k)
        results = retriever.invoke(query)
        log_message(f"Retrieved {len(results)} documents for query: '{query}'", level="info")
        return results
    except Exception as e:
        log_message(f"Error retrieving documents: {e}", level="error")
        return []