import os

from langchain_openai import OpenAIEmbeddings

from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL
from utils.helpers import log_message


class EmbeddingManager:
    """
    Manages the OpenAI embeddings model for text vectorization.
    """
    def __init__(self):
        try:
            # Load API key (from env or settings)
            api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set. Please configure it in your environment or settings.py.")

            # Initialize embeddings model
            self.model = OpenAIEmbeddings(
                openai_api_key=api_key,
                model=EMBEDDING_MODEL
            )
            log_message(f"EmbeddingManager initialized with model '{EMBEDDING_MODEL}'.")
        except Exception as e:
            log_message(f"Failed to initialize EmbeddingManager: {e}", level="error")
            raise

    def get_embeddings_model(self):
        """Return the embeddings model instance."""
        return self.model