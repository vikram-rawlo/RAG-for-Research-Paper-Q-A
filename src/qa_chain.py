import logging
from typing import List, Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from config import settings
from src.retriever import create_retriever
from utils.helpers import log_message

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_llm(model_name: Optional[str] = None, temperature: Optional[float] = None) -> ChatOpenAI:
    """
    Instantiate and return the Chat LLM wrapper.
    """
    try:
        return ChatOpenAI(
            model_name=model_name or settings.LLM_MODEL,
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=1000
        )
    except Exception as e:
        log_message(f"Error initializing LLM: {e}", level="error")
        raise


def create_prompt_template() -> PromptTemplate:
    """
    Build and return a PromptTemplate that instructs the model to answer only
    from provided context, and return a specific fallback message if the context
    doesn't contain the answer.
    """
    template = (
        "You are a helpful assistant that answers questions based only on the provided context.\n"
        "If the answer cannot be found in the context, respond exactly with:\n"
        "\"I don't have enough information to answer that question based on the provided documents.\"\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])


def process_source_documents(source_docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Convert retrieved Document objects into a simple metadata list for the UI.
    """
    processed = []
    for i, doc in enumerate(source_docs):
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        processed.append({
            "chunk_id": i + 1,
            "content_preview": preview,
            "source_file": doc.metadata.get("source", "Unknown"),
            "page_number": doc.metadata.get("page", "Unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "Unknown")
        })
    return processed


def calculate_confidence(retrieved_docs: List[Document]) -> float:
    """
    A simple heuristic confidence score:
    - More retrieved chunks -> higher base confidence (saturates at 1.0)
    - Add a small boost if page metadata exists
    """
    if not retrieved_docs:
        return 0.0

    base_confidence = min(len(retrieved_docs) / 5.0, 1.0)
    if any(doc.metadata.get("page") for doc in retrieved_docs):
        base_confidence = min(base_confidence + 0.1, 1.0)

    return round(base_confidence, 2)


def ask_question(
    question: str,
    k: Optional[int] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """
    Main QA entrypoint for the app:
      - obtains a retriever (wired to your Chroma DB / embeddings),
      - fetches relevant chunks for `question` (using `k` or settings default),
      - if no chunks found, returns a friendly 'no relevant documents' response,
      - otherwise builds a prompt and queries the LLM,
      - returns structured response containing answer, source docs and confidence.
    """
    try:
        log_message(f"Asking question: {question}", level="info")

        # Use configured default k if not provided
        k_val = k if k is not None else settings.SEARCH_RESULTS_COUNT

        # Create retriever (this will reuse your vector_store implementation)
        retriever = create_retriever(k=k_val)
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            log_message("No relevant documents found for query.", level="info")
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "source_documents": [],
                "confidence": 0.0,
                "retrieved_chunks": 0
            }

        # Build context from retrieved chunks
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Create LLM and prompt
        llm = create_llm(model_name=model_name, temperature=temperature)
        prompt = create_prompt_template()
        prompt_text = prompt.format(context=context, question=question)

        # Call the LLM. Depending on your langchain_openai wrapper, the call may vary.
        # Using `invoke` here follows the pattern used elsewhere in your project.
        # If your environment expects a different call (e.g., llm.generate or llm.__call__),
        # replace the next line accordingly.
        llm_response = llm.invoke(prompt_text)

        # Extract the model answer text safely
        answer_text = getattr(llm_response, "content", None)
        if answer_text is None:
            # Try other common attributes if needed
            answer_text = str(llm_response)

        return {
            "answer": answer_text.strip(),
            "source_documents": process_source_documents(retrieved_docs),
            "confidence": calculate_confidence(retrieved_docs),
            "retrieved_chunks": len(retrieved_docs)
        }

    except Exception as e:
        log_message(f"Error in ask_question: {e}", level="error")
        return {
            "answer": f"Error while processing your question: {str(e)}",
            "source_documents": [],
            "confidence": 0.0,
            "retrieved_chunks": 0
        }


def summarize_documents(
    k: Optional[int] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convenience function to summarize the collection by reusing ask_question.
    """
    return ask_question(
        question="Provide a comprehensive summary of the main topics and key points present in the retrieved documents.",
        k=k,
        model_name=model_name,
        temperature=temperature
    )