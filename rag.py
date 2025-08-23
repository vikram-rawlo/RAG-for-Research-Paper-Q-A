# rag.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")  # default if not set

# ---------------------------------------------------
# Initialize embeddings and vector DB
# ---------------------------------------------------
embeddings = OpenAIEmbeddings()

chroma_db = Chroma(
    persist_directory=chroma_db_dir,
    collection_name="langchain",
    embedding_function=embeddings
)

# ---------------------------------------------------
# Initialize LLM
# ---------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------
# Custom Prompt
# ---------------------------------------------------
prompt_template = """
You are a helpful assistant. Use the provided context to answer the question.
If you can find relevant information in the context, provide an answer based on that information.
Only if you cannot find ANY relevant information should you respond with:
"This tool only answers questions based on the documents in its database. Please ask something within that scope."

Context:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# ---------------------------------------------------
# Create Retriever + QA Chain
# ---------------------------------------------------
retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True,
    verbose=True
)

# ---------------------------------------------------
# Query function
# ---------------------------------------------------
def run_query(query: str):
    print("Running query:", query)
    try:
        result = qa_chain.invoke({"query": query})
        print("Chain finished.")
        answer = result["result"]
        sources = result["source_documents"]
        return answer, sources
    except Exception as e:
        print("❌ Error during query:", str(e))
        return None, None


# ---------------------------------------------------
# For direct testing
# ---------------------------------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        print("You typed:", query)
        if query.lower() == "exit":
            break
        answer, sources = run_query(query)
        print("\nAnswer:", answer)
        print("Number of source documents:", len(sources))
