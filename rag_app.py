"""
rag_app.py - Local RAG backend

Knowledge source:
    10_Day_Streamlit_Learning_Plan(1).pdf

Flow:
    PDF -> LangChain chunks -> HuggingFace embeddings -> FAISS (local)
        -> user question -> FAISS retrieval -> gpt-4o-mini -> answer
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT_NAME")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Keep everything local to this project.
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "10_Day_Streamlit_Learning_Plan.pdf"
FAISS_DIR = BASE_DIR / "faiss_db"

# Small local models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Create the small local embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store():
    """Read the PDF, split it, embed it and save FAISS locally."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Put the PDF in the same folder as rag_app.py."
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    vector_store.save_local(str(FAISS_DIR))

    return vector_store, len(chunks)


def get_vector_store():
    """Load an existing local FAISS DB, or create it the first time."""
    embeddings = get_embeddings()

    if FAISS_DIR.exists() and (FAISS_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vector_store, _ = build_vector_store()
    return vector_store


def ask_question(question: str, k: int = 3):
    """Retrieve relevant chunks and ask gpt-4o-mini using only retrieved context."""
    vector_store = get_vector_store()

    docs = vector_store.similarity_search(question, k=k)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""You are a helpful assistant for the Streamlit Learning Plan.

Answer the user's question using ONLY the context below.
If the answer is not present in the context, say:
"I couldn't find that information in the uploaded document."

Explain the answer in plain, easy-to-understand language.
Keep it short and clear. Do not mention chunks or page numbers.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    answer = llm.invoke(prompt)

    return answer, docs


if __name__ == "__main__":
    store, count = build_vector_store()
    print(f"FAISS database created at: {FAISS_DIR}")
    print(f"Created {count} document chunks.")
