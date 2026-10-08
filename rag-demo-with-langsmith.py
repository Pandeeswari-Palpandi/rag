from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT_NAME")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

query = "What is the purpose of LangSmith?"

loader = TextLoader("data/langsmith.txt", encoding="utf8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)

results = vectorstore.similarity_search(query, k=3)

context = "\n\n".join(r.page_content for r in results)

print("Query:", query)

if (os.getenv("OPENAI_API_KEY")):
    from openai import OpenAI
    from langsmith.wrappers import wrap_openai

    client = wrap_openai(OpenAI())
    prompt = "Use only this text: \n" + context + "\n\n Question: " + query
    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role": "user", "content": prompt}])
    print("\n Answer: ")
    print(response.choices[0].message.content)
else:
    print("\n (No OpenAI_API_KEY set) here is the context retrieved from the vector store:\n")
    print(context)
