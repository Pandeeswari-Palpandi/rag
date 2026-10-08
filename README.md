# Streamlit Learning Plan Chatbot

A Streamlit chat app that answers questions about the 10-Day Streamlit Learning
Plan PDF. It uses LangChain to split the document, Hugging Face sentence
embeddings and a local FAISS vector store to find relevant passages, then
OpenAI GPT-4o-mini to write a concise answer. Retrieved passages are available
in the **Relevant content** expander.

## Project files

- `pdf-chatbot-ui.py` — Streamlit interface.
- `rag_app.py` — PDF loading, vector-store creation, retrieval, and answer
  generation.
- `data/10_Day_Streamlit_Learning_Plan.pdf` — source document.
- `faiss_db/` — locally persisted FAISS index, created on first use.

## Setup

Run these commands from the project directory. Create and activate a virtual
environment if you do not already have one:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory. The app expects all four
variables to be defined; an empty value is acceptable for the LangSmith
settings when tracing is disabled:

```dotenv
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT_NAME=
```

Set `LANGSMITH_TRACING=true` and provide your LangSmith API key and project
name to enable tracing. The OpenAI API key is required for answers.

The PDF must be at `data/10_Day_Streamlit_Learning_Plan.pdf`. On the first
question, the app downloads the `sentence-transformers/all-MiniLM-L6-v2`
embedding model if it is not already available, builds the FAISS index, and
saves it under `faiss_db/`.

## Run

```powershell
streamlit run pdf-chatbot-ui.py
```

If Streamlit's file watcher logs repeated `ModuleNotFoundError: No module
named 'torchvision'` messages while inspecting optional Transformers image
modules, disable the watcher:

```powershell
streamlit run pdf-chatbot-ui.py --server.fileWatcherType none
```

With the watcher disabled, restart Streamlit after changing project files.
