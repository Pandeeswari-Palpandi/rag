"""
app.py - Streamlit UI for the local RAG chatbot
"""

import streamlit as st

from rag_app import ask_question, build_vector_store, FAISS_DIR


st.set_page_config(
    page_title="Streamlit Plan RAG",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    [data-testid="stChatMessage"] {
        padding: 1rem 1.25rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 1rem;
        margin-bottom: 0.75rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(49, 51, 63, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Streamlit 10 days Learning Plan Chatbot")
st.caption("Ask questions about your 10-Day Streamlit Learning Plan")


# Sidebar
with st.sidebar:
    st.header("⚙️ RAG Settings")
    st.write("**Vector Store:** FAISS")
    st.write("**LLM:** GPT-4o-mini")
    st.write("**Embeddings:** all-MiniLM-L6-v2")
    st.write("**Storage:** Local")

    if st.button("🔄 Rebuild FAISS DB", use_container_width=True):
        with st.spinner("Reading PDF and rebuilding vector store..."):
            try:
                build_vector_store()
                st.success("FAISS database rebuilt successfully.")
            except Exception as e:
                st.error(str(e))

    if FAISS_DIR.exists():
        st.success("FAISS DB is available")
    else:
        st.info("FAISS DB will be created on first question.")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask something about the 10-Day Streamlit Learning Plan...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔎 Searching FAISS and generating answer..."):
            try:
                answer, docs = ask_question(question, k=3)
                answer_text = answer.content

                st.markdown(answer_text)

                with st.expander("📚 Relevant content"):
                    st.markdown(
                        "\n\n---\n\n".join(doc.page_content for doc in docs)
                    )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text}
                )

            except Exception as e:
                st.error(
                    "Something went wrong. Make sure OpenAI API key is set."
                )
                st.code(str(e))
