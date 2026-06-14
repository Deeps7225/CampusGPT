import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>

.user-bubble {
    background-color: #DCF8C6;
    color: black;
    padding: 12px;
    border-radius: 15px;
    margin-left: 35%;
    margin-bottom: 10px;
}

.bot-bubble {
    background-color: #F5F5F5;
    color: black;
    padding: 12px;
    border-radius: 15px;
    margin-right: 35%;
    margin-bottom: 10px;
}

.citation {
    color: #666666;
    font-size: 12px;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)
with st.sidebar:

    st.title("CampusGPT")

    st.markdown("---")

    
    uploaded_file = st.file_uploader(
    "📄 Upload a PDF",
    type=["pdf"]
)

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
st.title("CampusGPT")

st.caption(
    "Ask questions about academics, hostel, fees, regulations and uploaded documents."
)



# -------------------------
# Load Embeddings
# -------------------------

@st.cache_resource
def load_vector_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


vector_store = load_vector_db()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# -------------------------
# PDF Upload
# -------------------------



if uploaded_file is not None:

    save_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(save_path)

    documents = loader.load()

    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = text_splitter.split_documents(
        documents
    )

    uploaded_db = FAISS.from_documents(
    split_docs,embeddings
    )

    st.session_state["uploaded_db"] = uploaded_db
    st.cache_resource.clear()

# -------------------------
# Load LLM
# -------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="YOUR_GROQ_API_KEY"
)

# -------------------------
# User Input
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_history = ""

for msg in st.session_state.messages[-6:]:
    chat_history += f"{msg['role']}: {msg['content']}\n"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question...")
if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    st.markdown(
    f"""
    <div class="user-bubble">
        {question}
    </div>
    """,
    unsafe_allow_html=True
    )
    

    if "uploaded_db" in st.session_state:

     results_with_scores = (
        st.session_state["uploaded_db"]
        .similarity_search_with_score(
            question,
            k=10
        )
    )

    else:

     results_with_scores = (
        vector_store
        .similarity_search_with_score(
            question,
            k=10
        )
    )
    
    results = [doc for doc, score in results_with_scores]

    

    best_score = results_with_scores[0][1]

    st.write("Best Similarity Score:", best_score)

    # -------------------------
    # General Knowledge Fallback
    # -------------------------

    if best_score > 1.5:

        response = llm.invoke(question)

        answer = f"""
        {response.content}

🌐 Source: General Knowledge
"""

    else:

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        prompt = f"""
You are a University AI Assistant.

Use the conversation history to understand references
such as he, she, his, her, it, they, etc.

Conversation History:
{chat_history}

Context:
{context}

Current Question:
{question}

Answer the question clearly.
"""

        response = llm.invoke(prompt)

        answer = f"""
{response.content}

---
📚 Source: xyz.pdf
📄 Page: {results[0].metadata['page_label']}
"""

    st.markdown(
    f"""
    <div class="bot-bubble">
        {response.content}

        
    """,
    unsafe_allow_html=True
)

    st.session_state.messages.append(
    {"role": "assistant", "content": answer}
)