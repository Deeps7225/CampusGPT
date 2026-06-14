import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

# -------------------------
# Load Embedding Model
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Load Saved FAISS Database
# -------------------------

vector_store = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# -------------------------
# Load Llama Model
# -------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="YOUR_GROQ_API_KEY"
)

# -------------------------
# Assistant Function
# -------------------------

def ask_university_assistant(question):

    results = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
    You are a University Assistant.

    Answer the question using the provided context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    print("\nANSWER:\n")
    print(response.content)

    print("\nSOURCE:\n")
    print(
        os.path.basename(
            results[0].metadata["source"]
        )
    )

    print("PAGE:")
    print(results[0].metadata["page_label"])


# -------------------------
# Main Loop
# -------------------------

while True:

    question = input("\nAsk Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    ask_university_assistant(question)