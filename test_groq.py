from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="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"
)

response = llm.invoke("Hello")

print(response.content)