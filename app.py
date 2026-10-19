import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json

load_dotenv()

st.set_page_config(page_title="Osho Chatbot", page_icon="🕉️", layout="wide")

# Load vector store
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return FAISS.load_local("data/vectorstore", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Vector store not found. Please run `python scripts/index_data.py` first. Error: {e}")
        return None

vectorstore = load_vectorstore()

# Groq LLM
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set. Please set it in environment or .env file.")
        return None
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=api_key)

llm = get_llm()

# LangGraph state
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    sources: List[dict]

def retrieve(state: GraphState):
    question = state["question"]
    if not vectorstore:
        return {"documents": [], "sources": []}
    docs = vectorstore.similarity_search(question, k=4)
    sources = []
    for doc in docs:
        sources.append({
            "title": doc.metadata.get("title", "Unknown"),
            "url": doc.metadata.get("source", ""),
            "snippet": doc.page_content[:300] + "..."
        })
    return {"documents": docs, "sources": sources}

def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('title', 'Discourse')}\n{doc.page_content}" for doc in documents])
    
    prompt = ChatPromptTemplate.from_template("""
You are an Osho chatbot. Answer the user's question based ONLY on the provided context from Osho's discourses.
Be insightful, direct, and in the spirit of Osho's teachings.
Always cite the sources you used.

Context:
{context}

Question: {question}

Answer with references:
""")
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return {"answer": response.content}

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app_graph = workflow.compile()

# Streamlit UI
st.title("🕉️ Osho Chatbot")
st.markdown("Ask questions based on Osho's discourses from [oshoworld.com](https://oshoworld.com/). Answers include source references.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['title']}**  \n[Link]({src['url']})  \n_{src['snippet']}_")

if prompt := st.chat_input("Ask Osho a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Consulting the discourses..."):
            if vectorstore and llm:
                result = app_graph.invoke({"question": prompt})
                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(f"**{src['title']}**  \n[Link]({src['url']})  \n_{src['snippet']}_")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                st.error("Setup incomplete. Check API key and vector store.")
