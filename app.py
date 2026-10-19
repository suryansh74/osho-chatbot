import os
from pathlib import Path
from typing import List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

st.set_page_config(page_title="Osho Chatbot", page_icon="🕉️", layout="wide")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return FAISS.load_local("data/vectorstore", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Vector store not found. Run indexing first. ({e})")
        return None


@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Set GROQ_API_KEY in .env")
        return None
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=api_key)


vectorstore = load_vectorstore()
llm = get_llm()


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    sources: List[dict]
    needs_evidence: bool
    evidence: str


def retrieve(state: GraphState):
    if not vectorstore:
        return {"documents": [], "sources": []}
    docs = vectorstore.similarity_search(state["question"], k=5)
    sources = []
    for doc in docs:
        sources.append({
            "title": doc.metadata.get("title", "Unknown"),
            "url": doc.metadata.get("source", ""),
            "local_audio": doc.metadata.get("local_audio", ""),
            "relative_audio": doc.metadata.get("relative_audio", ""),
            "audio_url": doc.metadata.get("audio_url", ""),
            "snippet": doc.page_content[:400] + "...",
            "language": doc.metadata.get("language", ""),
        })
    return {"documents": docs, "sources": sources}


def generate(state: GraphState):
    context = "\n\n---\n\n".join(
        f"Title: {doc.metadata.get('title', 'Discourse')}\n{doc.page_content}"
        for doc in state["documents"]
    )
    prompt = ChatPromptTemplate.from_template(
        """You are an Osho chatbot. Answer using ONLY the provided context from Osho's discourses.
Be insightful and in the spirit of Osho. Cite the discourse title(s).

Context:
{context}

Question: {question}

Answer with references:"""
    )
    response = (prompt | llm).invoke({"context": context, "question": state["question"]})
    return {"answer": response.content}


def decide_evidence(state: GraphState):
    check = ChatPromptTemplate.from_template(
        """Does this question or answer involve a specific historical claim, external story, or fact that would benefit from independent verification?
Answer only YES or NO.

Question: {question}
Answer: {answer}"""
    )
    decision = (check | llm).invoke({
        "question": state["question"],
        "answer": state.get("answer", "")[:600],
    })
    return {"needs_evidence": "YES" in decision.content.upper()}


def search_evidence(state: GraphState):
    if not state.get("needs_evidence"):
        return {"evidence": ""}
    return {
        "evidence": (
            "[Evidence node] This claim may benefit from external verification. "
            "Web search can be plugged in here later (Tavily / SerpAPI / etc.)."
        )
    }


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("decide_evidence", decide_evidence)
workflow.add_node("search_evidence", search_evidence)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "decide_evidence")
workflow.add_edge("decide_evidence", "search_evidence")
workflow.add_edge("search_evidence", END)
app_graph = workflow.compile()

# ---------- UI ----------
st.title("🕉️ Osho Chatbot")
st.markdown("Answers from your local Osho audio transcripts (or scraped data). Plays original audio when available.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources & Audio"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['title']}** ({src.get('language','')})")
                    audio_path = src.get("local_audio")
                    if audio_path and Path(audio_path).exists():
                        st.audio(audio_path)
                    elif src.get("audio_url"):
                        st.markdown(f"[Audio]({src['audio_url']})")
                    st.caption(src["snippet"])
        if msg.get("evidence"):
            with st.expander("External evidence"):
                st.markdown(msg["evidence"])

if prompt := st.chat_input("Ask about a teaching or story..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching discourses..."):
            if vectorstore and llm:
                result = app_graph.invoke({"question": prompt})
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                evidence = result.get("evidence", "")

                st.markdown(answer)
                if sources:
                    with st.expander("Sources & Audio"):
                        for src in sources:
                            st.markdown(f"**{src['title']}** ({src.get('language','')})")
                            audio_path = src.get("local_audio")
                            if audio_path and Path(audio_path).exists():
                                st.audio(audio_path)
                            elif src.get("audio_url"):
                                st.markdown(f"[Play audio]({src['audio_url']})")
                            st.caption(src["snippet"])
                if evidence:
                    with st.expander("External evidence"):
                        st.markdown(evidence)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "evidence": evidence,
                })
            else:
                st.error("Setup incomplete.")
