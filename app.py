import json
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

# ---------- Load resources ----------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return FAISS.load_local(
            "data/vectorstore", embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Vector store not found. Run `python scripts/index_data.py` after scraping. ({e})")
        return None


@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Set GROQ_API_KEY in environment or .env")
        return None
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=api_key)


vectorstore = load_vectorstore()
llm = get_llm()

# ---------- LangGraph ----------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    sources: List[dict]
    needs_evidence: bool
    evidence: str


def retrieve(state: GraphState):
    question = state["question"]
    if not vectorstore:
        return {"documents": [], "sources": []}
    docs = vectorstore.similarity_search(question, k=5)
    sources = []
    for doc in docs:
        sources.append(
            {
                "title": doc.metadata.get("title", "Unknown"),
                "url": doc.metadata.get("source", ""),
                "local_audio": doc.metadata.get("local_audio", ""),
                "audio_url": doc.metadata.get("audio_url", ""),
                "snippet": doc.page_content[:400] + "...",
                "language": doc.metadata.get("language", ""),
            }
        )
    return {"documents": docs, "sources": sources}


def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    context = "\n\n---\n\n".join(
        [
            f"Title: {doc.metadata.get('title', 'Discourse')}\nURL: {doc.metadata.get('source', '')}\n{doc.page_content}"
            for doc in documents
        ]
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a knowledgeable Osho chatbot. Answer the user's question using ONLY the provided context from Osho's discourses.
Be insightful, clear, and in the spirit of Osho's style.
Always mention which discourse(s) the answer comes from.

Context:
{context}

Question: {question}

Answer (with references):"""
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return {"answer": response.content}


def decide_evidence(state: GraphState):
    """Simple decision node: whether external evidence/search is useful.
    Currently uses a lightweight LLM check. Later you can plug in real web search."""
    question = state["question"]
    answer = state.get("answer", "")

    # Heuristic + LLM decision
    check_prompt = ChatPromptTemplate.from_template(
        """Does the following question or answer involve a specific historical claim, external story, 
scientific fact, or something that would benefit from independent verification outside Osho's discourses?
Answer only YES or NO.

Question: {question}
Answer excerpt: {answer}"""
    )
    decision = (check_prompt | llm).invoke({"question": question, "answer": answer[:500]})
    needs = "YES" in decision.content.upper()
    return {"needs_evidence": needs}


def search_evidence(state: GraphState):
    """Placeholder for web search. Later replace with Tavily / SerpAPI / DuckDuckGo.
    For now returns a note."""
    if not state.get("needs_evidence"):
        return {"evidence": ""}
    # TODO: integrate actual search tool here
    evidence = (
        "[Evidence search node activated]\n"
        "This is a placeholder. Later this node will call a web search API "
        "to verify historical or external claims mentioned in the discourse."
    )
    return {"evidence": evidence}


# Build graph
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

# ---------- Streamlit UI ----------
st.title("🕉️ Osho Chatbot")
st.markdown(
    "Personal chatbot based on Osho's discourses from [oshoworld.com](https://oshoworld.com/). "
    "Answers include source references and original audio when available."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources & Audio"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['title']}** ({src.get('language', '')})")
                    if src.get("url"):
                        st.markdown(f"[Original page]({src['url']})")
                    if src.get("local_audio") and Path(src["local_audio"]).exists():
                        st.audio(src["local_audio"])
                    elif src.get("audio_url"):
                        st.markdown(f"[Audio link]({src['audio_url']})")
                    st.caption(src["snippet"])
        if msg.get("evidence"):
            with st.expander("External evidence / verification"):
                st.markdown(msg["evidence"])

if prompt := st.chat_input("Ask about a teaching, story, or concept..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the discourses..."):
            if vectorstore and llm:
                result = app_graph.invoke({"question": prompt})
                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                evidence = result.get("evidence", "")

                st.markdown(answer)

                if sources:
                    with st.expander("Sources & Audio"):
                        for src in sources:
                            st.markdown(f"**{src['title']}** ({src.get('language', '')})")
                            if src.get("url"):
                                st.markdown(f"[Original page]({src['url']})")
                            # Local audio playback
                            local = src.get("local_audio")
                            if local and Path("data/full_corpus") / local:
                                audio_path = Path("data/full_corpus") / local
                                if audio_path.exists():
                                    st.audio(str(audio_path))
                            elif src.get("audio_url"):
                                st.markdown(f"[Play original audio]({src['audio_url']})")
                            st.caption(src["snippet"])

                if evidence:
                    with st.expander("External evidence / verification"):
                        st.markdown(evidence)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "evidence": evidence,
                    }
                )
            else:
                st.error("Setup incomplete. Check API key and vector store.")
