import os
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
def load_vs():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return FAISS.load_local("data/vectorstore", emb, allow_dangerous_deserialization=True)
    except Exception:
        return None


@st.cache_resource
def get_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=key)


vectorstore = load_vs()
llm = get_llm()


class GraphState(TypedDict):
    question: str
    language: str
    documents: List[Document]
    answer: str
    references: List[dict]


def retrieve(state: GraphState):
    if not vectorstore:
        return {"documents": [], "references": []}
    # Filter by language if possible
    lang = state.get("language", "english")
    docs = vectorstore.similarity_search(state["question"], k=6)
    filtered = [d for d in docs if d.metadata.get("language", "").lower() == lang]
    if not filtered:
        filtered = docs  # fallback
    refs = []
    for d in filtered[:4]:
        refs.append({
            "title": d.metadata.get("title", ""),
            "url": d.metadata.get("url", ""),
            "series": d.metadata.get("series", ""),
            "excerpt": d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""),
            "language": d.metadata.get("language", ""),
        })
    return {"documents": filtered[:4], "references": refs}


def generate(state: GraphState):
    """Faithful, readable answer. Strict instruction not to add external psychology or alter essence."""
    context = "\n\n---\n\n".join(
        f"Discourse: {d.metadata.get('title','')}\nURL: {d.metadata.get('url','')}\n{d.page_content}"
        for d in state["documents"]
    )
    prompt = ChatPromptTemplate.from_template("""
You are an Osho discourse assistant. Your only job is to present Osho's words clearly.

STRICT RULES:
- Use ONLY the text given in the Context below.
- Do NOT add your own interpretations, psychology, modern explanations, or extra ideas.
- Do NOT soften, expand, or "improve" the meaning. Preserve the original essence exactly.
- Make the answer easy to read (short paragraphs, clear sentences) but stay faithful.
- If the question spans multiple discourses, weave the relevant passages together without changing them.
- Always name the discourse(s) you are drawing from.

Context:
{context}

User question ({language}):
{question}

Readable answer (faithful to the original text only):
""")
    chain = prompt | llm
    resp = chain.invoke({
        "context": context,
        "question": state["question"],
        "language": state.get("language", "english"),
    })
    return {"answer": resp.content}


workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
graph = workflow.compile()

# ---------- UI ----------
st.title("🕉️ Osho Chatbot")
st.caption("Faithful answers from original Osho discourse transcripts (English & Hindi)")

lang = st.selectbox("Language", ["english", "hindi"], index=0)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("references"):
            with st.expander("Original transcript references"):
                for r in m["references"]:
                    st.markdown(f"**{r['title']}**  \nSeries: {r.get('series','')}  \n[Link]({r['url']})")
                    st.text(r["excerpt"])
                    st.divider()

if q := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        if not vectorstore or not llm:
            st.error("Vector store or GROQ_API_KEY missing. Run scraper + index first.")
        else:
            with st.spinner("Retrieving..."):
                result = graph.invoke({"question": q, "language": lang})
                answer = result.get("answer", "")
                refs = result.get("references", [])

                st.markdown("**Answer**")
                st.markdown(answer)

                if refs:
                    with st.expander("Original transcript references"):
                        for r in refs:
                            st.markdown(f"**{r['title']}**")
                            if r.get("url"):
                                st.markdown(f"[Open original discourse]({r['url']})")
                            st.text(r["excerpt"])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "references": refs,
                })
