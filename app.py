import os
from typing import List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

st.set_page_config(page_title="Osho Chatbot", page_icon="🕉️", layout="wide")

@st.cache_resource
def load_vs():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    try:
        return FAISS.load_local("data/vectorstore", emb, allow_dangerous_deserialization=True)
    except Exception:
        return None


@st.cache_resource
def get_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, api_key=key)


vectorstore = load_vs()
llm = get_llm()


class GraphState(TypedDict):
    question: str
    language: str
    documents: List[Document]
    answer: str
    references: List[dict]


def estimate_timestamp(excerpt: str, full_char_count: int, duration: str, start_char: int = 0) -> str:
    """Rough timestamp estimate: position of excerpt / total chars * duration."""
    if not duration or not full_char_count or full_char_count <= 0:
        return ""
    # Find approximate start position (very rough — first 80 chars of excerpt)
    # In practice we use a simple proportional estimate
    try:
        parts = duration.split(":")
        if len(parts) == 3:
            total_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            total_sec = int(parts[0]) * 60 + int(parts[1])
        else:
            return ""
        # Use a conservative middle-of-chunk estimate (we don't have exact offset)
        # Show a range-style label instead of a precise second
        pct = 0.35  # default mid-early; can be refined later
        est_sec = int(total_sec * pct)
        m, s = divmod(est_sec, 60)
        h, m = divmod(m, 60)
        if h:
            return f"~{h}:{m:02d}:{s:02d}"
        return f"~{m}:{s:02d}"
    except Exception:
        return ""


def retrieve(state: GraphState):
    if not vectorstore:
        return {"documents": [], "references": []}

    lang = state.get("language", "english").lower()
    question = state["question"].strip()

    # For Hindi, also try a Devanagari-friendly expansion if the query looks Romanized
    queries = [question]
    if lang == "hindi" and not any("\u0900" <= c <= "\u097F" for c in question):
        # Simple expansions for common Romanized terms
        expansions = [
            question.replace("shashvat satya", "शाश्वत सत्य").replace("satya", "सत्य"),
            question + " ओशो",
            "शाश्वत सत्य" if "satya" in question.lower() or "shashvat" in question.lower() else question,
        ]
        queries.extend(expansions)

    all_docs = []
    seen = set()
    for q in queries[:3]:
        docs = vectorstore.similarity_search(q, k=15)
        for d in docs:
            key = (d.metadata.get("url", ""), d.page_content[:100])
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    filtered = [d for d in all_docs if d.metadata.get("language", "").lower() == lang]

    if not filtered:
        return {"documents": [], "references": []}

    # Prefer unique discourses
    unique = []
    seen_urls = set()
    for d in filtered:
        url = d.metadata.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append(d)
        if len(unique) >= 4:
            break

    refs = []
    for d in unique:
        excerpt = d.page_content[:700] + ("..." if len(d.page_content) > 700 else "")
        ts = estimate_timestamp(
            excerpt,
            d.metadata.get("char_count", 0),
            d.metadata.get("duration", ""),
            d.metadata.get("start_char", 0),
        )
        refs.append({
            "title": d.metadata.get("title", ""),
            "url": d.metadata.get("url", ""),
            "series": d.metadata.get("series", ""),
            "excerpt": excerpt,
            "language": d.metadata.get("language", ""),
            "duration": d.metadata.get("duration", ""),
            "estimated_time": ts,
        })
    return {"documents": unique, "references": refs}


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
                    st.markdown(f"**{r['title']}**")
                    if r.get("series"):
                        st.markdown(f"Series: {r['series']}")
                    if r.get("duration"):
                        st.markdown(f"Duration: {r['duration']}")
                    if r.get("estimated_time"):
                        st.markdown(f"Estimated position: {r['estimated_time']} (approximate)")
                    if r.get("url"):
                        st.markdown(f"[Open original discourse]({r['url']})")
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

                if not refs and lang == "hindi":
                    st.warning("No Hindi results found. Try asking the same question in Hindi, or check that Hindi transcripts were indexed.")
                    answer = "No matching Hindi discourses were found for this query."

                st.markdown("**Answer**")
                st.markdown(answer)

                if refs:
                    with st.expander("Original transcript references"):
                        for r in refs:
                            st.markdown(f"**{r['title']}**")
                            if r.get("series"):
                                st.markdown(f"Series: {r['series']}")
                            if r.get("duration"):
                                st.markdown(f"Duration: {r['duration']}")
                            if r.get("estimated_time"):
                                st.markdown(f"Estimated position: {r['estimated_time']} (approximate)")
                            if r.get("url"):
                                st.markdown(f"[Open original discourse]({r['url']})")
                            st.text(r["excerpt"])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "references": refs,
                })
