"""
Index English and Hindi transcripts from data/english and data/hindi.
"""

import json
import os
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "data/vectorstore"


def load_documents():
    docs = []
    for lang in ["english", "hindi"]:
        lang_dir = Path("data") / lang
        if not lang_dir.exists():
            continue
        for jf in lang_dir.rglob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                text = (data.get("transcript") or "").strip()
                if len(text) < 200:
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "title": data.get("title", jf.stem),
                        "url": data.get("url", ""),
                        "language": data.get("language", lang),
                        "series": data.get("series", ""),
                        "duration": data.get("duration", ""),
                        "char_count": data.get("char_count", len(text)),
                        "source_file": str(jf),
                    }
                ))
            except Exception as e:
                print(f"Skip {jf}: {e}")
    return docs


def main():
    documents = load_documents()
    if not documents:
        print("No transcripts found in data/english or data/hindi. Run the scraper first.")
        return

    print(f"Loaded {len(documents)} discourses")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(documents)
    print(f"Chunks: {len(splits)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(splits, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vs.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()
