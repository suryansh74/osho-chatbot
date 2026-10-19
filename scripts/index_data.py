"""
Index all scraped transcripts into a FAISS vector store.
Also builds a simple metadata index for audio lookup.
"""

import json
import os
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CORPUS_DIR = Path("data/full_corpus/transcripts")
VECTORSTORE_PATH = "data/vectorstore"
METADATA_INDEX = "data/metadata_index.json"


def load_all_documents():
    docs = []
    if not CORPUS_DIR.exists():
        print(f"Corpus directory not found: {CORPUS_DIR}")
        print("Run the scraper first.")
        return docs

    for json_file in CORPUS_DIR.rglob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            transcript = data.get("transcript", "").strip()
            if not transcript or len(transcript) < 100:
                continue

            metadata = {
                "source": data.get("url", ""),
                "title": data.get("title", json_file.stem),
                "language": data.get("language", ""),
                "audio_url": data.get("audio_url", ""),
                "local_audio": data.get("local_audio", ""),
                "file": str(json_file),
            }
            docs.append(Document(page_content=transcript, metadata=metadata))
        except Exception as e:
            print(f"Skipping {json_file}: {e}")

    return docs


def main():
    print("Loading documents from corpus...")
    documents = load_all_documents()
    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} discourses")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")

    print("Creating embeddings (this can take a while on large corpora)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(splits, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to {VECTORSTORE_PATH}")

    # Save a lightweight metadata index for audio lookup
    meta = []
    for doc in documents:
        meta.append({
            "title": doc.metadata.get("title"),
            "url": doc.metadata.get("source"),
            "local_audio": doc.metadata.get("local_audio"),
            "audio_url": doc.metadata.get("audio_url"),
            "language": doc.metadata.get("language"),
        })
    with open(METADATA_INDEX, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Metadata index saved to {METADATA_INDEX}")


if __name__ == "__main__":
    main()
