"""
Index transcripts (from website scrape and/or local audio) into FAISS.
"""

import json
import os
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Prefer local audio transcripts if present, otherwise fall back to website scrape
CANDIDATE_DIRS = [
    Path("data/local_corpus"),
    Path("data/full_corpus/transcripts"),
    Path("data/samples"),
]
VECTORSTORE_PATH = "data/vectorstore"
METADATA_INDEX = "data/metadata_index.json"


def load_all_documents():
    docs = []
    for corpus_dir in CANDIDATE_DIRS:
        if not corpus_dir.exists():
            continue
        print(f"Loading from {corpus_dir}...")
        for json_file in corpus_dir.rglob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                transcript = (data.get("transcript") or "").strip()
                if len(transcript) < 80:
                    continue
                metadata = {
                    "source": data.get("url") or data.get("source") or str(json_file),
                    "title": data.get("title") or json_file.stem,
                    "language": data.get("language", ""),
                    "audio_url": data.get("audio_url", ""),
                    "local_audio": data.get("local_audio", ""),
                    "relative_audio": data.get("relative_audio", ""),
                    "file": str(json_file),
                }
                docs.append(Document(page_content=transcript, metadata=metadata))
            except Exception as e:
                print(f"Skipping {json_file}: {e}")
        # Also plain text samples
        for txt_file in corpus_dir.rglob("*.txt"):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                if len(text) > 80:
                    docs.append(Document(
                        page_content=text,
                        metadata={"title": txt_file.stem, "source": str(txt_file), "local_audio": ""}
                    ))
            except Exception:
                pass
    return docs


def main():
    documents = load_all_documents()
    if not documents:
        print("No documents found. Run the scraper or process_local_audio.py first.")
        return

    print(f"Loaded {len(documents)} documents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to {VECTORSTORE_PATH}")

    meta = []
    for doc in documents:
        meta.append({
            "title": doc.metadata.get("title"),
            "url": doc.metadata.get("source"),
            "local_audio": doc.metadata.get("local_audio"),
            "relative_audio": doc.metadata.get("relative_audio"),
            "language": doc.metadata.get("language"),
        })
    with open(METADATA_INDEX, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Metadata index saved to {METADATA_INDEX}")


if __name__ == "__main__":
    main()
