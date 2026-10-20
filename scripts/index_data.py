"""
Index English and Hindi transcripts with progress and incremental saving.
"""

import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

VECTORSTORE_PATH = "data/vectorstore"
BATCH_SIZE = 500  # embed and add this many chunks at a time


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
        print("No transcripts found in data/english or data/hindi.")
        return

    print(f"Loaded {len(documents)} discourses")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(documents)
    print(f"Total chunks: {len(splits)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    # Build in batches so progress is visible and index is saved incrementally
    vs = None
    for i in tqdm(range(0, len(splits), BATCH_SIZE), desc="Indexing batches"):
        batch = splits[i:i + BATCH_SIZE]
        if vs is None:
            vs = FAISS.from_documents(batch, embeddings)
        else:
            vs.add_documents(batch)
        # Save after every few batches
        if (i // BATCH_SIZE) % 5 == 0:
            vs.save_local(VECTORSTORE_PATH)

    if vs:
        vs.save_local(VECTORSTORE_PATH)
        print(f"\nVector store saved to {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()
