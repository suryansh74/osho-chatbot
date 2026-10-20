import argparse
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

VECTORSTORE_PATH = "data/vectorstore"
BATCH_SIZE = 500


def load_documents(max_discourses: int | None = None):
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
                if max_discourses and len(docs) >= max_discourses:
                    return docs
            except Exception as e:
                print(f"Skip {jf}: {e}")
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=2500,
                        help="Larger = fewer chunks = much faster indexing (default 2500)")
    parser.add_argument("--max-discourses", type=int, default=None,
                        help="Limit number of discourses for a quick test")
    args = parser.parse_args()

    documents = load_documents(args.max_discourses)
    if not documents:
        print("No transcripts found.")
        return

    print(f"Loaded {len(documents)} discourses")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=200
    )
    splits = splitter.split_documents(documents)
    print(f"Total chunks: {len(splits)} (chunk_size={args.chunk_size})")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    vs = None
    for i in tqdm(range(0, len(splits), BATCH_SIZE), desc="Indexing batches"):
        batch = splits[i:i + BATCH_SIZE]
        if vs is None:
            vs = FAISS.from_documents(batch, embeddings)
        else:
            vs.add_documents(batch)
        if (i // BATCH_SIZE) % 5 == 0:
            vs.save_local(VECTORSTORE_PATH)

    if vs:
        vs.save_local(VECTORSTORE_PATH)
        print(f"\nVector store saved to {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()