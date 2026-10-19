"""
Index discourse transcripts into a FAISS vector store for RAG.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json

DATA_DIR = Path("data/samples")  # Change to data/full_corpus after scraping
VECTORSTORE_PATH = "data/vectorstore"

def load_documents():
    docs = []
    # Load .txt samples
    if DATA_DIR.exists():
        loader = DirectoryLoader(str(DATA_DIR), glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs.extend(loader.load())
    
    # Also load any .json from full_corpus if present
    json_dir = Path("data/full_corpus")
    if json_dir.exists():
        for jf in json_dir.glob("*.json"):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
                if data.get("transcript"):
                    from langchain.schema import Document
                    docs.append(Document(
                        page_content=data["transcript"],
                        metadata={
                            "source": data.get("url", str(jf)),
                            "title": data.get("title", ""),
                            "language": data.get("language", ""),
                            "audio": data.get("audio", "")
                        }
                    ))
    return docs

def main():
    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents found. Add samples or run scraper.")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")
    
    print("Creating embeddings (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to {VECTORSTORE_PATH}")

if __name__ == "__main__":
    main()
