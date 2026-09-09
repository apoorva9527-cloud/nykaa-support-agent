from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import build_chunks


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"
COLLECTION_NAME = "nykaa_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def create_vector_store():
    """Create a persistent ChromaDB collection from recursive KB chunks."""
    chunks = build_chunks("recursive")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(DATABASE_FOLDER))

    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        metadatas=[
            {"document_id": chunk["document_id"]}
            for chunk in chunks
        ],
        embeddings=embeddings,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    return collection


if __name__ == "__main__":
    create_vector_store()