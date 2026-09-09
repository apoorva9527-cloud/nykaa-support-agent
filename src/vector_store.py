from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import (
    load_documents,
    create_fixed_chunks,
    create_sentence_chunks,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"

FIXED_COLLECTION_NAME = "nykaa_fixed_chunks"
SENTENCE_COLLECTION_NAME = "nykaa_sentence_chunks"

MODEL_NAME = "all-MiniLM-L6-v2"


def create_collection(client, collection_name, chunks, model):
    """Create a ChromaDB collection and store embedded chunks."""

    try:
        client.delete_collection(collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(collection_name)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Creating embeddings for {collection_name}...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        metadatas=[
            {
                "source": chunk["source"],
                "strategy": chunk["strategy"],
            }
            for chunk in chunks
        ],
        embeddings=embeddings,
    )

    print(
        f"Stored {collection.count()} chunks "
        f"in '{collection_name}'."
    )

    return collection


def create_vector_store():
    """Create separate ChromaDB collections for both chunking strategies."""

    print("Loading knowledge-base documents...")

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    print("\nCreating fixed-size chunks...")
    fixed_chunks = create_fixed_chunks(documents)

    print(f"Fixed-size chunks: {len(fixed_chunks)}")

    print("\nCreating sentence-based chunks...")
    sentence_chunks = create_sentence_chunks(documents)

    print(f"Sentence-based chunks: {len(sentence_chunks)}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nCreating persistent ChromaDB client...")
    client = chromadb.PersistentClient(
        path=str(DATABASE_FOLDER)
    )

    fixed_collection = create_collection(
        client=client,
        collection_name=FIXED_COLLECTION_NAME,
        chunks=fixed_chunks,
        model=model,
    )

    sentence_collection = create_collection(
        client=client,
        collection_name=SENTENCE_COLLECTION_NAME,
        chunks=sentence_chunks,
        model=model,
    )

    print("\n----------------------------------------")
    print("Vector stores created successfully!")
    print("----------------------------------------")

    print(
        f"Fixed collection: "
        f"{fixed_collection.count()} chunks"
    )

    print(
        f"Sentence collection: "
        f"{sentence_collection.count()} chunks"
    )

    print(
        f"Database location: "
        f"{DATABASE_FOLDER}"
    )

    return fixed_collection, sentence_collection


if __name__ == "__main__":
    create_vector_store()