from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"
COLLECTION_NAME = "nykaa_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def search_policies(query, top_k=3):
    """Return the most relevant knowledge-base chunks for a query."""
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(DATABASE_FOLDER))
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for index, document in enumerate(results["documents"][0]):
        matches.append(
            {
                "document_id": results["metadatas"][0][index]["document_id"],
                "text": document,
                "distance": round(results["distances"][0][index], 4),
            }
        )

    return matches


if __name__ == "__main__":
    test_query = "How long do I have to return a beauty product?"

    print(f"Query: {test_query}\n")

    for number, match in enumerate(search_policies(test_query), start=1):
        print(f"{number}. Source: {match['document_id']}")
        print(f"   Distance: {match['distance']}")
        print(f"   {match['text'][:220]}\n")