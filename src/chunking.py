import re
from pathlib import Path


def load_documents():
    """Load every Markdown policy document from the knowledge base."""
    knowledge_base = Path(__file__).resolve().parents[1] / "knowledge_base"
    documents = []

    for file_path in sorted(knowledge_base.glob("*.md")):
        documents.append(
            {
                "document_id": file_path.stem,
                "text": file_path.read_text(encoding="utf-8"),
            }
        )

    return documents


def fixed_size_chunks(text, chunk_size=300, overlap=50):
    """Split text by character count, keeping a small overlap."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap

    return [chunk for chunk in chunks if chunk]


def recursive_chunks(text, chunk_size=300):
    """Split text first by paragraphs, then by sentences if needed."""
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk += paragraph + "\n"
            continue

        if current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        if len(paragraph) <= chunk_size:
            current_chunk = paragraph + "\n"
        else:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            sentence_chunk = ""

            for sentence in sentences:
                if len(sentence_chunk) + len(sentence) + 1 <= chunk_size:
                    sentence_chunk += sentence + " "
                else:
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
                    sentence_chunk = sentence + " "

            if sentence_chunk:
                current_chunk = sentence_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def build_chunks(strategy="recursive"):
    """Create chunks from all knowledge-base documents."""
    all_chunks = []

    for document in load_documents():
        if strategy == "fixed":
            text_chunks = fixed_size_chunks(document["text"])
        else:
            text_chunks = recursive_chunks(document["text"])

        for index, chunk in enumerate(text_chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{document['document_id']}_{index}",
                    "document_id": document["document_id"],
                    "text": chunk,
                }
            )

    return all_chunks


if __name__ == "__main__":
    fixed = build_chunks("fixed")
    recursive = build_chunks("recursive")

    print(f"Fixed-size chunks: {len(fixed)}")
    print(f"Recursive chunks: {len(recursive)}")