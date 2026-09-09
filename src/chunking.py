from pathlib import Path
import re


KB_DIR = Path("knowledge_base")


def load_documents():
    """
    Load all Markdown knowledge-base documents.
    """

    documents = []

    for file_path in sorted(KB_DIR.glob("*.md")):
        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:
            documents.append(
                {
                    "source": file_path.name,
                    "text": text,
                }
            )

    return documents


def fixed_size_chunking(
    text,
    chunk_size=500,
    overlap=100,
):
    """
    Original fixed-size chunking strategy.

    Kept for the Part 1 comparison requirement.
    """

    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be greater than overlap"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def sentence_chunking(
    text,
    sentences_per_chunk=3,
):
    """
    Sentence-based chunking strategy.

    Markdown tables are kept together with their
    surrounding heading so important structured
    information is not separated unnecessarily.
    """

    lines = [
        line.rstrip()
        for line in text.strip().splitlines()
    ]

    chunks = []
    current_block = []

    def flush_block():
        if current_block:
            block = "\n".join(current_block).strip()

            if block:
                chunks.append(block)

            current_block.clear()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # Skip completely empty lines.
        if not line:
            flush_block()
            i += 1
            continue

        # Markdown heading.
        if line.startswith("#"):

            flush_block()

            heading = line

            # Collect the heading and its immediate
            # following content.
            block = [heading]

            i += 1

            while i < len(lines):

                next_line = lines[i].strip()

                if not next_line:
                    break

                if next_line.startswith("#"):
                    break

                block.append(next_line)

                # Keep Markdown tables together.
                if next_line.startswith("|"):
                    i += 1

                    while i < len(lines):
                        table_line = lines[i].strip()

                        if table_line.startswith("|"):
                            block.append(table_line)
                            i += 1
                        else:
                            break

                    continue

                i += 1

            chunks.append(
                "\n".join(block).strip()
            )

            continue

        # Markdown table that appears without a heading.
        if line.startswith("|"):

            table_block = [line]

            i += 1

            while i < len(lines):

                table_line = lines[i].strip()

                if table_line.startswith("|"):
                    table_block.append(table_line)
                    i += 1
                else:
                    break

            chunks.append(
                "\n".join(table_block).strip()
            )

            continue

        current_block.append(line)

        # Convert normal prose into sentence groups.
        combined_text = " ".join(
            current_block
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            combined_text.strip(),
        )

        if len(sentences) >= sentences_per_chunk:

            for start in range(
                0,
                len(sentences),
                sentences_per_chunk,
            ):

                group = sentences[
                    start:start + sentences_per_chunk
                ]

                if group:
                    chunks.append(
                        " ".join(group).strip()
                    )

            current_block.clear()

        i += 1

    flush_block()

    return chunks


def create_fixed_chunks(documents):
    """
    Create fixed-size chunks with metadata.
    """

    all_chunks = []

    for document in documents:

        chunks = fixed_size_chunking(
            document["text"]
        )

        for index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "chunk_id": (
                        f"{document['source']}"
                        f"_fixed_{index}"
                    ),
                    "strategy": "fixed_size",
                }
            )

    return all_chunks


def create_sentence_chunks(documents):
    """
    Create structure-aware sentence chunks.

    Important Markdown tables remain intact.
    """

    all_chunks = []

    for document in documents:

        chunks = sentence_chunking(
            document["text"]
        )

        for index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "chunk_id": (
                        f"{document['source']}"
                        f"_sentence_{index}"
                    ),
                    "strategy": "sentence",
                }
            )

    return all_chunks


if __name__ == "__main__":

    documents = load_documents()

    fixed_chunks = create_fixed_chunks(
        documents
    )

    sentence_chunks = create_sentence_chunks(
        documents
    )

    print(
        f"Documents loaded: "
        f"{len(documents)}"
    )

    print(
        f"Fixed-size chunks: "
        f"{len(fixed_chunks)}"
    )

    print(
        f"Sentence/structure-aware chunks: "
        f"{len(sentence_chunks)}"
    )

    print("\n=== Sample chunks ===")

    for chunk in sentence_chunks[:5]:

        print("\n" + "-" * 60)

        print(
            f"Source: {chunk['source']}"
        )

        print(chunk["text"])