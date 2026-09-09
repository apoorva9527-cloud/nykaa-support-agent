from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"

FIXED_COLLECTION_NAME = "nykaa_fixed_chunks"
SENTENCE_COLLECTION_NAME = "nykaa_sentence_chunks"

MODEL_NAME = "all-MiniLM-L6-v2"

# Calibrated during Part 1 evaluation.
GROUNDEDNESS_THRESHOLD = 0.90


CATEGORIES = {
    "beauty": [
        "beauty",
        "cosmetic",
        "cosmetics",
        "makeup",
        "skincare",
        "skin care",
    ],
    "apparel": [
        "apparel",
        "clothing",
        "dress",
        "shirt",
        "tshirt",
        "t shirt",
        "clothes",
    ],
    "footwear": [
        "footwear",
        "shoe",
        "shoes",
        "sandal",
        "sandals",
        "sneaker",
        "sneakers",
    ],
    "home": [
        "home",
        "household",
        "furniture",
        "kitchen",
    ],
    "electronics": [
        "electronics",
        "electronic",
        "laptop",
        "phone",
        "mobile",
        "headphone",
        "headphones",
    ],
}


POLICY_INTENTS = {
    "return_window": [
        "return window",
        "return period",
        "how many days",
        "how long can i return",
        "return",
        "returns",
    ],
    "refund": [
        "refund",
        "refunded",
        "money back",
    ],
    "exchange": [
        "exchange",
        "size exchange",
        "different size",
        "replace size",
    ],
    "delivery": [
        "delivery",
        "deliver",
        "shipment",
        "shipping",
        "delayed",
        "delay",
    ],
    "payment": [
        "payment",
        "pay",
        "paid",
        "payment failed",
        "transaction",
    ],
    "cancellation": [
        "cancel",
        "cancellation",
    ],
    "warranty": [
        "warranty",
        "warranties",
    ],
    "loyalty": [
        "loyalty",
        "points",
        "reward points",
    ],
    "damage": [
        "damaged",
        "damage",
        "broken",
        "incorrect item",
        "wrong item",
    ],
    "cod": [
        "cod",
        "cash on delivery",
    ],
    "international": [
        "international",
        "international shipping",
        "outside india",
    ],
    "escalation": [
        "escalate",
        "escalation",
        "support",
        "complaint",
    ],
}


POLICY_DOCUMENTS = {
    "return_window": "return_policy.md",
    "refund": "cod_refund.md",
    "exchange": "size_exchange.md",
    "delivery": "delivery_sla.md",
    "payment": "payment_failure.md",
    "cancellation": "cancellation.md",
    "warranty": "warranty.md",
    "loyalty": "loyalty_points.md",
    "damage": "damaged_item.md",
    "cod": "cod_refund.md",
    "international": "international_shipping.md",
    "escalation": "escalation_matrix.md",
}


def get_collection(strategy="fixed"):
    """Get the ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=str(DATABASE_FOLDER)
    )

    if strategy == "fixed":
        collection_name = FIXED_COLLECTION_NAME
    elif strategy == "sentence":
        collection_name = SENTENCE_COLLECTION_NAME
    else:
        raise ValueError(
            "strategy must be either 'fixed' or 'sentence'"
        )

    return client.get_collection(collection_name)


def normalize_text(text):
    """Normalize text for lexical matching."""

    return re.sub(
        r"[^a-z0-9\s]",
        " ",
        text.lower(),
    )


def detect_categories(query):
    """Detect product categories in a query."""

    query_text = normalize_text(query)

    detected = []

    for category, keywords in CATEGORIES.items():

        if any(
            keyword in query_text
            for keyword in keywords
        ):
            detected.append(category)

    return detected


def detect_intents(query):
    """Detect policy intent in a query."""

    query_text = normalize_text(query)

    detected = []

    # Check specific intents first.
    for intent, keywords in POLICY_INTENTS.items():

        if any(
            keyword in query_text
            for keyword in keywords
        ):
            detected.append(intent)

    # "return window" is specifically a return-window
    # question, not a generic return question.
    if (
        "return window" in query_text
        or "return period" in query_text
        or "how many days" in query_text
    ):
        if "return_window" not in detected:
            detected.insert(0, "return_window")

    return detected


def keyword_score(query, document):
    """Calculate meaningful lexical overlap."""

    query_words = set(
        normalize_text(query).split()
    )

    document_words = set(
        normalize_text(document).split()
    )

    stop_words = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "for",
        "to",
        "of",
        "do",
        "i",
        "can",
        "how",
        "long",
        "does",
        "my",
        "me",
        "in",
        "on",
        "and",
        "or",
        "are",
        "be",
        "with",
        "if",
        "should",
        "when",
        "will",
    }

    meaningful_words = (
        query_words - stop_words
    )

    if not meaningful_words:
        return 0.0

    matched_words = (
        meaningful_words & document_words
    )

    return (
        len(matched_words)
        / len(meaningful_words)
    )


def category_score(query, document):
    """Calculate category match score."""

    categories = detect_categories(query)

    if not categories:
        return 0.0

    document_text = normalize_text(
        document
    )

    matches = 0

    for category in categories:

        if any(
            keyword in document_text
            for keyword in CATEGORIES[category]
        ):
            matches += 1

    return matches / len(categories)


def intent_score(query, document):
    """Calculate policy-intent match score."""

    intents = detect_intents(query)

    if not intents:
        return 0.0

    document_text = normalize_text(
        document
    )

    matches = 0

    for intent in intents:

        if any(
            keyword in document_text
            for keyword in POLICY_INTENTS[intent]
        ):
            matches += 1

    return matches / len(intents)


def document_priority(query, source, document):
    """
    Apply a deterministic priority to the policy document
    that owns the requested policy.

    This prevents a generic return-related chunk from
    outranking the actual return-policy table.
    """

    intents = detect_intents(query)
    categories = detect_categories(query)

    priority = 0.0

    for intent in intents:

        expected_source = POLICY_DOCUMENTS.get(
            intent
        )

        if (
            expected_source
            and source == expected_source
        ):
            priority += 1.0

    # Category-specific return-window questions need
    # the actual return-policy document.
    if (
        "return_window" in intents
        and categories
        and source == "return_policy.md"
    ):
        priority += 2.0

    # For a return-window question, a chunk containing
    # the category table is especially valuable.
    if (
        "return_window" in intents
        and categories
        and source == "return_policy.md"
        and "|" in document
        and any(
            category in normalize_text(document)
            for category in categories
        )
    ):
        priority += 3.0

    return priority


def calculate_relevance_score(
    query,
    source,
    document,
    distance,
):
    """
    Calculate the final hybrid relevance score.
    """

    semantic_score = 1.0 / (
        1.0 + max(distance, 0.0)
    )

    lexical = keyword_score(
        query,
        document,
    )

    category = category_score(
        query,
        document,
    )

    intent = intent_score(
        query,
        document,
    )

    priority = document_priority(
        query,
        source,
        document,
    )

    score = (
        0.30 * semantic_score
        + 0.15 * lexical
        + 0.15 * category
        + 0.15 * intent
        + 0.25 * priority
    )

    return round(
        score,
        4,
    )


def is_relevant_candidate(
    query,
    source,
    document,
    distance,
):
    """
    Determine whether a result is sufficiently grounded.

    Normal semantic results must satisfy the calibrated
    threshold.

    Strong policy-document matches can also be retained
    when semantic distance is higher.
    """

    if distance <= GROUNDEDNESS_THRESHOLD:
        return True

    priority = document_priority(
        query,
        source,
        document,
    )

    return priority >= 3.0


def search_policies(
    query,
    top_k=3,
    strategy="sentence",
    apply_threshold=True,
):
    """
    Hybrid policy retrieval using:

    1. Semantic similarity
    2. Keyword overlap
    3. Category matching
    4. Policy-intent matching
    5. Policy-document priority
    """

    model = SentenceTransformer(
        MODEL_NAME
    )

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    collection = get_collection(
        strategy
    )

    candidate_k = max(
        top_k * 5,
        20,
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for index, document in enumerate(
        results["documents"][0]
    ):

        distance = results[
            "distances"
        ][0][index]

        source = results[
            "metadatas"
        ][0][index]["source"]

        lexical = keyword_score(
            query,
            document,
        )

        category = category_score(
            query,
            document,
        )

        intent = intent_score(
            query,
            document,
        )

        priority = document_priority(
            query,
            source,
            document,
        )

        relevance = calculate_relevance_score(
            query,
            source,
            document,
            distance,
        )

        matches.append(
            {
                "source": source,
                "strategy": results[
                    "metadatas"
                ][0][index]["strategy"],
                "text": document,
                "distance": round(
                    distance,
                    4,
                ),
                "keyword_score": round(
                    lexical,
                    4,
                ),
                "category_score": round(
                    category,
                    4,
                ),
                "intent_score": round(
                    intent,
                    4,
                ),
                "document_priority": round(
                    priority,
                    4,
                ),
                "relevance_score": relevance,
            }
        )

    if apply_threshold:

        grounded_matches = [
            match
            for match in matches
            if is_relevant_candidate(
                query,
                match["source"],
                match["text"],
                match["distance"],
            )
        ]

    else:
        grounded_matches = matches

    grounded_matches.sort(
        key=lambda match: (
            -match["document_priority"],
            -match["relevance_score"],
            match["distance"],
        )
    )

    return grounded_matches[:top_k]


if __name__ == "__main__":

    test_queries = [
        "What is the return window for beauty products?",
        "How long does a COD refund take?",
        "What should I do if my payment failed?",
        "What is the weather in Delhi today?",
    ]

    for query in test_queries:

        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        results = search_policies(
            query,
            top_k=3,
            strategy="sentence",
            apply_threshold=True,
        )

        if not results:
            print("\nNo grounded results found.")

        for number, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n{number}. "
                f"{result['source']}"
            )

            print(
                f"Distance: "
                f"{result['distance']}"
            )

            print(
                f"Keyword score: "
                f"{result['keyword_score']}"
            )

            print(
                f"Category score: "
                f"{result['category_score']}"
            )

            print(
                f"Intent score: "
                f"{result['intent_score']}"
            )

            print(
                f"Document priority: "
                f"{result['document_priority']}"
            )

            print(
                f"Relevance score: "
                f"{result['relevance_score']}"
            )

            print(
                f"Text: "
                f"{result['text'][:400]}"
            )