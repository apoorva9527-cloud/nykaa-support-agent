from retrieval import search_policies


TEST_CASES = [
    {
        "query": "What is the return window for beauty products?",
        "expected_document": "return_policy.md",
    },
    {
        "query": "How long does a COD refund take?",
        "expected_document": "cod_refund.md",
    },
    {
        "query": "What should I do if my payment failed?",
        "expected_document": "payment_failure.md",
    },
    {
        "query": "Can I exchange footwear for a different size?",
        "expected_document": "size_exchange.md",
    },
    {
        "query": "When should a delayed delivery be escalated?",
        "expected_document": "delivery_sla.md",
    },
]


OUT_OF_SCOPE_CASES = [
    "What is the weather in Delhi today?",
    "Who is the Prime Minister of India?",
    "How do I cook biryani?",
    "What is the capital of France?",
    "Tell me a joke about cricket.",
]


CANDIDATE_THRESHOLDS = [
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
]


def evaluate_strategy(strategy):
    """Evaluate one retrieval strategy using Precision@3 and Recall@3."""

    precision_scores = []
    recall_scores = []

    print("\n" + "=" * 60)
    print(f"EVALUATING: {strategy.upper()} CHUNKING")
    print("=" * 60)

    for case in TEST_CASES:

        results = search_policies(
            case["query"],
            top_k=3,
            strategy=strategy,
        )

        retrieved_documents = [
            result["source"]
            for result in results
        ]

        relevant_count = retrieved_documents.count(
            case["expected_document"]
        )

        precision_at_3 = relevant_count / 3

        recall_at_3 = (
            1.0
            if case["expected_document"] in retrieved_documents
            else 0.0
        )

        precision_scores.append(precision_at_3)
        recall_scores.append(recall_at_3)

        print(f"\nQuery: {case['query']}")
        print(f"Expected: {case['expected_document']}")
        print(f"Retrieved: {retrieved_documents}")
        print(f"Precision@3: {precision_at_3:.2f}")
        print(f"Recall@3: {recall_at_3:.2f}")

    average_precision = (
        sum(precision_scores) / len(precision_scores)
    )

    average_recall = (
        sum(recall_scores) / len(recall_scores)
    )

    print("\n--- Strategy Summary ---")
    print(f"Average Precision@3: {average_precision:.2f}")
    print(f"Average Recall@3: {average_recall:.2f}")

    return {
        "strategy": strategy,
        "precision_at_3": average_precision,
        "recall_at_3": average_recall,
    }


def collect_distance_data(strategy):
    """
    Collect distance values for relevant and out-of-scope queries.
    Lower Chroma distance means higher semantic similarity.
    """

    relevant_distances = []
    out_of_scope_distances = []

    print("\n" + "=" * 60)
    print(f"THRESHOLD DATA COLLECTION: {strategy.upper()}")
    print("=" * 60)

    for case in TEST_CASES:

        results = search_policies(
            case["query"],
            top_k=3,
            strategy=strategy,
        )

        matching_results = [
            result
            for result in results
            if result["source"] == case["expected_document"]
        ]

        if matching_results:
            best_match = min(
                matching_results,
                key=lambda result: result["distance"],
            )

            relevant_distances.append(
                best_match["distance"]
            )

            print(
                f"Relevant | "
                f"{case['expected_document']} | "
                f"distance={best_match['distance']:.4f}"
            )

    for query in OUT_OF_SCOPE_CASES:

        results = search_policies(
            query,
            top_k=1,
            strategy=strategy,
        )

        if results:
            distance = results[0]["distance"]
            out_of_scope_distances.append(distance)

            print(
                f"Out-of-scope | "
                f"distance={distance:.4f} | "
                f"query={query}"
            )

    return relevant_distances, out_of_scope_distances


def calibrate_threshold(strategy):
    """
    Empirically select a distance threshold.

    A result is considered grounded when:
        distance <= threshold

    The threshold is selected by maximizing balanced accuracy
    across relevant and out-of-scope examples.
    """

    relevant_distances, out_of_scope_distances = (
        collect_distance_data(strategy)
    )

    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION")
    print("=" * 60)

    print(
        f"\nRelevant examples: {len(relevant_distances)}"
    )

    print(
        f"Out-of-scope examples: "
        f"{len(out_of_scope_distances)}"
    )

    best_threshold = None
    best_score = -1

    for threshold in CANDIDATE_THRESHOLDS:

        true_positive = sum(
            distance <= threshold
            for distance in relevant_distances
        )

        false_negative = sum(
            distance > threshold
            for distance in relevant_distances
        )

        true_negative = sum(
            distance > threshold
            for distance in out_of_scope_distances
        )

        false_positive = sum(
            distance <= threshold
            for distance in out_of_scope_distances
        )

        total_relevant = (
            true_positive + false_negative
        )

        total_out_of_scope = (
            true_negative + false_positive
        )

        if total_relevant == 0 or total_out_of_scope == 0:
            continue

        true_positive_rate = (
            true_positive / total_relevant
        )

        true_negative_rate = (
            true_negative / total_out_of_scope
        )

        balanced_accuracy = (
            true_positive_rate + true_negative_rate
        ) / 2

        print(
            f"Threshold={threshold:.2f} | "
            f"TPR={true_positive_rate:.2f} | "
            f"TNR={true_negative_rate:.2f} | "
            f"Balanced Accuracy={balanced_accuracy:.2f}"
        )

        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_threshold = threshold

    print("\n--- Selected Threshold ---")
    print(f"Strategy: {strategy}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Balanced Accuracy: {best_score:.2f}")

    return best_threshold


def compare_strategies():
    """Compare both chunking strategies."""

    fixed_results = evaluate_strategy("fixed")
    sentence_results = evaluate_strategy("sentence")

    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 60)

    print(
        f"\nFixed-size | "
        f"Precision@3: "
        f"{fixed_results['precision_at_3']:.2f} | "
        f"Recall@3: "
        f"{fixed_results['recall_at_3']:.2f}"
    )

    print(
        f"Sentence   | "
        f"Precision@3: "
        f"{sentence_results['precision_at_3']:.2f} | "
        f"Recall@3: "
        f"{sentence_results['recall_at_3']:.2f}"
    )

    if (
        sentence_results["precision_at_3"]
        > fixed_results["precision_at_3"]
    ):
        selected_strategy = "sentence"
    else:
        selected_strategy = "fixed"

    print(
        f"\nSelected production strategy: "
        f"{selected_strategy}"
    )

    threshold = calibrate_threshold(
        selected_strategy
    )

    return selected_strategy, threshold


if __name__ == "__main__":
    compare_strategies()