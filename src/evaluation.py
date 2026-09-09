from retrieval import search_policies


TEST_CASES = [
    {
        "query": "What is the return window for beauty products?",
        "expected_document": "return_policy",
    },
    {
        "query": "How long does a COD refund take?",
        "expected_document": "cod_refund",
    },
    {
        "query": "What should I do if my payment failed?",
        "expected_document": "payment_failure",
    },
    {
        "query": "Can I exchange footwear for a different size?",
        "expected_document": "size_exchange",
    },
    {
        "query": "When should a delayed delivery be escalated?",
        "expected_document": "delivery_sla",
    },
]


def evaluate_retrieval():
    precision_scores = []
    recall_scores = []

    for case in TEST_CASES:
        results = search_policies(case["query"], top_k=3)
        retrieved_documents = [result["document_id"] for result in results]

        relevant_count = retrieved_documents.count(case["expected_document"])
        precision_at_3 = relevant_count / 3
        recall_at_3 = 1.0 if relevant_count > 0 else 0.0

        precision_scores.append(precision_at_3)
        recall_scores.append(recall_at_3)

        print(f"\nQuery: {case['query']}")
        print(f"Expected: {case['expected_document']}")
        print(f"Retrieved: {retrieved_documents}")
        print(f"Precision@3: {precision_at_3:.2f}")
        print(f"Recall@3: {recall_at_3:.2f}")

    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)

    print("\n--- Overall Evaluation ---")
    print(f"Average Precision@3: {average_precision:.2f}")
    print(f"Average Recall@3: {average_recall:.2f}")


if __name__ == "__main__":
    evaluate_retrieval()