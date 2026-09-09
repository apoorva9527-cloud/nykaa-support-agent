import json
from pathlib import Path

from agent import answer_query
from retrieval import search_policies


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_FOLDER / "logs" / "rag_triad_results.json"


TEST_CASES = [
    {
        "id": "Q01",
        "query": "What is the return window for beauty products?",
        "expected_source": "return_policy.md",
        "expected_keywords": ["7 days", "unopened", "unused"],
    },
    {
        "id": "Q02",
        "query": "How long does a COD refund take?",
        "expected_source": "cod_refund.md",
        "expected_keywords": ["5 to 7 business days", "quality check"],
    },
    {
        "id": "Q03",
        "query": "What should I do if my payment failed?",
        "expected_source": "payment_failure.md",
        "expected_keywords": ["retry", "payment"],
    },
    {
        "id": "Q04",
        "query": "Can I exchange footwear after buying it?",
        "expected_source": "size_exchange.md",
        "expected_keywords": ["footwear", "exchange"],
    },
    {
        "id": "Q05",
        "query": "What is the delivery SLA?",
        "expected_source": "delivery_sla.md",
        "expected_keywords": ["delivery"],
    },
    {
        "id": "Q06",
        "query": "Can I cancel my order?",
        "expected_source": "cancellation.md",
        "expected_keywords": ["cancel"],
    },
    {
        "id": "Q07",
        "query": "How does reverse pickup work?",
        "expected_source": "reverse_pickup.md",
        "expected_keywords": ["reverse pickup"],
    },
    {
        "id": "Q08",
        "query": "What should I do if my item arrived damaged?",
        "expected_source": "damaged_item.md",
        "expected_keywords": ["damaged"],
    },
    {
        "id": "Q09",
        "query": "What is the warranty policy?",
        "expected_source": "warranty.md",
        "expected_keywords": ["warranty"],
    },
    {
        "id": "Q10",
        "query": "How can I use my loyalty points?",
        "expected_source": "loyalty_points.md",
        "expected_keywords": ["loyalty", "points"],
    },
    {
        "id": "Q11",
        "query": "Can I exchange an apparel item for another size?",
        "expected_source": "size_exchange.md",
        "expected_keywords": ["apparel", "exchange", "size"],
    },
    {
        "id": "Q12",
        "query": "Can I return an electronics product?",
        "expected_source": "return_policy.md",
        "expected_keywords": ["electronics", "7 days"],
    },
    {
        "id": "Q13",
        "query": "How long does an international shipment take?",
        "expected_source": "international_shipping.md",
        "expected_keywords": ["international", "shipping"],
    },
    {
        "id": "Q14",
        "query": "When should a customer support issue be escalated?",
        "expected_source": "escalation_matrix.md",
        "expected_keywords": ["escalation"],
    },
    {
        "id": "Q15",
        "query": "What should I do if my COD refund is delayed?",
        "expected_source": "cod_refund.md",
        "expected_keywords": ["refund", "5 to 7 business days"],
    },
]


def normalize(text):
    return " ".join(
        text.lower().split()
    )


def calculate_context_relevance(test_case):
    results = search_policies(
        test_case["query"],
        top_k=3,
        strategy="sentence",
        apply_threshold=False,
    )

    if not results:
        return 0.0, results

    expected_source = test_case["expected_source"]

    source_match = any(
        result["source"] == expected_source
        for result in results
    )

    if source_match:
        return 1.0, results

    return 0.0, results


def calculate_answer_relevance(
    test_case,
    answer,
):
    answer_lower = normalize(answer)

    keywords = test_case["expected_keywords"]

    matched = sum(
        1
        for keyword in keywords
        if normalize(keyword) in answer_lower
    )

    if not keywords:
        return 0.0

    return round(
        matched / len(keywords),
        2,
    )


def calculate_groundedness(
    test_case,
    response,
):
    expected_source = test_case["expected_source"]

    sources = response.get(
        "sources",
        [],
    )

    grounded = response.get(
        "grounded",
        False,
    )

    if (
        grounded
        and expected_source in sources
    ):
        return 1.0

    return 0.0


def run_evaluation():

    results = []

    print("\n" + "=" * 80)
    print("NYKAA RAG TRIAD EVALUATION")
    print("=" * 80)

    for test_case in TEST_CASES:

        query = test_case["query"]

        print("\n" + "-" * 80)
        print(
            f"{test_case['id']}: {query}"
        )

        context_score, retrieved = (
            calculate_context_relevance(
                test_case
            )
        )

        response = answer_query(query)

        answer_score = (
            calculate_answer_relevance(
                test_case,
                response["answer"],
            )
        )

        groundedness_score = (
            calculate_groundedness(
                test_case,
                response,
            )
        )

        result = {
            "id": test_case["id"],
            "query": query,
            "expected_source": test_case[
                "expected_source"
            ],
            "retrieved_sources": [
                item["source"]
                for item in retrieved
            ],
            "context_relevance": context_score,
            "answer_relevance": answer_score,
            "groundedness": groundedness_score,
            "overall_score": round(
                (
                    context_score
                    + answer_score
                    + groundedness_score
                )
                / 3,
                2,
            ),
        }

        results.append(result)

        print(
            f"Context Relevance : "
            f"{context_score:.2f}"
        )

        print(
            f"Answer Relevance  : "
            f"{answer_score:.2f}"
        )

        print(
            f"Groundedness      : "
            f"{groundedness_score:.2f}"
        )

        print(
            f"Overall Score     : "
            f"{result['overall_score']:.2f}"
        )

    total = len(results)

    avg_context = round(
        sum(
            item["context_relevance"]
            for item in results
        )
        / total,
        2,
    )

    avg_answer = round(
        sum(
            item["answer_relevance"]
            for item in results
        )
        / total,
        2,
    )

    avg_groundedness = round(
        sum(
            item["groundedness"]
            for item in results
        )
        / total,
        2,
    )

    avg_overall = round(
        sum(
            item["overall_score"]
            for item in results
        )
        / total,
        2,
    )

    summary = {
        "total_queries": total,
        "average_context_relevance": avg_context,
        "average_answer_relevance": avg_answer,
        "average_groundedness": avg_groundedness,
        "average_overall_score": avg_overall,
    }

    output = {
        "evaluation": "RAG Triad",
        "strategy": "sentence",
        "results": results,
        "summary": summary,
    }

    OUTPUT_FILE.parent.mkdir(
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("RAG TRIAD SUMMARY")
    print("=" * 80)

    print(
        f"Queries evaluated : {total}"
    )

    print(
        f"Average Context Relevance : "
        f"{avg_context:.2f}"
    )

    print(
        f"Average Answer Relevance  : "
        f"{avg_answer:.2f}"
    )

    print(
        f"Average Groundedness      : "
        f"{avg_groundedness:.2f}"
    )

    print(
        f"Average Overall Score     : "
        f"{avg_overall:.2f}"
    )

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    run_evaluation()