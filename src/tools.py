import csv
from pathlib import Path

try:
    from src.retrieval import search_policies
except ModuleNotFoundError:
    from src.retrieval import search_policies   from retrieval import search_policies


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
ORDERS_FILE = PROJECT_FOLDER / "data" / "orders.csv"


def search_knowledge_base(query):
    """Search the policy knowledge base and return the top 3 matches."""
    return search_policies(query, top_k=3)


def calculate_escalation_score(order):
    """
    Calculate a normalized escalation score in the range [0, 1].

    Formula:
        escalation_score =
            0.6 * delayed_shipment
            + 0.4 * normalized_recency

    normalized_recency is days_since_created / 30.
    """

    delayed_signal = (
        1.0
        if str(order["delayed_shipment"]).strip().lower() == "yes"
        else 0.0
    )

    days_since_created = int(order["days_since_created"])
    normalized_recency = min(max(days_since_created / 30.0, 0.0), 1.0)

    score = (
        0.6 * delayed_signal
        + 0.4 * normalized_recency
    )

    return round(score, 4)


def check_order_status(order_id: str) -> dict:
    """
    Look up one order and calculate its escalation score.

    The escalation score combines delayed shipment with a
    normalized recency signal derived from days_since_created.
    """

    normalized_order_id = order_id.strip().upper()

    with ORDERS_FILE.open(encoding="utf-8") as file:
        orders = csv.DictReader(file)

        for order in orders:
            if order["order_id"].upper() == normalized_order_id:

                escalation_score = calculate_escalation_score(order)

                return {
                    "found": True,
                    "order": order,
                    "status": order["status"],
                    "order_value_inr": float(order["order_value_inr"]),
                    "escalation_score": escalation_score,
                }

    return {
        "found": False,
        "message": f"No order found for ID: {normalized_order_id}",
        "escalation_score": 0.0,
    }


if __name__ == "__main__":
    print(check_order_status("ORD1001"))