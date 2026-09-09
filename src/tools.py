import csv
from pathlib import Path

from retrieval import search_policies


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
ORDERS_FILE = PROJECT_FOLDER / "data" / "orders.csv"


def search_knowledge_base(query):
    """Search the policy knowledge base and return the top 3 matches."""
    return search_policies(query, top_k=3)


def check_order_status(order_id):
    """Look up one order from the synthetic orders dataset."""
    normalized_order_id = order_id.strip().upper()

    with ORDERS_FILE.open(encoding="utf-8") as file:
        orders = csv.DictReader(file)

        for order in orders:
            if order["order_id"].upper() == normalized_order_id:
                return {
                    "found": True,
                    "order": order,
                }

    return {
        "found": False,
        "message": f"No order found for ID: {normalized_order_id}",
    }


if __name__ == "__main__":
    print(check_order_status("ORD1001"))