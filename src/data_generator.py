import csv
import random
from pathlib import Path

random.seed(42)

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]

DAYS_BY_STATUS = {
    "Placed": (0, 2),
    "Shipped": (2, 7),
    "Delivered": (3, 30),
    "Returned": (10, 40),
    "Refunded": (12, 45),
}


def generate_orders(count=50):
    orders = []

    for number in range(1, count + 1):
        status = random.choice(STATUSES)
        min_days, max_days = DAYS_BY_STATUS[status]

        orders.append(
            {
                "order_id": f"ORD{1000 + number}",
                "customer_id": f"CUST{200 + number}",
                "category": random.choice(CATEGORIES),
                "status": status,
                "order_value_inr": random.randint(299, 9999),
                "days_since_created": random.randint(min_days, max_days),
                "delayed_shipment": (
                    "Yes"
                    if status == "Shipped" and random.random() < 0.35
                    else "No"
                ),
            }
        )

    return orders


def save_orders(orders):
    project_folder = Path(__file__).resolve().parents[1]
    output_file = project_folder / "data" / "orders.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=orders[0].keys())
        writer.writeheader()
        writer.writerows(orders)

    print(f"Created {len(orders)} orders in: {output_file}")


if __name__ == "__main__":
    save_orders(generate_orders())