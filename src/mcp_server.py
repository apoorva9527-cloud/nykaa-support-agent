import csv
from pathlib import Path

from mcp.server.fastmcp import FastMCP


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
ORDERS_FILE = PROJECT_FOLDER / "data" / "orders.csv"

mcp = FastMCP("Nykaa Order Support")


@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Get order status from the Nykaa order dataset."""

    normalized_order_id = order_id.strip().upper()

    with ORDERS_FILE.open(
        encoding="utf-8"
    ) as file:

        orders = csv.DictReader(file)

        for order in orders:

            if (
                order["order_id"].upper()
                == normalized_order_id
            ):
                return {
                    "found": True,
                    "order": order,
                }

    return {
        "found": False,
        "message": (
            f"No order found for ID: "
            f"{normalized_order_id}"
        ),
    }


if __name__ == "__main__":
    mcp.run()