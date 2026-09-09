import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
SERVER_FILE = PROJECT_FOLDER / "src" / "mcp_server.py"


async def check_order(order_id: str):

    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_FILE)],
    )

    async with stdio_client(
        server_parameters
    ) as (
        read_stream,
        write_stream,
    ):

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            await session.initialize()

            tools = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools.tools:
                print(f"- {tool.name}")

            result = await session.call_tool(
                "get_order_status",
                arguments={
                    "order_id": order_id
                },
            )

            print(
                f"\nMCP response for {order_id}:"
            )

            for content in result.content:
                print(content.text)


async def main():

    print("=" * 70)
    print("NYKAA MCP CLIENT-SERVER ROUND TRIP")
    print("=" * 70)

    await check_order("ORD1001")
    await check_order("ORD1002")

    print("\n" + "=" * 70)
    print("MCP ROUND TRIP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())