import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class RetryState(TypedDict, total=False):
    attempts: int
    status: str
    result: str


MAX_RETRIES = 3


def unreliable_operation(attempt):
    print(f"Attempt {attempt}")

    if attempt < 3:
        print("Transient failure occurred.")

        raise RuntimeError(
            "Simulated transient service failure"
        )

    print("Transient failure recovered.")

    return {
        "status": "success",
        "result": "Operation completed after retry.",
    }


def retry_node(state: RetryState):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            result = unreliable_operation(
                attempt
            )

            return {
                "attempts": attempt,
                "status": result["status"],
                "result": result["result"],
            }

        except RuntimeError as error:

            last_error = error

            if attempt < MAX_RETRIES:

                print(
                    f"Retry {attempt} of "
                    f"{MAX_RETRIES - 1}"
                )

                time.sleep(0.5)

            else:

                print(
                    f"Retry {attempt} of "
                    f"{MAX_RETRIES}"
                )

    raise last_error


def build_graph():

    graph = StateGraph(RetryState)

    graph.add_node(
        "retry_operation",
        retry_node,
    )

    graph.add_edge(
        START,
        "retry_operation",
    )

    graph.add_edge(
        "retry_operation",
        END,
    )

    return graph.compile()


def main():

    print("=" * 70)
    print(
        "RETRY POLICY / TRANSIENT FAILURE DEMONSTRATION"
    )
    print("=" * 70)

    graph = build_graph()

    result = graph.invoke(
        {
            "attempts": 0,
            "status": "started",
        }
    )

    print("\nFinal result:")
    print(result)

    print("\n" + "=" * 70)

    if result.get("status") == "success":

        print(
            "RETRY DEMONSTRATION PASSED"
        )

        print(
            f"Successful on attempt: "
            f"{result.get('attempts')}"
        )

    else:

        print(
            "RETRY DEMONSTRATION FAILED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()