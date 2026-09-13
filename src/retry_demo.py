import random
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class RetryState(TypedDict, total=False):
    attempts: int
    status: str
    result: str


MAX_ATTEMPTS = 4
INITIAL_INTERVAL = 0.5
MAX_INTERVAL = 2.0
JITTER = 0.1


def unreliable_operation(attempt: int):
    print(f"Attempt {attempt}")

    # Simulate transient failures on the first two attempts.
    if attempt < 3:
        print("Transient failure occurred.")
        raise RuntimeError("Simulated transient service failure")

    print("Transient failure recovered.")
    return {
        "status": "success",
        "result": "Operation completed after retry.",
    }


def retry_node(state: RetryState):
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = unreliable_operation(attempt)

            return {
                "attempts": attempt,
                "status": result["status"],
                "result": result["result"],
            }

        except RuntimeError as error:
            last_error = error

            if attempt < MAX_ATTEMPTS:
                base_delay = min(
                    INITIAL_INTERVAL * (2 ** (attempt - 1)),
                    MAX_INTERVAL,
                )

                jitter = random.uniform(0, JITTER)
                delay = min(base_delay + jitter, MAX_INTERVAL)

                print(
                    f"Retry {attempt}/{MAX_ATTEMPTS - 1} "
                    f"after {delay:.2f} seconds "
                    f"(exponential backoff + jitter)"
                )

                time.sleep(delay)

            else:
                print(f"Maximum attempts reached: {MAX_ATTEMPTS}")

    raise last_error


builder = StateGraph(RetryState)

builder.add_node("retry_operation", retry_node)

builder.add_edge(START, "retry_operation")
builder.add_edge("retry_operation", END)

graph = builder.compile()


if __name__ == "__main__":
    print("RETRY POLICY DEMONSTRATION")
    print(f"Maximum attempts: {MAX_ATTEMPTS}")
    print(f"Initial interval: {INITIAL_INTERVAL} seconds")
    print(f"Maximum interval: {MAX_INTERVAL} seconds")
    print(f"Jitter: 0 to {JITTER} seconds")
    print()

    result = graph.invoke({})

    print()
    print("FINAL RESULT")
    print(result)