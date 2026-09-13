import random
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class RetryState(TypedDict, total=False):
    attempts: int
    status: str
    result: str


# =========================================================
# RETRY CONFIGURATION
# =========================================================

MAX_ATTEMPTS = 4
INITIAL_INTERVAL = 0.5
MAX_INTERVAL = 2.0
JITTER = 0.2


# =========================================================
# SIMULATED TRANSIENT OPERATION
# =========================================================

def unreliable_operation(attempt: int):
    print(f"Attempt {attempt}")

    # First 2 attempts fail transiently.
    if attempt < 3:
        print("Transient failure occurred.")
        raise RuntimeError("Simulated transient service failure")

    print("Transient failure recovered.")

    return {
        "status": "success",
        "result": "Operation completed after exponential-backoff retry.",
    }


# =========================================================
# RETRY NODE
# =========================================================

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

            if attempt >= MAX_ATTEMPTS:
                print("Maximum retry attempts reached.")
                break

            # Exponential backoff:
            # 0.5 -> 1.0 -> 2.0 -> capped at 2.0
            base_interval = min(
                INITIAL_INTERVAL * (2 ** (attempt - 1)),
                MAX_INTERVAL,
            )

            # Random jitter in the range [-JITTER, +JITTER]
            jitter_value = random.uniform(
                -JITTER,
                JITTER,
            )

            wait_time = max(
                0.0,
                base_interval + jitter_value,
            )

            print(
                f"Retry scheduled: base={base_interval:.2f}s, "
                f"jitter={jitter_value:+.2f}s, "
                f"wait={wait_time:.2f}s"
            )

            time.sleep(wait_time)

    raise last_error


# =========================================================
# LANGGRAPH
# =========================================================

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


# =========================================================
# DEMONSTRATION
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RETRY POLICY / EXPONENTIAL BACKOFF DEMONSTRATION")
    print("=" * 70)

    print(f"Maximum attempts : {MAX_ATTEMPTS}")
    print(f"Initial interval : {INITIAL_INTERVAL} seconds")
    print(f"Maximum interval : {MAX_INTERVAL} seconds")
    print(f"Jitter range     : +/- {JITTER} seconds")
    print()

    graph = build_graph()

    result = graph.invoke(
        {
            "attempts": 0,
            "status": "pending",
            "result": "",
        }
    )

    print()
    print("Final result:")
    print(result)

    print()
    print("=" * 70)
    print("RETRY DEMONSTRATION PASSED")
    print(f"Successful on attempt: {result['attempts']}")
    print("=" * 70)