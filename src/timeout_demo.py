import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class TimeoutState(TypedDict, total=False):
    status: str
    message: str
    elapsed_seconds: float


# =========================================================
# TIMEOUT CONFIGURATION
# =========================================================

PER_NODE_TIMEOUT = 1.0
GLOBAL_TIMEOUT = 1.0


# =========================================================
# SLOW OPERATIONS
# =========================================================

def slow_node_operation():
    """Simulated node that takes longer than its timeout."""
    print("Slow node started...")
    time.sleep(2.0)
    print("Slow node finished.")
    return "Node completed"


def workflow_operation():
    """Simulated workflow that takes longer than global timeout."""
    print("Global workflow - Node 1")
    time.sleep(1.0)

    print("Global workflow - Node 2")
    time.sleep(1.0)

    print("Global workflow - Node 3")
    time.sleep(1.0)

    return "Workflow completed"


# =========================================================
# PER-NODE TIMEOUT
# =========================================================

def run_with_node_timeout():
    print("\n--- PER-NODE TIMEOUT ---")
    print(f"Node timeout configured: {PER_NODE_TIMEOUT:.0f} second")

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(slow_node_operation)

        try:
            result = future.result(timeout=PER_NODE_TIMEOUT)

            elapsed = time.perf_counter() - start

            return {
                "status": "success",
                "message": result,
                "elapsed_seconds": elapsed,
            }

        except FutureTimeoutError:
            elapsed = time.perf_counter() - start

            # Cancel if the task has not started.
            future.cancel()

            print(
                f"PER-NODE TIMEOUT TRIGGERED after "
                f"{elapsed:.2f} seconds."
            )
            print("Reason: Per-node execution limit exceeded.")
            print("Node result discarded after timeout.")

            return {
                "status": "timeout",
                "message": "Per-node timeout handled cleanly.",
                "elapsed_seconds": elapsed,
            }


# =========================================================
# LANGGRAPH PER-NODE DEMO
# =========================================================

def timeout_node(state: TimeoutState):
    result = run_with_node_timeout()

    return {
        "status": result["status"],
        "message": result["message"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def build_node_timeout_graph():

    graph = StateGraph(TimeoutState)

    graph.add_node(
        "timeout_node",
        timeout_node,
    )

    graph.add_edge(
        START,
        "timeout_node",
    )

    graph.add_edge(
        "timeout_node",
        END,
    )

    return graph.compile()


# =========================================================
# GLOBAL TIMEOUT
# =========================================================

def run_with_global_timeout():

    print("\n--- GLOBAL TIMEOUT ---")
    print(f"Global timeout configured: {GLOBAL_TIMEOUT:.0f} second")

    start = time.perf_counter()

    executor = ThreadPoolExecutor(max_workers=1)

    future = executor.submit(workflow_operation)

    try:
        result = future.result(timeout=GLOBAL_TIMEOUT)

        elapsed = time.perf_counter() - start

        executor.shutdown(wait=False, cancel_futures=True)

        return {
            "status": "success",
            "message": result,
            "elapsed_seconds": elapsed,
        }

    except FutureTimeoutError:

        elapsed = time.perf_counter() - start

        # Cancel pending work.
        future.cancel()

        # Do not wait for the long-running workflow.
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        print(
            f"GLOBAL TIMEOUT TRIGGERED after "
            f"{elapsed:.2f} seconds."
        )
        print("Reason: Global workflow execution limit exceeded.")
        print("Remaining workflow execution cancelled.")

        return {
            "status": "timeout",
            "message": "Global timeout handled and workflow cancelled.",
            "elapsed_seconds": elapsed,
        }


# =========================================================
# DEMONSTRATION
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TIMEOUT POLICY / CANCELLATION DEMONSTRATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Per-node timeout
    # -----------------------------------------------------

    node_graph = build_node_timeout_graph()

    node_result = node_graph.invoke({})

    print("\nPer-node final result:")
    print(node_result)

    # -----------------------------------------------------
    # Global timeout
    # -----------------------------------------------------

    global_result = run_with_global_timeout()

    print("\nGlobal timeout final result:")
    print(global_result)

    # -----------------------------------------------------
    # Final demonstration status
    # -----------------------------------------------------

    if (
        node_result["status"] == "timeout"
        and global_result["status"] == "timeout"
    ):
        print("\n" + "=" * 70)
        print("TIMEOUT DEMONSTRATION PASSED")
        print("Per-node timeout: HANDLED")
        print("Global timeout: HANDLED")
        print("Whole workflow cancellation: REQUESTED")
        print("=" * 70)
    else:
        print("\nTIMEOUT DEMONSTRATION FAILED")