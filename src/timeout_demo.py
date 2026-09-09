import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class TimeoutState(TypedDict, total=False):
    status: str
    message: str


# =========================================================
# PER-NODE TIMEOUT DEMO
# =========================================================

def slow_node(state: TimeoutState):
    print("Slow node started...")
    time.sleep(2)
    print("Slow node finished.")

    return {
        "status": "completed",
        "message": "Slow node completed.",
    }


def run_with_node_timeout(timeout_seconds=1):

    print("\n--- PER-NODE TIMEOUT ---")
    print(
        f"Node timeout configured: "
        f"{timeout_seconds} second"
    )

    start_time = time.perf_counter()

    try:

        result = slow_node(
            {
                "status": "started"
            }
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        if elapsed > timeout_seconds:
            raise TimeoutError(
                "Per-node timeout exceeded."
            )

        print("Node result:")
        print(result)

    except TimeoutError as error:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"PER-NODE TIMEOUT TRIGGERED "
            f"after {elapsed:.2f} seconds."
        )

        print(f"Reason: {error}")


# =========================================================
# GLOBAL TIMEOUT DEMO
# =========================================================

def workflow_node_one(state: TimeoutState):

    print("Global workflow - Node 1")
    time.sleep(0.5)

    return {
        "status": "node1_done"
    }


def workflow_node_two(state: TimeoutState):

    print("Global workflow - Node 2")
    time.sleep(2)

    return {
        "status": "node2_done"
    }


def workflow_node_three(state: TimeoutState):

    print("Global workflow - Node 3")
    time.sleep(0.5)

    return {
        "status": "node3_done"
    }


def build_workflow():

    graph = StateGraph(
        TimeoutState
    )

    graph.add_node(
        "node_one",
        workflow_node_one,
    )

    graph.add_node(
        "node_two",
        workflow_node_two,
    )

    graph.add_node(
        "node_three",
        workflow_node_three,
    )

    graph.add_edge(
        START,
        "node_one",
    )

    graph.add_edge(
        "node_one",
        "node_two",
    )

    graph.add_edge(
        "node_two",
        "node_three",
    )

    graph.add_edge(
        "node_three",
        END,
    )

    return graph.compile()


def run_with_global_timeout(
    timeout_seconds=1
):

    print("\n--- GLOBAL TIMEOUT ---")
    print(
        f"Global timeout configured: "
        f"{timeout_seconds} second"
    )

    graph = build_workflow()

    start_time = time.perf_counter()

    try:

        result = graph.invoke(
            {
                "status": "started"
            }
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        if elapsed > timeout_seconds:
            raise TimeoutError(
                "Global workflow timeout exceeded."
            )

        print("Workflow result:")
        print(result)

    except TimeoutError as error:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"GLOBAL TIMEOUT TRIGGERED "
            f"after {elapsed:.2f} seconds."
        )

        print(f"Reason: {error}")


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("TIMEOUT POLICY DEMONSTRATION")
    print("=" * 70)

    run_with_node_timeout(
        timeout_seconds=1
    )

    run_with_global_timeout(
        timeout_seconds=1
    )

    print("\n" + "=" * 70)
    print("TIMEOUT DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()