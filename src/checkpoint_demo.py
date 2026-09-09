import sqlite3
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
CHECKPOINT_DB = PROJECT_FOLDER / "logs" / "checkpoints.sqlite"


class DemoState(TypedDict, total=False):
    message: str
    step1_done: bool
    step2_done: bool
    step3_done: bool


def step_one(state: DemoState):
    print("STEP 1 EXECUTED")

    return {
        "step1_done": True,
        "message": "Step 1 completed",
    }


def step_two(state: DemoState):
    print("STEP 2 EXECUTED")

    # LangGraph pauses execution here.
    # The checkpoint is saved before waiting for resume.
    approval = interrupt(
        {
            "message": "Workflow paused at Step 2.",
            "action": "Resume to continue Step 2.",
        }
    )

    print(
        f"STEP 2 RESUMED WITH: {approval}"
    )

    return {
        "step2_done": True,
        "message": "Step 2 completed",
    }


def step_three(state: DemoState):
    print("STEP 3 EXECUTED")

    return {
        "step3_done": True,
        "message": "Workflow completed",
    }


def build_graph(checkpointer):
    graph = StateGraph(DemoState)

    graph.add_node(
        "step_one",
        step_one,
    )

    graph.add_node(
        "step_two",
        step_two,
    )

    graph.add_node(
        "step_three",
        step_three,
    )

    graph.add_edge(
        START,
        "step_one",
    )

    graph.add_edge(
        "step_one",
        "step_two",
    )

    graph.add_edge(
        "step_two",
        "step_three",
    )

    graph.add_edge(
        "step_three",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer
    )


def main():

    print("=" * 70)
    print("SQLITE INTERRUPT / RESUME DEMONSTRATION")
    print("=" * 70)

    CHECKPOINT_DB.parent.mkdir(
        exist_ok=True
    )

    connection = sqlite3.connect(
        str(CHECKPOINT_DB),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(
        connection
    )

    graph = build_graph(
        checkpointer
    )

    config = {
        "configurable": {
            "thread_id": "nykaa-interrupt-demo-002"
        }
    }

    # -----------------------------------------------------
    # FIRST RUN
    # -----------------------------------------------------

    print("\n--- FIRST RUN ---")
    print("Starting workflow...")

    first_result = graph.invoke(
        {
            "message": "Start workflow"
        },
        config=config,
    )

    print("\nFirst run returned:")
    print(first_result)

    print(
        "\nWorkflow is now INTERRUPTED at Step 2."
    )

    # -----------------------------------------------------
    # CHECKPOINT STATE
    # -----------------------------------------------------

    checkpoint_state = graph.get_state(
        config
    )

    print("\n--- CHECKPOINT STATE ---")

    print(
        "Next node(s):",
        checkpoint_state.next,
    )

    print(
        "Same thread ID:",
        config["configurable"]["thread_id"],
    )

    # -----------------------------------------------------
    # RESUME
    # -----------------------------------------------------

    print("\n--- RESUME RUN ---")

    resumed_result = graph.invoke(
        Command(
            resume="Approved - continue workflow"
        ),
        config=config,
    )

    print("\nResumed result:")
    print(resumed_result)

    print("\n" + "=" * 70)
    print("CHECKPOINT RESUME DEMONSTRATION COMPLETE")
    print("=" * 70)

    print("\nSQLite database:")
    print(CHECKPOINT_DB)

    print("\nThread ID:")
    print(
        config["configurable"]["thread_id"]
    )

    connection.close()


if __name__ == "__main__":
    main()