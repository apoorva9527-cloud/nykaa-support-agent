import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


NODE_TIMEOUT = 1.0
GLOBAL_TIMEOUT = 1.0


def slow_node():
    print("Slow node started...")
    time.sleep(2.0)
    print("Slow node finished.")
    return "slow node completed"


def run_per_node_timeout():
    print("--- PER-NODE TIMEOUT ---")
    print(f"Node timeout configured: {NODE_TIMEOUT:.0f} second")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(slow_node)

        try:
            result = future.result(timeout=NODE_TIMEOUT)
            print(f"Node result: {result}")
        except FutureTimeoutError:
            print("PER-NODE TIMEOUT TRIGGERED.")
            print("Reason: Node exceeded configured timeout.")
            future.cancel()


def workflow_node(name, duration, stop_event):
    print(f"Global workflow - {name}")

    start = time.perf_counter()

    while time.perf_counter() - start < duration:
        if stop_event.is_set():
            print(f"{name} cancelled by global timeout.")
            return "cancelled"

        time.sleep(0.05)

    print(f"{name} completed.")
    return "completed"


def run_global_timeout():
    print("--- GLOBAL TIMEOUT ---")
    print(f"Global timeout configured: {GLOBAL_TIMEOUT:.0f} second")

    stop_event = threading.Event()

    executor = ThreadPoolExecutor(max_workers=3)

    futures = [
        executor.submit(workflow_node, "Node 1", 0.7, stop_event),
        executor.submit(workflow_node, "Node 2", 1.5, stop_event),
        executor.submit(workflow_node, "Node 3", 1.5, stop_event),
    ]

    start_time = time.perf_counter()

    try:
        while True:
            elapsed = time.perf_counter() - start_time

            if elapsed >= GLOBAL_TIMEOUT:
                print("GLOBAL TIMEOUT TRIGGERED.")
                print("Reason: Global workflow timeout exceeded.")

                stop_event.set()

                for future in futures:
                    future.cancel()

                break

            if all(future.done() for future in futures):
                print("Global workflow completed successfully.")
                break

            time.sleep(0.05)

    finally:
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    print("TIMEOUT POLICY DEMONSTRATION")
    print()

    run_per_node_timeout()
    print()

    run_global_timeout()
    print()

    print("TIMEOUT DEMONSTRATION COMPLETE")