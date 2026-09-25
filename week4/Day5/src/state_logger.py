"""
Day 5 - Task 5: State Logging.
Every node transition gets an annotated trace entry: which node ran, what
it read from state, what it decided, and when. This is what lets a
developer (or a grader) reconstruct exactly why the agent did what it did
for any given call, after the fact.
"""
import time
import functools


def logged_node(node_name):
    """Decorator: wraps a LangGraph node function so every call appends a
    trace entry to state['execution_trace'] and updates state['current_node'].
    The node function itself just returns its normal partial-state update;
    this decorator adds the logging fields on top."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            t0 = time.time()
            update = fn(state) or {}
            elapsed_ms = round((time.time() - t0) * 1000, 2)

            trace_entry = {
                "node": node_name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_ms": elapsed_ms,
                "intent_at_entry": state.get("intent"),
                "update_keys": list(update.keys()),
            }
            update["current_node"] = node_name
            update["execution_trace"] = [trace_entry]
            return update
        return wrapper
    return decorator


def print_trace(state):
    """Human-readable execution trace for a completed turn/call."""
    print("=== Execution Trace ===")
    for entry in state["execution_trace"]:
        print(f"  [{entry['timestamp']}] {entry['node']:<22} "
              f"({entry['elapsed_ms']:>6}ms)  intent={entry['intent_at_entry']}  "
              f"updated={entry['update_keys']}")
