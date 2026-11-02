"""Offline trajectory labels (B4-3.1.37). Measurement only: a pure function over already-recorded tool executions.

It never runs during an agent run and cannot retry, stop or execute anything. Decision record:
phases/phase-03/decision-record.md. A label describes syntax, not whether the behavior was wrong.
"""
import json

MIN_REPEATS = 3
MIN_CYCLES = 2


def classify(executions):
    """executions: [(tool, args, outcome), ...] in order -> REPEATED_CALL | OSCILLATION | NO_PROGRESS | NONE."""
    calls = [(t, json.dumps(a, sort_keys=True, default=str)) for t, a, _ in executions]
    outcomes = [json.dumps(o, sort_keys=True, default=str) for _, _, o in executions]
    run_start = 0
    repeated = False
    for i in range(1, len(calls) + 1):
        if i == len(calls) or calls[i] != calls[run_start]:
            if i - run_start >= MIN_REPEATS:
                repeated = True
                if len(set(outcomes[run_start:i])) == 1:
                    return "NO_PROGRESS"
            run_start = i
    if repeated:
        return "REPEATED_CALL"
    for i in range(len(calls) - 2 * MIN_CYCLES + 1):
        a, b = calls[i], calls[i + 1]
        if a != b and all(calls[i + j] == (a if j % 2 == 0 else b) for j in range(2 * MIN_CYCLES)):
            return "OSCILLATION"
    return "NONE"
