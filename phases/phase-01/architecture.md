# Phase 1 Architecture

## Diagram 1 — basic agent loop
```mermaid
flowchart TD
  U[User question] --> M[LLM sees messages + tool schemas]
  M -->|requests tool| T[Execute tool]
  T -->|result appended to messages| M
  M -->|plain text| F[Final answer]
```

## Diagram 2 — LangGraph (observed span tree)
```mermaid
flowchart LR
  S((START)) --> A[agent node: llm.invoke]
  A --> C{tools_condition}
  C -->|tool_calls| T[tools node: ToolNode]
  T --> A
  C -->|none| E((END))
```
State: `MessagesState` (list of messages, append reducer). Loop = graph cycle. Cap = `recursion_limit` (2*MAX_STEPS+1).

## Diagram 3 — Pydantic AI (observed span tree)
```mermaid
flowchart TD
  R[Agent.run_sync] --> I[invoke_agent span]
  I --> L1[chat span: model request]
  L1 -->|tool call parts| X[execute_tool span]
  X -->|ToolReturnPart| L2[chat span]
  L2 -->|TextPart| O[output]
```
Loop is internal to `Agent.run` (a pydantic-graph). State = message history + run context. Cap = `UsageLimits(request_limit)`.

## Diagram 4 — failure path
```mermaid
flowchart TD
  C[Model emits divide 10,0] --> V{args valid?}
  V -->|no| VR[LangGraph: error text to model / PA: ToolReturnPart outcome=retried, up to retries]
  V -->|yes| X[tool runs, raises ZeroDivisionError]
  X --> LG[LangGraph default: Error text fed to model, run continues]
  X --> PA[Pydantic AI default: exception escapes run_sync, run ends]
  VR --> M[model tries again or explains]
  LG --> M
```

## Diagram 5 — observability path
```mermaid
flowchart LR
  A1[LangGraph agent] --> I1[OpenInference LangChain instrumentor]
  A2[Pydantic AI agent] --> I2[Agent.instrument_all + OpenInferenceSpanProcessor]
  I1 --> O[OpenTelemetry SDK TracerProvider]
  I2 --> O
  O -->|OTLP/HTTP + Basic auth| LF[Langfuse Cloud Hobby]
```
Agent app and observability are separate; Langfuse is hosted, nothing self-hosted.

## Shared pieces (`phase-01/common/lab_common`)
`tools.py` (4 typed functions, docstrings = descriptions), `prompts.py` (instructions, MAX_STEPS=8), `dataset.py` (6 scenarios, RunResult), `evaluate.py` (scorer), `config.py` (env provider), `tracing.py`.
