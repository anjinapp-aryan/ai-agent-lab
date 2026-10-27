# LangGraph vs Pydantic AI (v1.2.18 vs v2.62.0)

Evidence: code written, scripted-model runs, span trees. `[unobserved]` = needs live runs.

| Capability | LangGraph | Pydantic AI |
|---|---|---|
| Agent loop | You wire it: `agent` node ↔ `ToolNode` with conditional edge `tools_condition`. ~10 lines of graph code. | Hidden inside `Agent.run` (internal graph). One constructor call. |
| Tool definition | Plain function passed to `ToolNode` / `bind_tools` (auto-wrapped as a LangChain tool) | Plain function passed to `Agent(tools=...)` |
| Tool schema | Derived from signature + docstring via LangChain tool layer | Derived from signature + docstring via pydantic |
| State | Explicit, typed: `MessagesState` dict with append reducer; visible and customizable | Implicit message history on the run; accessible via `result.all_messages()` |
| Context | State schema | Dependency injection `deps_type` / `RunContext` (not used in Phase 1) |
| Validation | Pydantic on tool args, surfaced as `ToolInvocationError` text | Pydantic on args + `output_type` validation; structured error list |
| Retry | No retry counter in loop; error text goes back, model decides, cap = recursion_limit | First-class per-tool `retries`; exhausted → `UnexpectedModelBehavior` |
| Error handling | Tool exceptions → error ToolMessage by default (observed); toggle `handle_tool_errors` | Tool exceptions escape by default; `ModelRetry` for model-visible errors |
| Control flow | Arbitrary graph: branches, cycles, subgraphs, interrupts | Linear agent loop; `pydantic-graph` for custom flows `[K]` |
| Observability | OpenInference LangChain instrumentor; 14 spans for 2-tool run, node-level CHAIN/AGENT spans | Native OTel spans (`invoke_agent`/`chat`/`execute_tool`); 6 spans for same run; OpenInference processor adds span kinds |
| Testing | Write your own fake chat model (needed `bind_tools` stub) | Built-in `FunctionModel`/`TestModel` — easier |
| Debugging | Per-node visibility, checkpoint/time-travel available `[unobserved]` | Fewer moving parts; error surfaces as raw exception |
| Extensibility | Highest (custom nodes/edges/state) | High for agents/tools/capabilities; graph is the escape hatch |
| Boilerplate | More (state, nodes, edges, compile) | Less |
| Learning transparency | High: the loop is visible code | Lower: the loop is inside the library, but the code is shorter |
| Production suitability | Strong for long-running/stateful/HITL flows (checkpointers) `[K]` | Strong for typed, validated single-agent services; durable-exec integrations `[K]` |

## When LangGraph
Need explicit control flow, shared state, branching, human interrupts, resumability, or multi-agent graphs; want to *see* the loop.

## When Pydantic AI
Need a typed, validated tool-using agent fast; strict failure defaults and retry budgets; easy offline testing; structured outputs.

No overall winner.

## Abstraction analysis (concept → abstraction → mechanism)
| Concept | LangGraph | Pydantic AI | Underlying mechanism |
|---|---|---|---|
| Model's decision | `AIMessage.tool_calls` | `ModelResponse` with `ToolCallPart` | Provider returns structured tool-call JSON (name, args, id) |
| Loop | Graph cycle + `tools_condition` | `Agent.run` internal graph | `while` loop around model call |
| Stop | Edge to `END` when no tool_calls | Text part with no tool calls = output | Model answers without requesting an action |
| Tool result | `ToolMessage(tool_call_id)` appended to `messages` | `ToolReturnPart` in next `ModelRequest` | Result appended to history keyed by call id |
| State | `MessagesState` | message history | The list of messages |
| Step budget | `recursion_limit` | `UsageLimits.request_limit` | counter in the loop |
| Error to model | `ToolMessage(status="error")` | `ToolReturnPart(outcome="retried")` | Error string in the tool-result slot |
| Provider | `ChatOpenAI(base_url)` | `OpenAIChatModel(OpenAIProvider(base_url))` | HTTP POST /chat/completions with `tools=[...]` |
