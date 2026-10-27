# Framework-Free Mental Model

```python
messages = [system, user_question]
for step in range(MAX_STEPS):                      # termination guard #1: budget
    response = model(messages, tools=TOOL_SCHEMAS)  # 1. model call
    messages.append(response)                       # 2. state grows
    if not response.tool_calls:                     # 3. stop condition: no action requested
        return response.text
    for call in response.tool_calls:                # 4. model chose action(s)
        try:
            args = validate(TOOL_SCHEMAS[call.name], call.args)   # 5. validation
            result = TOOLS[call.name](**args)                     # 6. execution
        except Exception as e:
            result = f"error: {e}"                  # 7. error handling policy: feed back, or raise
        messages.append(tool_result(call.id, result))  # 8. result injection
raise StepLimitExceeded                             # termination guard #2
```

- **Model call**: stateless; it sees the entire `messages` list every time.
- **Tool selection**: the model emits a structured request (name + JSON args) from schemas derived from the function signature and docstring. The model never runs anything.
- **Execution**: ordinary code in your process.
- **Result injection**: result appended as a tool message tied to the call id; the next model call reads it.
- **Termination**: the model answers without requesting a tool, or the budget runs out.
- **Error handling**: a policy choice — return the error text to the model (it may recover) or raise (run ends). Both frameworks let you pick; defaults differ (see failure-analysis.md).
- **State**: `messages` is the state. Everything else (graph nodes, run contexts) is bookkeeping around it.

Agent = goal (user question) + environment (tools) + actions (tool calls) + observations (tool results) + decision policy (the LLM) + state (`messages`) + termination. A single LLM call has none of the loop: no observation feeds back into a further decision.
Not every loop is intelligent: if the sequence of tools is fixed in code, it is a workflow with an LLM step, not an agent.

This is not an implementation to build — both frameworks already are this loop.
