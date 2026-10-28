# Phase 2 Architecture — who owns what in tool calling (Batches 1–5)

Every row is backed by a measured experiment (batch.experiment in brackets). Layer names are used consistently across the Phase 2 docs.

```
USER request
   |
   v
MODEL ............ decides: tool or no tool, which tool, which arguments, whether to ask, final wording
   |                influenced by: user wording (verb echo [5.8]), tool descriptions incl. injected instructions [5.16],
   |                previous tool output incl. injected instructions [5.18, 3.13], system prompt [5.17], request envelope bytes [5.10]
   |   tool call (name + args)          <- NOT an authority; never a security boundary
   v
FRAMEWORK ........ builds the request (tool order, schema keywords, tool_choice, temperature) [5.14]
   |                rejects unknown tool names and invalid args, feeds errors back [5.M, 1.x]
   |                loop, retries, limits, timeouts [1.14, 2.x]; executes exactly what was selected [5.M]
   |                provides interception points and trusted context [3.2]; does not authorize, deduplicate or sanitize
   v
RUNTIME .......... threads / event loop; overlap of batched calls; cancellation reaches async only [4.x]
   v
APPLICATION ...... identity (trusted context, per request) [3.7, 4.15]; permission + tenant/resource policy [3.1–3.4]
   |                which tools exist for this caller and with which descriptions (tool definitions are code) [5.16]
   |                tool_choice policy [5.15]; audit; denial semantics [3.6]
   v
BUSINESS / DATA .. idempotency (atomic key) [2.4, 4.9]; invariants and constraints; side effects; last line of defense
```

## Q&A (Batch 5)
| # | Question | Answer (evidence) |
|---|---|---|
| Q1 | Who selects the tool? | The model. Both frameworks forward the definitions in declaration order and execute exactly the selected valid tool [5.M, 5.14]. Live selections were identical across frameworks wherever the request bytes were identical. |
| Q2 | How much does naming matter? | Little when descriptions differ (`order_op_a` 6/6) [5.4]; decisive among near-synonyms, where the tool whose name echoes the user's verb wins [5.8]. |
| Q3 | How much do descriptions matter? | A richer *benign* description changed nothing in this domain [5.5]; an *instruction* in a description changed selection 20/20 [5.16]. Descriptions are prompts. |
| Q4 | How much does schema quality matter? | Not for selection or for argument normalization with this model [5.6]; it matters as **framework validation** (pattern/missing arg rejected before the body) [5.M]. Schema keywords can change model behavior indirectly [5.10 ablation]. |
| Q5 | How does tool count affect selection? | No accuracy drop up to 20 tools for a clear request; cost grows ~linearly (~23 tokens per tool) [5.7]. |
| Q6 | When should an agent ask for clarification? | When required information is missing (no order id: asked 10/10 per wording) or the intent is open ("Do something…": asked 10/10, both models). For intent-vague but read-safe wordings the two models differed [5.9]. If clarification is required, make it a design rule (prompt or code); do not rely on model defaults. |
| Q7 | What happens when no suitable tool exists? | No invented tool names (0/20); behavior then depends on the model *and the request envelope*: honest refusal, pretended capability (PA envelope, 5/5), or a refusal justified by an invented policy (120b) [5.10]. The framework rejects invented names if they occur [5.M]. |
| Q8 | Can malicious tool descriptions influence selection? | Yes, 20/20, including triggering an unrequested side-effecting call [5.16]. |
| Q9 | Can malicious tool output influence subsequent selection? | Yes, 10/10 [5.18] (and 20/20 in Batch 3). |
| Q10 | Can authorization be delegated to the LLM? | No. The model selected restricted tools 10/10 without role info, refrained 10/10 with it, and could be redirected by injections; only the application check held every time [5.17, 5.18, 3.x]. **Never rely on LLM tool selection as an authorization boundary.** |
| Q11 | What belongs to the model? | Intent interpretation, tool/no-tool choice, arguments, clarification, wording. |
| Q12 | What belongs to the framework? | Request construction (which keywords, order, `tool_choice`), validation of names/args, loop/limits, delivery of results/errors, hook points. Its defaults are part of the experiment: compare captured requests, not code. |
| Q13 | What belongs to the runtime? | Concurrency/threads/cancellation of the executed calls [Batch 4]; nothing about selection. |
| Q14 | What belongs to the application? | Identity, authorization, the tool catalog and its descriptions (reviewed like code), `tool_choice` policy, clarification policy, audit. |
| Q15 | What belongs to the business/data layer? | Idempotency, constraints and the side effects themselves; it must hold even if every layer above chose wrongly. |

## Design rules derived (Phase 2)
1. Treat every model tool call as an untrusted request; authorize every call at the application boundary with trusted context [3.x, 5.17].
2. Treat tool definitions as code and as prompt: review them, pin third-party ones, diff them in CI [5.16].
3. Treat tool output as untrusted data [3.8, 5.18].
4. Compare frameworks (and upgrades) by captured requests, not by code [5.14, 5.10].
5. Put idempotency and atomicity in the data layer [2.4, 4.9].
6. Make clarification and refusal explicit product rules if they matter; model defaults differ between models [5.9, 5.10].
