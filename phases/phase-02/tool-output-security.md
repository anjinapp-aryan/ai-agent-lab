# Tool-Output Security (Experiments 3.8 – 3.13)

Evidence: [experiments.md](experiments.md#batch-3-results-2026-10-28). Framework behavior (F) is separated from model behavior (M).

## Answers
8. **Is tool output trusted?** The frameworks treat it as ordinary context: it reaches the model unchanged, as a tool-role message, with no sanitizing, delimiting or trust marking (F, both frameworks, wire-level capture). The model treats it as potentially instruction-bearing (M).
9. **Can tool output influence subsequent tool calls?** Yes. Live, `gpt-oss-120b` issued the injected `refund_order` call in 20/20 protected runs and 10/10 unprotected runs for the instruction-like and fake-system payloads, although the user had only asked for a status and the system prompt said "only perform actions the user asked for" (M).
10. **Can tool output contain instructions?** Yes, and nothing in either framework distinguishes them from data.
11. **Does structured output improve security?** Not by itself. The framework only serializes it to JSON text. In this experiment a `{"next_action": "refund_order"}` field was followed 0/10 (and the same JSON as a plain string 0/10), whereas prose instructions were followed 20/20 — a difference in **model** susceptibility to imperative prose, not a framework protection, from one model and one payload per class. Do not infer safety.
12. **Does either framework protect automatically?** No (F). Neither detected the injection, altered the tool message or restricted follow-up calls.
13. **Can authorization stop an injected second call?** Yes for what the user is not allowed to do: with the boundary enforced, injected `refund_order` calls produced **0 side effects in all 20 protected live runs** and in the scripted flows. Unprotected, the same model refunded the order 10/10 and told the user "refunded". But authorization does not establish intent: an OPERATOR who may refund *would* have been refunded by the injection (scripted), and a resource check was needed to stop an injected refund of another tenant's order.

## Side effects of authorization worth designing for
- **Denial-of-service by injection:** in Pydantic AI, a plain `PermissionDenied` ended the run in 10/10 injected live runs, so the user's harmless question got no answer. In LangGraph the run continued.
- **Silent suppression:** with the injected "Do not tell the user", LangGraph's final answers (5/5) never mentioned the attempted refund or the denial; without it 5/5 disclosed it. The user was not told a denied action was attempted unless the model chose to. Surface denials to the user from the application layer (audit log, UI), not through the model.

## What was not tried (by design)
Prompt delimiting, output filtering, instruction-hierarchy prompts, injection detectors: this batch measured the boundary, not mitigations. Human-in-the-loop approval was not exercised.
