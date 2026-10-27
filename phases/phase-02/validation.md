# Argument and Result Validation (Experiments 2, 3, 4, 5, 11)

Evidence: scripted tool calls; tool body execution log; error text fed back; wire payloads. Pinned in `tests/phase_02/test_batch1_tool_mechanics.py`.

## Arguments
| Input | LangGraph | Pydantic AI | Tool runs? |
|---|---|---|---|
| missing required | error fed to model | error fed to model | no |
| `"hello"` for int | rejected | rejected | no |
| `"10"` for int | **coerced to 10** | **coerced to 10** | yes |
| `10.0` for int | coerced to 10 | coerced to 10 | yes |
| `10.5` for int | rejected | rejected | no |
| `True` for int | **coerced to 1** | **coerced to 1** | yes |
| `None`, `[1]`, `"1e2"` | rejected | rejected | no |
| extra `secret=` (plain function) | **dropped silently, tool runs** | **rejected** (`extra_forbidden`) | LG yes / PA no |
| extra `secret=` (`args_schema` extra=forbid) | rejected | n/a | no |
| extra `secret=` (`**kwargs` function) | dropped (function sees `{}`) | **passed through** (function sees it) | yes |
| enum `"XYZ"`, `"usd"`, `"EUR "` | rejected | rejected | no |

## Architect answers
- **Who validates?** The framework's tool wrapper, using Pydantic, before the function body runs. The function is never reached on failure.
- **Coercion:** `"10"` → `10`, `10.0` → `10`, `True` → `1`; `10.5`, `None`, `"1e2"` fail. Both frameworks identical because both use Pydantic lax mode. A tool that needs strict typing must opt in (e.g. Pydantic `strict`, custom validator); this was not tested.
- **Extra arguments:** a framework default, and they differ. LangGraph ignores, Pydantic AI rejects, `**kwargs` flips it. Never rely on "extras are rejected" without testing the exact framework and signature.
- **Enums:** exact match only, no case/whitespace normalization; the error lists allowed values, giving the model what it needs to self-correct (live self-correction untested).

## Results (Experiment 11)
- **Return types are not validated.** A function annotated `-> int` returning `"abc"` is accepted by both.
- **No size limit or truncation:** a 1,000,000-character result is sent whole by both.
- **Serialization differs:** `None` → `"null"` (LG) vs `""` (PA); `NaN` → `"NaN"` (invalid JSON) vs `"null"`; non-serializable object → `repr()` (LG, the model sees an object description) vs `PydanticSerializationError` that **ends the run** (PA).
- **Who validates tool output?** Nobody by default. If output must be bounded or typed, the tool (or an explicit wrapper) must do it. Prompt-injection consequences are Experiment 12.
