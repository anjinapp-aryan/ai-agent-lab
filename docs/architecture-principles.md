# Architecture Principles

- Framework-neutral: no single framework forced on every experiment.
- Small dependency set; each dependency justified in an audit.
- Model-provider agnostic where possible (LiteLLM / framework model adapters), so lock-in stays visible.
- Every experiment is traced (OpenTelemetry-compatible) and has at least one deterministic check.
- Bounded loops: max steps, timeouts, budget on every agent run.
- Humans approve irreversible actions.
- Experiments are runnable, breakable, and write up what failed.
