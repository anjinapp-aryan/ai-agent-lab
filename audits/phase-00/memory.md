# E. Agent Memory

| Project | License | Latest | Pushed | Decision |
|---|---|---|---|---|
| Mem0 | Apache-2.0 | ts-v3.3.1 (09-25) | 10-26 | REUSE — vector-backed extraction/retrieval, most adopted |
| Graphiti (Zep) | Apache-2.0 | v0.31.2 (10-20) | 10-26 | REUSE (Phase 3) — temporal knowledge-graph memory; needs graph DB `[K]` |
| LangMem | MIT | no GitHub release | 10-26 | REUSE (Phase 3) — semantic/episodic/procedural taxonomy on LangGraph store `[K]` |
| Letta (MemGPT) | Apache-2.0 | 0.16.8 (05-14) | 10-18 | REFERENCE ONLY — OS-style memory idea; full server, cadence slowed |

All `[V]` for license/release/push dates.

Build: none (building a memory framework or vector DB is forbidden). Use a stock local vector store; check its license at Phase 3 start.
Phase 3 comparison: same conversation through Mem0, Graphiti, LangMem; inspect what each stores and mis-retrieves.
