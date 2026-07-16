# perf

kind: session (parallel branch)
agent: reviewer

source:
```prose
perf = session: reviewer
  prompt: "Review for performance issues"
```

---

Reviewed `tools/src/openprose_tools/` (canonical.py, ledger.py, verify.py, inspect_run.py, models.py, cli.py — ~300 LOC total). Package is small and mostly O(n) as intended; found a few concrete inefficiencies, no algorithmic blow-ups.

1. **`ledger.py:41-43` (`load_run`)** — `ledger_path.read_text(encoding="utf-8").splitlines()` reads the entire `receipts.jsonl` into one string, then materializes a second full list of line strings, before any JSON parsing starts. Since receipts.jsonl is append-only and grows for the life of a run, this means peak memory is ~2x file size before processing begins. Streaming with `for lineno, raw in enumerate(ledger_path.open(encoding="utf-8"), start=1)` would avoid the intermediate copy.

2. **`inspect_run.py:37-51`** — `by_basis.setdefault(basis, {"receipts": 0, ...})` and `by_agent.setdefault(agent, {...})` construct a brand-new dict literal on *every* line, even when the key already exists, because Python evaluates `setdefault`'s default argument eagerly regardless of whether it's used. For a run with thousands of receipts concentrated on a few agents/bases, this is one wasted dict allocation per line. Guard with `row = by_agent.get(agent); if row is None: row = by_agent[agent] = {...}`.

3. **`inspect_run` + `verify_ledger` double traversal** — `inspect_run()` (inspect_run.py:17) calls `verify_ledger(raw)`, which itself re-validates every receipt via pydantic and recomputes `receipt_content_hash` (canonical_json + sha256) for each line, then `inspect_run` does a second independent full pass over `raw.lines` for dispositions/usage. Two full O(n) passes with the expensive hashing/validation work concentrated in the first — acceptable at current scale but worth merging if ledgers grow large or `inspect --json` becomes a hot path (e.g. CI polling).

4. **`canonical.py:_validate` + `json.dumps`** — `canonical_json` does one recursive Python-level walk (`_validate`) plus a second implicit traversal inside `json.dumps(sort_keys=True)`. Called once per receipt per verify pass; a hand-rolled single-pass canonical encoder would halve this but is not warranted unless ledgers get very large.

No blocking issues; all are constant-factor, not correctness or scaling-order problems.
