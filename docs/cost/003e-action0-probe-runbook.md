# ADR-ECO-003e Action №0 — bounded exposure probe (runbook)

Goal: runtime-confirm the static audit (usage wiring absent on flagged
paths) and measure which adapter paths carry token usage. 1–2 pre-limited
runs; no claim of full statistics. This is the acceptance gate for the
003e implementation in atp-platform.

## Steps

1. Pick 1–2 cheap suites that exercise different adapter paths, e.g.:
   - CLI path: `method/cases/code-review` with ONE routable agent
     (`method/run_pipe_check.py` limited to a single agent), or any small
     suite via `--adapter=cli`.
   - HTTP path: an `examples/` suite against a local demo agent, if wired.
2. Export the capture sink before the run:

   ```bash
   export ATP_USAGE_CAPTURE_PATH=_bench_output/003e-probe/usage.jsonl
   ```

3. Run the suite(s) with a hard external limit (small case count, one
   agent, one run each — the "bounded" in bounded probe).
4. Build the evidence report:

   ```bash
   uv run python -m atp.cost.probe_report _bench_output/003e-probe/usage.jsonl
   ```

5. Read the report against the acceptance gate:
   - `cost_usd populated: 0/N` → confirms cost_usd is never set (audit
     finding holds at runtime).
   - `model known: 0/N` → confirms no model id reaches the seam (the
     `model="unknown"` finding).
   - per-adapter `with_usage` → which paths actually carry tokens; paths
     with usage but no price identity are the $-exposure to fix first.
6. Paste the report into the 003e thread in
   `../prograph-vault/authored/decisions/` review notes (or the PR), and
   order M1 adapter adoption by the token volume column, per the ADR
   ("fix by money, not by which gap looks scariest").

## Rollback

Unset `ATP_USAGE_CAPTURE_PATH` — the seam degrades to NullUsageCapture;
no other behavior changes.
