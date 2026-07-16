# security

kind: session (parallel branch)
agent: reviewer

source:
```prose
security = session: reviewer
  prompt: "Review for security vulnerabilities"
```

---

Scope: `tools/src/openprose_tools/` (cli.py, ledger.py, canonical.py, inspect_run.py, verify.py, models.py). This is a local, offline CLI that reads `receipts.jsonl`/`run.json` from a user-supplied path, verifies a hash chain, and prints a report. No network I/O, no subprocess/`eval`/`exec`/`pickle`/`yaml.load`, no templating — the attack surface is small.

Findings (none rise to "vulnerability"; noting hardening opportunities):

1. **Unbounded file reads (low, DoS)** — `ledger.py:42` (`ledger_path.read_text(...)`) and `:62` (`manifest_path.read_text(...)`) load the entire file into memory with no size cap before `json.loads`. A crafted or corrupted `receipts.jsonl` (e.g. a huge single line, or gigabytes of valid JSONL) can exhaust memory in a single process. Not remotely exploitable, but worth a size guard if run-dirs are ever untrusted input (e.g. fed from a shared/CI artifact store).

2. **No path confinement on `run_dir` (informational)** — `load_run()` (`ledger.py:26`) accepts any path, including outside the expected `.prose/runs/` tree, and follows symlinks transparently. Since this is a local CLI operating with the invoking user's own privileges on a path they choose, this is expected/benign — flagging only in case this loader is ever wrapped by a service that accepts a run-dir from an untrusted caller, at which point path traversal / symlink-escape checks would become necessary.

3. **Trust boundary is honest, not a bug** — `verify.py`'s docstring correctly scopes itself to chain consistency "given a trusted manifest," not tamper-proofing; content hashes are SHA-256 over canonical JSON (`canonical.py`), which is appropriate for integrity-detection but not for authenticity (no signing/HMAC) — fine for its stated purpose of catching accidental truncation/corruption, not for defending against a malicious ledger author.

No injection, deserialization, secrets-handling, or auth issues found. `pydantic` models use `extra="ignore"` consistently and deliberately (hashing is done over raw dicts, not model dumps), which is correct and avoids a class of hash-mismatch bugs, not a security gap.
