# OpenProse Compile IR — `openprose.compile-ir.v1`

The machine contract between the **intelligent compiler** (an LLM reading
`compiler.md`) and **deterministic tooling** (runtime, CI, inspectors).
`prose compile <file>` freezes what it understood about a program —
statement inventory, agent/block tables, session wiring, properties,
diagnostics — into one content-addressed JSON artifact that tools can
validate without a model.

Division of labor:

- The **LLM compiler** produces the IR: it resolves semantics (context
  wiring, agent binding, property merging) that no regex can.
- **`openprose-tools ir-check`** validates it: schema, content hash,
  source freshness, and internal consistency are mechanically decidable.
- The **VM** consumes it at run time (`prose.md`): statement IDs and
  wiring come from a fresh IR rather than being re-derived mid-run.

## Artifact layout

| Context | Path |
| ------- | ---- |
| User project | `.prose/dist/{program-stem}.ir.json` |
| Active pointer | `.prose/dist/manifest.active.json` — `{"programs": {"<source path>": "<ir path>"}}` |
| This repo's committed corpus | `examples/dist/{example-stem}.ir.json` |

An IR is **stale** the moment its source file's bytes change; stale IRs
must never be consumed (see Freshness).

## Top-level schema

A concrete (abbreviated) instance:

```json
{
  "v": "openprose.compile-ir.v1",
  "program": "examples/16-parallel-reviews.prose",
  "source": {
    "path": "examples/16-parallel-reviews.prose",
    "content_hash": "sha256:35db…"
  },
  "state_backend": "filesystem",
  "inputs": {},
  "agents": {
    "reviewer": {
      "model": "sonnet",
      "prompt_hash": "sha256:5561…",
      "persist": null,
      "skills": [],
      "retry": null,
      "backoff": null
    }
  },
  "blocks": {},
  "statements": [
    {"id": "s001", "line": 4, "kind": "agent_def", "name": "reviewer"},
    {
      "id": "s002", "line": 9, "kind": "parallel",
      "modifiers": {"join": "all", "on_fail": "fail-fast",
                    "count": null, "max_concurrent": null},
      "branches": ["s003", "s004", "s005"]
    },
    {
      "id": "s003", "line": 10, "kind": "session", "agent": "reviewer",
      "binding": "security", "prompt_hash": "sha256:16b0…",
      "context": [], "is_output": false
    }
  ],
  "outputs": {},
  "diagnostics": [],
  "hash_algorithm": "sha256",
  "content_hash": "sha256:9c41…"
}
```

### Fields

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `v` | string | Always `"openprose.compile-ir.v1"` |
| `program` | string | Source path as invoked (display identity) |
| `source.path` | string | Path of the compiled file |
| `source.content_hash` | string | `sha256:` over the source file's exact bytes — the freshness anchor |
| `state_backend` | string | Resolved backend (`filesystem` default) |
| `inputs` | object | `{name: description}` for every `input` declaration |
| `agents` | object | Agent table — see below |
| `blocks` | object | Block table — `{name: {"params": [...], "statements": ["sNNN", …]}}` |
| `statements` | array | The full statement inventory — see below |
| `outputs` | object | `{output name: "sNNN"}` — statement that declares each program output |
| `diagnostics` | array | Compiler findings: `{"severity", "code" (optional), "line", "message"}` — errors here mean the program does not compile; the IR still records why |
| `hash_algorithm` | string | Always `"sha256"` |
| `content_hash` | string | Content address of this IR — over the canonical form of everything except `content_hash` itself |

### Agent table entries

`{model, prompt_hash, persist, skills, retry, backoff}` — `prompt_hash` is
`sha256:` over the exact prompt text (UTF-8 bytes of the string content,
quotes excluded, indentation as written). Unset properties are JSON
`null`; `skills` defaults to `[]`. Additional properties the compiler
resolved (e.g. `permissions`) may be included; consumers ignore unknown
keys.

### Statement entries

Every statement gets an entry, in **source order**, with:

- `id` — per `openprose.statement-id.v1` (`contracts/receipt.md`).
  **Static base IDs only** (`s001`, `s002`, …): the IR describes the
  program, not an execution — dynamic suffixes (`.x/.i/.b/.d`) appear
  only in receipts. IDs are dense and ascending from `s001`.
- `line` — 1-based source line where the statement starts.
- `kind` — one of: `use`, `import` (unspecified construct, carried
  as-written), `input`, `output`, `agent_def`, `block_def`, `session`,
  `resume`, `let`, `const`, `assignment`, `parallel`, `repeat`, `for`,
  `loop`, `if`, `elif`, `else`, `choice`, `option`, `try`, `catch`,
  `finally`, `do`, `throw`, `other`.
- Kind-specific fields (all optional, `null`/absent when inapplicable):
  `name` (definitions), `binding` (the variable a `let`/branch binds),
  `agent` (session's resolved agent or `null` for anonymous),
  `prompt_hash`, `context` (list of names wired in), `is_output`
  (declares a program output), `branches` / `body` (lists of statement
  IDs), `modifiers` (parallel/loop options), `condition_count` (number
  of discretion markers in the statement).

The statement graph is **structural, not semantic**: it records what is
wired where, never prompt text or values (hashes only — the IR must be
safe to commit and cheap to diff).

## Canonical form and hashing

Identical to the receipt contract (`contracts/receipt.md`): sorted keys,
no whitespace, UTF-8 strings unescaped, **integers only** (floats are
invalid), `content_hash = "sha256:" + hex(sha256(canonical(ir sans
content_hash)))`. One IR compiled twice from identical source by
different compilers should differ only where genuine semantic judgment
differs — and never in formatting.

## Freshness (`compile --check`)

An IR is **fresh** iff `source.content_hash` equals the sha256 of the
current source bytes. `openprose-tools ir-check <file.prose>`:

- exit 0 — IR exists, valid, fresh
- exit 1 — IR missing, schema-invalid, internally inconsistent, or stale
- exit 2 — source unreadable

`prose compile --check` (SKILL.md) is the embodied alias: it runs
`ir-check` when tooling is available and reports; it never silently
recompiles.

## Validation rules (`ir-check`)

1. `v` known; JSON object; schema-valid (unknown fields ignored).
2. `content_hash` recomputes over the canonical form.
3. `source.content_hash` matches the current source file (freshness).
4. Statement IDs: well-formed base IDs, unique, dense ascending from
   `s001`; no dynamic suffixes.
5. Reference integrity: every `agent` in a statement exists in `agents`;
   every ID in `branches`/`body`/`blocks.*.statements`/`outputs` exists
   in `statements`.
6. `diagnostics` containing `severity: "error"` → the IR is a record of
   a failed compile; `ir-check` exits 1 (a broken program must not pass
   the gate just because its IR honestly says it is broken).

## Versioning

Same policy as receipts: fields may be added under a bumped `v`; never
renamed, retyped, or reinterpreted. Consumers MUST ignore unknown fields
and refuse unknown `v` values.
