# Agent Evaluation Suite

A file-based registry of evaluation cases for AI agents working on business-context tasks. The
package is designed to be picked up by the evaluation platform directly: every case is a file
that conforms to a single JSON Schema, so the platform can discover, validate, and run cases
without bespoke parsing.

## What this is for

Most tasks fail to separate a strong agent from a weak one, because on easy, well-specified work
almost every competent agent succeeds. The cases here are built around two ideas that make them
discriminating: each carries a **trap** (a specific, anticipated failure mode plus a binary
check that detects it) and belongs to a **sweep** (a series of variants of growing difficulty,
so you can locate the point where an agent collapses rather than getting a single pass/fail at
an arbitrary difficulty). The full rationale is in the methodology.

## Package contents

| File | Purpose |
|------|---------|
| `agent-eval-case.schema.json` | The contract. Every case is validated against this (JSON Schema, draft 2020-12). |
| `METHODOLOGY.md` / `METHODOLOGY.ru.md` | The framework explained in prose — axes, traps, sweeps, governance. |
| `GLOSSARY.md` / `GLOSSARY.ru.md` | Shared vocabulary, each term mapped to its schema field. |
| `CASE_GENERATOR.md` | System instruction for an LLM agent that authors new cases. |
| `cases/` | The case registry, one YAML file per case, grouped by family. |
| `gold/` | Reference answers referenced by graders. |
| `0. archive/` | Superseded cases — excluded from runs and from any analysis. |

## Repository layout

```
/
  agent-eval-case.schema.json
  README.md
  METHODOLOGY.md            METHODOLOGY.ru.md
  GLOSSARY.md               GLOSSARY.ru.md
  CASE_GENERATOR.md
  cases/
    req-extraction/
      case-req-extraction-fabricated-deadline-clean-001.yaml
      case-req-extraction-fabricated-deadline-moderate-001.yaml
      case-req-extraction-fabricated-deadline-severe-001.yaml
  gold/
  0. archive/
```

## Key concepts in one minute

- **Case** — the atomic unit: one input task plus its grading logic. Authored as one YAML file.
- **Family** — a parametrized template for one capability; expands into a sweep.
- **Sweep** — cases from one family at increasing `axis_level` (`clean → mild → moderate →
  severe`). The level where the binary check starts failing is the signal.
- **Construction axis** — what the task *varies* (e.g. `information_conditions`).
- **Capability** — what the case *scores* (e.g. `calibration`). Distinct from the axis.
- **critical_check** — the binary, must-pass guard tied to the failure mode; separate from the
  graded rubric. A case fails if this fails, regardless of rubric score.
- **Suite type** — governance role: `regression` (frozen), `probe` (mutable), `held_out`
  (hidden during agent development).

## The example sweep

`cases/req-extraction/` ships a complete, schema-valid sweep that doubles as a template. The
family scores `calibration` along `information_conditions`; the shared trap is fabricating a
value absent from the source. Pressure grows across the three levels: an explicit deadline
(`clean`), a vague qualifier that tempts quantification (`moderate`), and a missing deadline
surrounded by present ones (`severe`).

## Validating cases

Every case must pass the schema. The schema enforces structure and the integrity of the trap
(for example, a `requirements_volatility` case must include an injected turn; a rubric grader
must carry a rubric). Two checks are **not** expressible in JSON Schema and belong to an
authoring linter or review:

- rubric weights must sum to 1.0;
- the `critical_check` must semantically match the `expected_failure_mode`.

A minimal validation pass, for reference:

```python
import glob, yaml, json
from jsonschema import Draft202012Validator
schema = json.load(open("agent-eval-case.schema.json"))
v = Draft202012Validator(schema)
for f in glob.glob("cases/**/*.yaml", recursive=True):
    doc = yaml.safe_load(open(f))
    errs = list(v.iter_errors(doc))
    print(f, "VALID" if not errs else [e.message for e in errs])
```

Cases are authored in YAML for readability; they are validated as the JSON the schema describes.
Quote ISO dates (`created: "2026-06-06"`) so YAML does not parse them as date objects.

## Language policy

English is the source of truth for all artifacts. `METHODOLOGY` and `GLOSSARY` also ship a
Russian variant (`.ru.md`); when the English changes, update the Russian to match. Field names,
enum values, and tag values are always English, including inside the Russian documents.

## Authoring a new case

See `CASE_GENERATOR.md` for the full procedure and a worked example. In short: pick one
capability and one construction axis, invent a plausible business surface story, design the trap
once for the whole family, produce the sweep, validate against the schema, and run the
self-check. A case that cannot state a sharp failure mode and a binary check for it is not yet
ready.
