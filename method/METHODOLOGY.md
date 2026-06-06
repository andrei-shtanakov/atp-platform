# Agent Evaluation Methodology

A framework for building evaluation tasks that actually discriminate between AI agents,
rather than tasks on which every agent looks equally successful.

This document describes the *why* and the *how* in prose. The machine-readable contract
lives in `agent-eval-case.schema.json`; the vocabulary lives in `GLOSSARY.md`.

---

## 1. The core problem

On easy, fully-specified tasks, almost every competent agent succeeds. Such tasks tell you
nothing — they don't separate a strong agent from a weak one. A useful evaluation task does
two things that an ordinary task does not:

1. It carries a **trap** — a specific, anticipated way a weak agent fails.
2. It belongs to a **sweep** — a series of variants of increasing difficulty, so you can
   locate the *point of collapse* instead of getting a single pass/fail at an arbitrary
   difficulty.

Everything below exists to make those two ideas operational and repeatable.

---

## 2. The unit of work: the case

A **case** is the atom: one input task plus the grading logic to score the agent's response.
Cases are never written one at a time by hand at scale. Instead you author a **family** — a
parametrized template for one capability — and let it expand into a sweep of cases that vary
along a single axis.

The mental shift: you don't write a thousand tests, you write a few dozen families, each of
which unfolds into a sweep.

```
suite  ──contains──▶  families  ──expand into──▶  cases (instances)
                                                      │
                                                  run against
                                                  an agent ▶ scorecard
```

---

## 3. The trap is the heart of the case

Three fields, taken together, are what make a case discriminating. If any is vague, the case
degrades into "rate the beauty of the answer" and stops separating agents.

- **expected_failure_mode** — the precise way a weak agent is expected to fail. Not "gives a
  bad answer", but "invents a plausible deadline for the requirement that has none".
- **distractor** — the element that *pressures* the agent toward that failure: pattern
  pressure, a tempting shortcut that yields a wrong result, or an instruction embedded in the
  input data.
- **critical_check** — a single, binary, must-pass assertion bound to the failure mode. It is
  deliberately separated from the graded rubric. The rubric measures quality on a gradient;
  the critical_check is the guard that says "this specific trap was avoided". A case fails if
  the critical_check fails, regardless of how good the rest of the answer looks.

---

## 4. Two families of axes

A frequent confusion is to mix "what you measure" with "what you vary". They live in different
planes. Keeping them separate is what makes the framework orthogonal.

### Construction axes — what you *vary* in the task

These generate the diversity of tasks. A sweep moves along exactly one of them.

| Construction axis          | What it varies                                                        |
|----------------------------|-----------------------------------------------------------------------|
| `information_conditions`   | Complete & consistent input → incomplete, contradictory, or noisy     |
| `horizon_autonomy`         | Single step → long chain the agent must decompose and hold state for  |
| `action_surface`           | Read-only → actions with side effects (write, send, irreversible)     |
| `adversarial_environment`  | Clean input → prompt injection, misleading fragments, tempting shortcuts |
| `requirements_volatility`  | Stable spec → a new constraint injected mid-task                       |
| `output_structure`         | Free form → a strict schema with naming conventions                   |

### Measurement axes (capabilities) — what you *score*

These don't generate tasks; they define what quality you extract from a run.

| Capability          | The question it answers                                              |
|---------------------|---------------------------------------------------------------------|
| `correctness`       | Is the result right, against a gold answer or rubric?               |
| `calibration`       | Does the agent know the limits of its knowledge and report blockers instead of confabulating? |
| `efficiency`        | At what cost — steps, tokens, time?                                  |
| `safety_compliance` | Does it respect scope rules and refuse prohibited actions?          |
| `recoverability`    | Does it notice its own error and correct it, rather than entrench it? |
| `adaptation`        | Does it revise completed work when requirements change?             |

---

## 5. Families and sweeps: the curve of collapse

A family fixes one capability, then produces cases at increasing `axis_level`:
`clean → mild → moderate → severe`. You run the whole sweep against an agent and watch where
the `critical_check` begins to fail. **That point of collapse is the signal** — far more
informative than a single binary verdict at a random difficulty.

Example: the `req-extraction` family tests `calibration` along `information_conditions`.
- `clean` — every requirement has a clear deadline.
- `severe` — one requirement has no deadline at all, while all its neighbours do (the
  distractor). A well-calibrated agent marks it absent; a weak one fabricates one.

---

## 6. Suites and governance

Cases are grouped into suites by purpose. Separating these from day one prevents the set from
silently overfitting to itself.

- **`regression`** — frozen. Re-run on every agent version to catch quality regressions.
- **`probe`** — mutable. Hunts for new failure modes; expected to churn.
- **`held_out`** — hidden during agent development, so the agent can't be tuned to the test.

---

## 7. How a case is graded

Grading combines two layers, by design:

1. **critical_check** (binary) — the guard tied to the failure mode. Must pass.
2. **rubric** (graded) — weighted criteria for the fuzzy aspects of quality; weights sum to 1.0.

The `scoring` field states the aggregation rule. The standard rule: *if the critical_check
fails, the case fails outright; otherwise the score is the weighted rubric sum.*

Grader types range from fully automatic to human: `exact`, `regex`, `programmatic`, `rubric`,
`model_graded`, `human`. Prefer the most automatic type the case allows — it makes the suite
cheap to re-run.

---

## 8. How the platform consumes this

The package is a versioned, file-based registry — one case per file, validated against the
schema. Suggested layout:

```
/                       repository root
  agent-eval-case.schema.json     # the contract every case must satisfy
  METHODOLOGY.md                  # this file
  GLOSSARY.md                     # shared vocabulary
  cases/                          # one file per instance, grouped by family
    req-extraction/
      case-req-extract-...-clean-001.yaml
      case-req-extract-...-severe-001.yaml
  gold/                           # reference answers referenced by graders
  0. archive/                     # superseded cases — excluded from runs and analysis
```

Cases are YAML for human authoring; they conform to the JSON Schema. The schema enforces the
structural integrity of the trap (e.g. a volatility case must script an injected turn; a
rubric grader must carry a rubric). Two things the schema deliberately does *not* check, left
to an authoring linter:

- that rubric weights sum to 1.0;
- that the `critical_check` semantically matches the `expected_failure_mode` (a human or
  model-graded acceptance review).

---

## 9. Authoring a new case — the short version

1. Pick one **capability** to put under the lens.
2. Pick one **construction axis** to vary, and the **axis_level** for this instance.
3. Write the task and its input artifacts. Plant the **distractor**.
4. State the **expected_failure_mode** in one concrete sentence.
5. Write the **critical_check** that detects exactly that failure — binary, must-pass.
6. Add a **rubric** for graded quality where it matters.
7. Assign the **suite_type** (`regression` / `probe` / `held_out`).
8. Validate against the schema; submit to the registry.

A case that cannot articulate a sharp failure mode and a binary check for it is not yet ready
to be a case.
