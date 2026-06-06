# Glossary

Shared vocabulary for the agent evaluation framework. Where a term maps directly to a field in
`agent-eval-case.schema.json`, the field name is shown in `code`. See `METHODOLOGY.md` for how
the pieces fit together.

---

**Action surface** (`construction_axis: action_surface`)
The construction axis that varies what the agent is allowed to *do* — from read-only tasks to
tasks with side effects. Closely tied to `side_effects`.

**Adaptation** (`capability: adaptation`)
The capability to revise already-completed work when requirements change mid-task, rather than
applying the new rule only to subsequent steps.

**Adversarial environment** (`construction_axis: adversarial_environment`)
The construction axis that introduces hostile elements: prompt injection, misleading
fragments, or tempting shortcuts that lead to a wrong result. Tests whether the agent
distinguishes data from instructions and quality from noise.

**Axis level** (`axis_level`)
The ordinal position of a case on its sweep, from easiest to hardest:
`clean → mild → moderate → severe`. The level at which the `critical_check` starts failing is
the point of collapse.

**Calibration** (`capability: calibration`)
The capability to know the limits of one's own knowledge and to report blockers honestly
instead of confabulating. The primary defence against hallucination.

**Capability** (`capability`)
A measurement axis — the quality a case is designed to score. One of: `correctness`,
`calibration`, `efficiency`, `safety_compliance`, `recoverability`, `adaptation`.

**Case** (a.k.a. instance)
The atomic unit of evaluation: one input task plus its grading logic. Validated against the
schema; typically authored as one YAML file.

**Confabulation**
Producing plausible but fabricated content to fill a gap (an invented figure, a non-existent
deadline) instead of reporting that the information is absent. The classic target of
`calibration` cases.

**Construction axis** (`construction_axis`)
A property of the *task* that is varied to build a sweep. One of: `information_conditions`,
`horizon_autonomy`, `action_surface`, `adversarial_environment`, `requirements_volatility`,
`output_structure`. Distinct from a capability, which is what you measure.

**Critical check** (`grader.critical_check`)
A single binary, must-pass assertion bound to the `expected_failure_mode`. Separated from the
rubric on purpose: it is the guard that decides whether the specific trap was avoided. If it
fails, the case fails regardless of rubric score.

**Distractor** (`distractor`)
The element of a case that pressures the agent toward the failure mode — pattern pressure, a
seductive shortcut, or an embedded instruction. Without a distractor a trap rarely springs.

**Efficiency** (`capability: efficiency`)
The capability to reach a correct result at low cost — measured in steps, tokens, time, or
money. "Performance" in the narrow sense.

**Expected failure mode** (`expected_failure_mode`)
The precise, anticipated way a weak agent is expected to fail on this case. Stated as one
concrete sentence, not a generality. The core around which grading is built.

**Family**
A parametrized template for one capability that expands into a sweep of cases varying along a
single construction axis. The real working unit of authoring — you write families, not
individual cases.

**Gold** (`grader.gold`)
A reference answer (path or inline) against which the response is compared. Required for
`exact` graders.

**Grader** (`grader`)
The block describing how a run is scored: its `type`, optional `gold` and `rubric`, the
mandatory `critical_check`, and the `scoring` aggregation rule.

**Grader type** (`grader.type`)
How scoring is performed: `exact`, `regex`, `programmatic`, `rubric`, `model_graded`, or
`human`. Prefer the most automatic type a case allows.

**Held-out suite** (`suite_type: held_out`)
A suite hidden during agent development so the agent cannot be tuned to the test. Guards
against overfitting to the evaluation itself.

**Horizon / autonomy** (`construction_axis: horizon_autonomy`)
The construction axis that varies task length and how much the agent must decompose and hold
state on its own. Tests memory and plan stability over a long chain.

**Information conditions** (`construction_axis: information_conditions`)
The construction axis spanning complete-and-consistent input through incomplete,
contradictory, or noisy input. Where uncertainty lives.

**Inject** (`turns[].role: inject`)
A scripted turn that introduces a new constraint *after* the agent has begun working. The
mechanism behind `requirements_volatility` cases.

**Instance**
Synonym for case — a single concrete input plus its grader.

**Measurement axis**
A synonym-level grouping for the capabilities; the planes along which result quality is scored,
as opposed to construction axes which generate tasks.

**Point of collapse**
The `axis_level` on a sweep at which the agent begins to fail the `critical_check`. The headline
signal a sweep is designed to surface.

**Probe suite** (`suite_type: probe`)
A mutable suite whose job is to hunt for new failure modes. Expected to churn, unlike a
regression suite.

**Provenance** (`provenance`)
Origin metadata for a case: author, creation date, and source of the input material.

**Recoverability** (`capability: recoverability`)
The capability to notice one's own error and correct it, rather than entrench it in subsequent
steps.

**Regression suite** (`suite_type: regression`)
A frozen suite re-run on every agent version to catch quality regressions.

**Rubric** (`grader.rubric`)
Weighted, graded criteria for the fuzzy aspects of quality. Weights should sum to 1.0. Measures
a gradient, in contrast to the binary `critical_check`.

**Run**
One execution of a suite against a specific agent version, producing a scorecard.

**Safety / compliance** (`capability: safety_compliance`)
The capability to follow scope rules and to refuse prohibited or out-of-bounds actions.

**Scorecard**
The output of a run: the per-case and aggregate results for one agent version.

**Side effects** (`environment.side_effects`)
The reversibility of the actions a case permits: `none`, `reversible`, or `irreversible`.
Drives the safety class of the case.

**Suite**
A named collection of families/cases grouped by purpose. Its `suite_type` governs how it is
used.

**Suite type** (`suite_type`)
The governance role of a suite: `regression`, `probe`, or `held_out`.

**Sweep**
A series of cases from one family at increasing `axis_level`, run together to locate the point
of collapse.

**Turn** (`turns[]`)
One step in a multi-turn interaction script. Roles: `user` (normal input), `inject` (a new
mid-task constraint), `assistant` (a scripted prior agent turn).
