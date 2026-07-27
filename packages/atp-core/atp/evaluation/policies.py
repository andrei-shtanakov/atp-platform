"""Named evaluation policies, one per execution context.

A policy answers one question — which assertion types may run here — and it is
answered by the *server*, never by the submission. Nothing in a suite can
widen it: `EvaluationPipeline` consults the policy before it resolves an
evaluator, so a disallowed one is never even constructed.

Two contexts exist today.

`TRUSTED_LOCAL` is the CLI: the operator runs their own suite on their own
machine, so withholding evaluators would only get in the way.

`UNTRUSTED_SUBMISSION` is the benchmark plane, where the input arrives from a
self-service token holder. It admits only evaluators that inspect the response
and the trace. Excluded, and why:

* **code execution** (`pytest`, `npm`, `lint`, `code_exec`, `custom_command`) —
  runs code derived from the submission inside the API process;
* **network calls** (`llm_eval`, `factuality`) — spends money per submission
  and makes the score non-reproducible;
* **host filesystem reads** (`file_exists`, `file_contains`, …) — the
  directory comes from the suite, not from the sandbox the submission was
  materialized into, so the check measures something unrelated and answers a
  question about the server's own disk while doing it.

The first two are classes this plane will eventually run elsewhere, and they
stay excluded until there is an isolated worker to run them in: durable queue,
resource and time limits, network policy, cost budget, idempotency,
cancellation, audit trail. Until every one of those exists, "temporarily allow
it, we'll add limits later" is how an API process ends up executing strangers'
code. See ADR-008 tracks C and D.

`filesystem` is a different kind of exclusion — not dangerous *machinery* but
a wrong *interface*. It needs a sandbox it is given rather than one it is told
about, which is ADR-008 track B and needs no worker at all.

`composite` used to be excluded here for a third reason and no longer is. It
resolved its leaves through the global registry, so nesting `pytest` under it
walked straight past this policy; it now builds them through the resolver it
is handed, so its leaves are filtered by whatever policy governs it and a
withheld leaf is reported as unevaluated rather than as a failure (ADR-008
track A).

The allowlist is derived from the behaviour classification in
:mod:`atp.evaluation.vocabulary` rather than typed out here. A hand-kept list
would need updating every time an evaluator is added, and the failure mode of
forgetting is that something executing quietly becomes permitted.
"""

from __future__ import annotations

from atp.evaluation.pipeline import EvaluationPolicy
from atp.evaluation.vocabulary import deterministic_assertion_types

#: The CLI: the operator's own suite, on the operator's own machine.
TRUSTED_LOCAL = EvaluationPolicy(name="trusted_local")

#: The benchmark plane: input from a self-service token holder.
UNTRUSTED_SUBMISSION = EvaluationPolicy(
    name="untrusted_submission",
    allowed_assertion_types=deterministic_assertion_types(),
)
