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
  and makes the score non-reproducible.

Both are classes this plane will eventually run elsewhere, and they stay
excluded until there is an isolated worker to run them in: durable queue,
resource and time limits, network policy, cost budget, idempotency,
cancellation, audit trail. Until every one of those exists, "temporarily allow
it, we'll add limits later" is how an API process ends up executing strangers'
code. See ADR-008 tracks C and D.

Two evaluators used to be excluded here and no longer are, in both cases
because the interface was fixed rather than the policy loosened.

`composite` resolved its leaves through the global registry, so nesting
`pytest` under it walked straight past this policy; it now builds them through
the resolver it is handed, so its leaves are filtered by whatever policy
governs it and a withheld leaf is reported as unevaluated rather than as a
failure (ADR-008 track A).

`filesystem` took its root from the suite, so it inspected a directory of the
server's own instead of the sandbox the submission was materialized into — and
answered questions about that directory to whoever ran the benchmark. It is
now given its root by the composition and can address nothing outside it
(ADR-008 track B), which means on this plane it inspects the submission and
only the submission.

The allowlist is derived from the behaviour classification in
:mod:`atp.evaluation.vocabulary` rather than typed out here. A hand-kept list
would need updating every time an evaluator is added, and the failure mode of
forgetting is that something executing quietly becomes permitted.
"""

from __future__ import annotations

from atp.evaluation.pipeline import EvaluationPolicy
from atp.evaluation.vocabulary import deterministic_assertion_types

#: The CLI: the operator's own suite, on the operator's own machine.
TRUSTED_LOCAL = EvaluationPolicy(name="trusted_local", trusted=True)

#: The benchmark plane: input from a self-service token holder.
UNTRUSTED_SUBMISSION = EvaluationPolicy(
    name="untrusted_submission",
    allowed_assertion_types=deterministic_assertion_types(),
)
