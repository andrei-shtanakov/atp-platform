# Run State — 20260716-083117-6pgtu7

**Program:** examples/30-captains-chair-simple.prose
**State backend:** filesystem
**Input:** `task` = "Write a one-paragraph project description for the
OpenProse receipt ledger feature (an append-only, hash-chained audit trail
for AI orchestration runs)." (`bindings/task.md`)
**Status:** completed

## Execution

```prose
input task: "What to accomplish"                   # s001 -> bindings/task.md

agent captain:  ...                                # s002 (definition)
agent executor: ...                                # s003 (definition)
agent critic:   ...                                # s004 (definition)

let plan = session: captain                        # s005 ✓ -> bindings/plan.md
  prompt: "Break down this task into work items: {task}"

parallel:                                          # s006 ✓ joined (all, 2/2)
  work = session: executor                         # s007 ✓ -> bindings/work.md
    prompt: "Execute the plan"
    context: plan
  review = session: critic                         # s008 ✓ -> bindings/review.md
    prompt: "Identify potential issues with this approach"
    context: plan

let synthesis = ""                                 # s009 (declare-once)
if **critic found issues that affect the work**:   # s010; discretion s010.d1 -> TRUE
  synthesis = session: captain                     # s011 ✓ -> bindings/synthesis.md
    prompt: "Integrate the work while addressing critic's concerns"
    context: { work, review }
else:
  synthesis = session: captain                     # s012 (not executed)
    prompt: "Validate and summarize the completed work"
    context: { work, review }

output result = synthesis                          # s013 ✓ (control receipt)
```

## Discretion evaluation (s010.d1)

Condition: `**critic found issues that affect the work**` → **TRUE**.
Basis: the critic flagged a missing merge/assembly step and an
invented-scope risk, and the executor's paragraph overstated detectability
("immediately detectable" without the trusted-head caveat). The if-branch
ran; the captain's synthesis tempered the claim to chain consistency
verifiable against a trusted `ledger_head` anchor.

The outcome is recorded in the `s010.d1` receipt (`detail`) — on replay,
substitute it instead of re-evaluating.

## Notes

- This run also exercises the Phase 0 fix to this example (declare-once +
  reassign instead of dual `output result` in both branches).
- Token usage basis: **estimated** — substrate reports per-subagent totals
  without an input/output split; output estimated from binding size.
- Receipt ledger: 9 receipts (run_start, s005, 2 branches, join,
  discretion, s011, output_registered, run_end), anchored in `run.json`.
