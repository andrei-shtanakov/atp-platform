# Correctness Rule KB (`code-review-correctness`)

Reference rules cited by the `code-review-correctness` case family. Each rule is
also embedded inline in the `kb-rules` artifact of every case that cites it, so
cases remain self-contained.

- **LOG-001** — boundary/off-by-one comparisons must keep exactly the intended count.
- **LOG-002** — conditionals/predicates must correctly express the intended access/validity logic.
- **SPEC-001** — monetary amounts are integer cents.
- **SPEC-002** — list endpoints must paginate, with limit ≤ 100.

> False-positive (`F*`) cases cite no rule: the point is to NOT flag compliant code.
