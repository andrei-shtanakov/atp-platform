#!/usr/bin/env python3
"""Generate `DIGESTS.json` — the machine-readable half of the score contract.

The handoff document tells a human what the contract is; its pins are now
recomputed by `TestHandoffPinsAreRecomputed` (#298/#299), so the prose can no
longer drift from the bytes. This file closes the *other* half, which stayed
manual and lived on the consumer's side.

maestro vendors the fixtures with a `PIN` and guards two things separately:
copy-integrity (their bytes still match their pins) and upstream-drift (we have
not moved past the pinned commit). The second one could only be checked against
a sibling checkout of this repo, and **skipped** when there was none — so for an
installed user it degraded into `@trigger` prose, exactly the defect #298 fixed
here. A published digest map turns that check into "download one file and
compare", with no checkout and no skip (maestro#204).

Two format decisions came from the consumer and are worth stating:

* **`score_contract.py` is in the map.** It is not a fixture, but a fixture that
  drifts from its parser surfaces on their side as a failure on a live run
  rather than a red test. Pinning the payloads without the code that defines
  their meaning would leave the sharper failure unguarded.
* **No commit SHA.** The obvious field to add, and it cannot be filled: at the
  moment this runs, the commit that will carry the file does not exist yet. A
  field that is always stale or always empty is worse than an absent one.

Regenerate with:

    uv run python scripts/write_score_contract_digests.py

The sidecar is asserted against a fresh recomputation in
`tests/unit/dashboard/test_score_contract.py`, so forgetting to regenerate is a
red test, not a silent lie.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "benchmark_score_contract"
SIDECAR = FIXTURES_DIR / "DIGESTS.json"

#: The contract module the fixtures are payloads of. Repo-relative, because a
#: bare basename is ambiguous in this repo and the consumer resolves paths, not
#: names (their vendoring test already carries the upstream path per file).
CONTRACT_MODULE = "packages/atp-dashboard/atp/dashboard/benchmark/score_contract.py"

#: The fixture whose `score_semantics` block is the canonical one — a real
#: completion-only payload, `coverage` included. Kept identical to the constant
#: the handoff test uses; the two are asserted equal there.
CANONICAL_SOURCE = "run_status_completion_only.json"

#: Bumping this means the shape below changed, not the contract it describes.
SIDECAR_FORMAT_VERSION = 1


def fixture_names() -> list[str]:
    """Payload fixtures, sorted — the sidecar itself is not one of them.

    `DIGESTS.json` lives beside the fixtures it describes, so every listing of
    "the fixtures" has to exclude it explicitly. Anything that globs `*.json`
    here and forgets will either try to read a payload out of the sidecar or
    demand a pin for it.
    """
    return sorted(
        path.name for path in FIXTURES_DIR.glob("*.json") if path.name != SIDECAR.name
    )


def canonical_semantics_digest() -> str:
    """sha256 of the canonical `score_semantics`: sorted keys, no whitespace."""
    payload = json.loads((FIXTURES_DIR / CANONICAL_SOURCE).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload["score_semantics"], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_sidecar() -> dict[str, Any]:
    """The sidecar as it must be on disk, computed from the bytes on disk."""
    from atp.dashboard.benchmark.score_contract import SCORE_CONTRACT_VERSION

    files = {
        f"tests/fixtures/benchmark_score_contract/{name}": hashlib.sha256(
            (FIXTURES_DIR / name).read_bytes()
        ).hexdigest()
        for name in fixture_names()
    }
    files[CONTRACT_MODULE] = hashlib.sha256(
        (REPO_ROOT / CONTRACT_MODULE).read_bytes()
    ).hexdigest()

    return {
        "_generated_by": "scripts/write_score_contract_digests.py — do NOT hand-edit",
        "sidecar_format_version": SIDECAR_FORMAT_VERSION,
        "contract_version": SCORE_CONTRACT_VERSION,
        "canonical_score_semantics_sha256": canonical_semantics_digest(),
        "files": dict(sorted(files.items())),
    }


def render(sidecar: dict[str, Any]) -> str:
    """Stable bytes: sorted keys, trailing newline, so a no-op run is a no-op diff."""
    return json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    text = render(build_sidecar())
    previous = SIDECAR.read_text(encoding="utf-8") if SIDECAR.is_file() else None
    SIDECAR.write_text(text, encoding="utf-8")
    print(f"{'unchanged' if previous == text else 'wrote'} {SIDECAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
