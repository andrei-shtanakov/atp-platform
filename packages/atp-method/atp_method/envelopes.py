"""Shared, spawner-agnostic output envelopes for the agent-eval-case methodology.

The envelope is the output contract handed to a spawner — it belongs to the
capability, NOT to any one spawner (ADR-006 seam #2). Keeping it here, in the
installed package, lets every shim import the SAME envelope (the shim subprocess
runs under the venv interpreter, so atp_method is importable) instead of one shim
importing another — which removes the N×M drift and keeps the API-vs-CLI ablation
equivalent by construction.
"""

from typing import Any

# Pinned model for the code-review vertical (override per shim via CLAUDE_MODEL).
# Shared so both spawners pin the SAME model (the ablation's equivalence guard).
DEFAULT_MODEL = "claude-opus-4-8"

REVIEW_ENVELOPE = (
    "You are a senior code reviewer. Review the material below. Output ONLY a JSON "
    "array of findings (no prose, no markdown fence). Each finding is an object with "
    'keys: "rule_id" (the rule/CWE id), "file", "anchor" (the exact offending code '
    'substring), "severity" (critical|major|minor), "fix". If the code is compliant, '
    "output an empty array [].\n\n{task}"
)

# capability -> envelope. One entry today; a new capability adds one here.
_ENVELOPES: dict[str, str] = {"review": REVIEW_ENVELOPE}


def get_envelope(capability: str = "review") -> str:
    """Return the output envelope for a capability. Raises KeyError if unknown."""
    return _ENVELOPES[capability]


def build_prompt(request: dict[str, Any], envelope: str) -> str:
    """Wrap an ATPRequest's task + inline artifacts in the given envelope.

    Artifacts (the diff, the rules) live under ``task.input_data["artifacts"]``
    — that is where the loader emits them and how the CLI adapter serializes the
    request. The ATP ``Context`` model has no artifacts field, so the earlier
    ``context.artifacts`` read always came up empty, silently handing the model
    an empty review.
    """
    task = request.get("task") or {}
    body = task.get("description", "")
    artifacts = (task.get("input_data") or {}).get("artifacts", []) or []
    for art in artifacts:
        if art.get("content"):
            body += f"\n\n--- {art.get('id', 'artifact')} ---\n{art['content']}"
    return envelope.format(task=body)
