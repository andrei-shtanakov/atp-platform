"""Shared, spawner-agnostic output envelopes for the agent-eval-case methodology.

The envelope is the output contract handed to a spawner — it belongs to the
capability, NOT to any one spawner (ADR-006 seam #2). Keeping it here, in the
installed package, lets every shim import the SAME envelope (the shim subprocess
runs under the venv interpreter, so atp_method is importable) instead of one shim
importing another — which removes the N×M drift and keeps the API-vs-CLI ablation
equivalent by construction.
"""

import json
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

GENERIC_ENVELOPE = (
    "Output ONLY the answer in the exact format requested below, with no prose "
    "and no markdown fence.\n\n{task}"
)

# capability -> envelope. One entry today; a new capability adds one here.
_ENVELOPES: dict[str, str] = {"review": REVIEW_ENVELOPE}


def get_envelope(capability: str = "review") -> str:
    """Return the output envelope for a capability. Raises KeyError if unknown."""
    return _ENVELOPES[capability]


def build_prompt(request: dict[str, Any], envelope: str) -> str:
    """Wrap an ATPRequest's task + inline artifacts in an envelope.

    Artifacts (the diff, the rules) live under ``task.input_data["artifacts"]``
    — that is where the loader emits them and how the CLI adapter serializes the
    request. The ATP ``Context`` model has no artifacts field, so the earlier
    ``context.artifacts`` read always came up empty, silently handing the model
    an empty review.

    When the case carries an ``output_contract.format_instruction`` (in
    ``task.input_data``), use the generic envelope + that instruction so a
    non-review capability is not forced through the review findings envelope.
    Otherwise fall back to the passed-in envelope (review).
    """
    task = request.get("task") or {}
    body = task.get("description", "")
    input_data = task.get("input_data") or {}
    artifacts = input_data.get("artifacts", []) or []
    for art in artifacts:
        if art.get("content"):
            body += f"\n\n--- {art.get('id', 'artifact')} ---\n{art['content']}"
    corpus = input_data.get("artifact_corpus") or {}
    if input_data.get("run_mode") == "read_only_corpus" and corpus:
        corpus_id = corpus.get("id", "corpus")
        paths = corpus.get("files") or []
        if paths:
            path_list = "\n".join(f"- {path}" for path in paths)
            body += (
                "\n\nRead-only corpus files are available to your "
                "file-reading tool. Cite source paths relative to the "
                f"corpus root. Corpus id: {corpus_id}. Available paths:\n"
                f"{path_list}"
            )
    contract = input_data.get("output_contract") or {}
    instruction = contract.get("format_instruction")
    if instruction:
        schema = contract.get("schema")
        schema_text = (
            f"\n\nResponse JSON Schema:\n{json.dumps(schema, indent=2, sort_keys=True)}"
            if schema
            else ""
        )
        return GENERIC_ENVELOPE.format(task=f"{body}\n\n{instruction}{schema_text}")
    return envelope.format(task=body)
