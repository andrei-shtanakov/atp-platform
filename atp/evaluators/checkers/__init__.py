"""Deterministic checker registry + built-in registrations (Phase A-1)."""

from atp.evaluators.checkers.registry import (
    Checker,
    get_checker,
    list_checkers,
    register_checker,
)
from atp.evaluators.citation_grounding.checker import citation_grounding_check
from atp.evaluators.findings.checker import findings_check
from atp.evaluators.json_path.checker import json_path_check
from atp.evaluators.openprose_receipts.checker import receipt_chain_check

register_checker("citation_grounding", citation_grounding_check)
register_checker("findings_match", findings_check)
register_checker("json_path", json_path_check)
register_checker("receipt_chain", receipt_chain_check)

__all__ = [
    "Checker",
    "get_checker",
    "list_checkers",
    "register_checker",
]
