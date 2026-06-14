"""Deterministic checker registry + built-in registrations (Phase A-1)."""

from atp.evaluators.checkers.registry import (
    Checker,
    get_checker,
    list_checkers,
    register_checker,
)
from atp.evaluators.findings.checker import findings_check

register_checker("findings_match", findings_check)

__all__ = [
    "Checker",
    "get_checker",
    "list_checkers",
    "register_checker",
]
