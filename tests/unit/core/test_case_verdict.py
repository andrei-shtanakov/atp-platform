"""Tests for the shared CaseVerdict model (Phase A-1)."""

from atp.core.results import CaseVerdict


def test_caseverdict_defaults_minimal() -> None:
    v = CaseVerdict(critical_pass=True)
    assert v.critical_pass is True
    assert v.malformed is False
    assert v.recall == 0.0
    assert v.fp_count == 0
    assert v.rubric_score == 0.0
    assert v.details == {}
    assert v.grader_version == ""


def test_caseverdict_roundtrips_dump() -> None:
    v = CaseVerdict(
        critical_pass=False,
        malformed=True,
        recall=0.5,
        precision=0.5,
        fp_count=2,
        details={"missed": ["x"]},
        grader_version="findings_match@1",
    )
    d = v.model_dump()
    assert d["malformed"] is True and d["fp_count"] == 2
    assert CaseVerdict(**d) == v
