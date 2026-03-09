"""Tests for Result type (Success/Failure)."""

from atp.core.result import Failure, Success


class TestSuccess:
    def test_value(self) -> None:
        r = Success(42)
        assert r.value == 42
        assert r.is_success is True
        assert r.is_failure is False

    def test_generic_type(self) -> None:
        r: Success[str] = Success("hello")
        assert r.value == "hello"


class TestFailure:
    def test_error_message(self) -> None:
        r = Failure("something went wrong")
        assert r.error == "something went wrong"
        assert r.is_success is False
        assert r.is_failure is True

    def test_error_with_code(self) -> None:
        r = Failure("timeout", code="TIMEOUT")
        assert r.code == "TIMEOUT"

    def test_error_with_cause(self) -> None:
        exc = ValueError("bad input")
        r = Failure("validation failed", cause=exc)
        assert r.cause is exc

    def test_defaults(self) -> None:
        r = Failure("oops")
        assert r.code is None
        assert r.cause is None


class TestResultUnion:
    def test_pattern_matching(self) -> None:
        results = [Success(10), Failure("err")]
        for r in results:
            if isinstance(r, Success):
                assert r.value == 10
            elif isinstance(r, Failure):
                assert r.error == "err"
