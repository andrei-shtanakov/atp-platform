"""Tests for the observer pattern module."""

from atp.core.observer import (
    CompositeObserver,
    ErrorCollector,
    LoggingObserver,
    get_observer,
    set_observer,
)


class TestLoggingObserver:
    def test_record_error_no_crash(self) -> None:
        obs = LoggingObserver()
        obs.record_error(ValueError("test"), context="unit_test")

    def test_record_event_no_crash(self) -> None:
        obs = LoggingObserver()
        obs.record_event("test_event", {"key": "value"})


class TestErrorCollector:
    def test_collects_errors(self) -> None:
        collector = ErrorCollector()
        collector.record_error(ValueError("a"), "ctx1")
        collector.record_error(TypeError("b"), "ctx2")
        assert collector.error_count == 2
        assert collector.errors[0].context == "ctx1"
        assert collector.errors[1].context == "ctx2"

    def test_collects_events(self) -> None:
        collector = ErrorCollector()
        collector.record_event("start", {"step": 1})
        collector.record_event("end")
        assert len(collector.events) == 2
        assert collector.events[0].name == "start"
        assert collector.events[0].data == {"step": 1}

    def test_clear(self) -> None:
        collector = ErrorCollector()
        collector.record_error(ValueError("x"))
        collector.record_event("y")
        collector.clear()
        assert collector.error_count == 0
        assert len(collector.events) == 0

    def test_summary(self) -> None:
        collector = ErrorCollector()
        collector.record_error(ValueError("a"))
        collector.record_error(ValueError("b"))
        collector.record_error(TypeError("c"))
        s = collector.summary()
        assert s["total_errors"] == 3
        assert s["errors_by_type"]["ValueError"] == 2
        assert s["errors_by_type"]["TypeError"] == 1


class TestCompositeObserver:
    def test_delegates_to_all(self) -> None:
        c1 = ErrorCollector()
        c2 = ErrorCollector()
        composite = CompositeObserver(c1, c2)
        composite.record_error(ValueError("x"), "ctx")
        composite.record_event("ev")
        assert c1.error_count == 1
        assert c2.error_count == 1
        assert len(c1.events) == 1
        assert len(c2.events) == 1


class TestModuleLevelObserver:
    def test_default_is_logging(self) -> None:
        obs = get_observer()
        assert isinstance(obs, LoggingObserver)

    def test_set_and_get(self) -> None:
        collector = ErrorCollector()
        original = get_observer()
        try:
            set_observer(collector)
            assert get_observer() is collector
        finally:
            set_observer(original)
