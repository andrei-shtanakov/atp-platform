"""Tests for WEB_CONCURRENCY startup assertion."""

import pytest

from atp.dashboard.v2.factory import assert_single_worker


def test_single_worker_ok(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    assert_single_worker()  # must not raise


def test_default_is_single_worker(monkeypatch):
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    assert_single_worker()  # default = 1


def test_multi_worker_raises(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with pytest.raises(RuntimeError, match="WEB_CONCURRENCY=1"):
        assert_single_worker()
