"""codex_cli usage must be normalized to cloud_pricing_usage_v1 at the shim edge:
cached_input is a SUBSET of input; leaving it inside input over-counts billable
input, and adding it to cache_read as well would double-count."""

from __future__ import annotations

from method.spawners.codex_cli_shim import normalize_usage


def test_cached_split_out_of_input_no_double_count() -> None:
    # Codex raw: input=1000 (incl. 800 cached), output=200, cached=800.
    norm = normalize_usage(input_tokens=1000, output_tokens=200, cached_input=800)
    assert norm["input_tokens"] == 200  # uncached billable
    assert norm["cache_read_tokens"] == 800
    assert norm["cache_creation_tokens"] == 0
    # invariant: input + cache_read + cache_creation == full prompt input
    assert (
        norm["input_tokens"] + norm["cache_read_tokens"] + norm["cache_creation_tokens"]
        == 1000
    )


def test_no_cached_is_identity() -> None:
    norm = normalize_usage(input_tokens=500, output_tokens=50, cached_input=None)
    assert norm["input_tokens"] == 500
    assert norm["cache_read_tokens"] == 0
