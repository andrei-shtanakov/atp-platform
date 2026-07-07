"""Contract test: pins the real litellm.cost_per_token cache API.

Monkeypatched unit tests guard our code but not our ASSUMPTIONS about litellm.
This test runs against the installed library and fails loudly if the signature,
return shape, or cache semantics drift. Marked slow/opt-in: skipped when the
[pricing] extra is not installed.
"""

from __future__ import annotations

import pytest

litellm = pytest.importorskip("litellm")


# A known Anthropic model with published cache tariffs, currently present in
# litellm's model_cost map. "claude-3-5-sonnet-20240620" (the model originally
# specified for this test) has since been dropped from litellm's registry
# (litellm 1.91.0, checked 2026-07-07: `cost_per_token` raises "This model
# isn't mapped yet" for it) — an unrelated registry-churn issue, not a
# cache-semantics one. Swap this constant if litellm drops this model too.
_PRICED_MODEL = "claude-sonnet-4-5-20250929"


@pytest.mark.slow
def test_cost_per_token_accepts_cache_kwargs_and_returns_pair() -> None:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=_PRICED_MODEL,
        prompt_tokens=1000,
        completion_tokens=500,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=0,
    )
    assert isinstance(prompt_cost, float)
    assert isinstance(completion_cost, float)
    # completion is billed at the full output rate.
    assert completion_cost > 0


@pytest.mark.slow
def test_cache_read_is_cheaper_than_full_input() -> None:
    """Pins the semantic we depend on: cache_read is discounted, and prompt_tokens
    is passed INCLUSIVE of the cache classes (litellm subtracts internally).
    If this fails, Task 3's argument wiring must flip to exclusive."""
    full, _ = litellm.cost_per_token(
        model=_PRICED_MODEL,
        prompt_tokens=1000,
        completion_tokens=0,
        cache_read_input_tokens=0,
    )
    with_cache, _ = litellm.cost_per_token(
        model=_PRICED_MODEL,
        prompt_tokens=1000,  # inclusive: 900 cache_read + 100 uncached
        completion_tokens=0,
        cache_read_input_tokens=900,
    )
    # Same total prompt tokens, but 900 read from cache => strictly cheaper.
    assert with_cache < full
