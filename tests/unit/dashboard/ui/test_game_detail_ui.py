"""Tests for the public per-game detail page (/ui/games/{name})."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.v2.factory import create_test_app
from atp.dashboard.v2.game_copy import GAME_COPY


@pytest.fixture
async def client():
    app = create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_game_detail_404_for_unknown_game(client: AsyncClient):
    resp = await client.get("/ui/games/not_a_real_game")
    assert resp.status_code == 404


@pytest.mark.anyio
@pytest.mark.parametrize("game_name", sorted(GAME_COPY.keys()))
async def test_game_detail_renders_for_all_registered_games(
    client: AsyncClient, game_name: str
):
    """Every game with copy renders a 200 with its title and sections."""
    resp = await client.get(f"/ui/games/{game_name}")
    assert resp.status_code == 200, resp.text[:500]

    copy = GAME_COPY[game_name]
    # Core content must be present. HTML autoescaping converts
    # apostrophes to &#39; so compare the escaped form.
    escaped_title = copy.title.replace("'", "&#39;")
    assert escaped_title in resp.text or copy.title in resp.text
    assert "What is this game?" in resp.text
    assert "What&#39;s the point?" in resp.text or "What's the point?" in resp.text
    assert "Rules" in resp.text
    assert "How to participate" in resp.text


@pytest.mark.anyio
async def test_game_detail_public_no_auth_required(client: AsyncClient):
    """Anonymous callers can read a game detail page."""
    resp = await client.get("/ui/games/prisoners_dilemma")
    assert resp.status_code == 200
    # Must not redirect to login
    assert "Login" not in resp.text[:500]


@pytest.mark.anyio
async def test_pd_detail_shows_payoff_matrix(client: AsyncClient):
    resp = await client.get("/ui/games/prisoners_dilemma")
    assert resp.status_code == 200
    # 2x2 payoff cells for PD: (3,3), (0,5), (5,0), (1,1)
    text = resp.text
    assert "3 , 3" in text
    assert "0 , 5" in text
    assert "5 , 0" in text
    assert "1 , 1" in text


@pytest.mark.anyio
async def test_bos_detail_mentions_asymmetry(client: AsyncClient):
    """BoS page must surface the asymmetric-preferences nuance."""
    resp = await client.get("/ui/games/battle_of_sexes")
    assert resp.status_code == 200
    assert "your_preferred" in resp.text
    assert "A-preferring" in resp.text or "prefer" in resp.text.lower()


@pytest.mark.anyio
async def test_coming_soon_game_shows_disclaimer(client: AsyncClient):
    """Games with available=False render the 'tournaments coming soon' CTA."""
    # auction is one of the still-coming-soon games (along with
    # colonel_blotto and congestion). public_goods used to live here
    # but shipped as a live game in the public-goods tournament rollout.
    resp = await client.get("/ui/games/auction")
    assert resp.status_code == 200
    assert "coming soon" in resp.text.lower()
    # Full rules still shown
    assert "Rules" in resp.text


@pytest.mark.anyio
async def test_available_game_shows_participate_steps(client: AsyncClient):
    """Available games render the 4-step onboarding callout."""
    resp = await client.get("/ui/games/prisoners_dilemma")
    assert resp.status_code == 200
    # The 4 numbered steps
    assert "Register" in resp.text
    assert "Create an agent" in resp.text
    assert "Connect via MCP" in resp.text
    assert "Join a tournament" in resp.text


@pytest.mark.anyio
async def test_game_detail_action_example_rendered(client: AsyncClient):
    resp = await client.get("/ui/games/prisoners_dilemma")
    assert resp.status_code == 200
    assert "choice" in resp.text
    assert "cooperate" in resp.text


@pytest.mark.anyio
async def test_games_list_links_to_detail(client: AsyncClient):
    """/ui/games must expose clickable detail-page links."""
    resp = await client.get("/ui/games")
    assert resp.status_code == 200
    # Check that at least one known game links to /ui/games/<name>
    assert "/ui/games/prisoners_dilemma" in resp.text


@pytest.mark.parametrize("game_name", sorted(GAME_COPY.keys()))
def test_action_example_is_valid_json(game_name: str):
    """Every action_example must parse with json.loads.

    Narrative comments ("// or X") belong in action_notes, not inside
    the code block labeled as JSON. Guards against drift back to
    JSON-with-comments.
    """
    copy = GAME_COPY[game_name]
    parsed = json.loads(copy.action_example)
    assert isinstance(parsed, dict)


def test_el_farol_boundary_matches_game_implementation():
    """El Farol public copy must match the code's strict-inequality rule.

    ``el_farol.py`` treats a slot as ``crowded`` when
    ``occupancy >= capacity_threshold`` and ``happy`` only when
    ``occupancy < capacity_threshold`` (strict). Earlier copy used
    "at or below the capacity threshold" for the +1 case, which flips
    the boundary and tells users the wrong rule at occupancy ==
    threshold. This guards against that specific regression.
    """
    copy = GAME_COPY["el_farol"]
    setup_rules_payoff = "\n".join([copy.setup, copy.payoff_formula, *copy.rules])

    # Must describe +1 condition as strictly below, not "at or below".
    assert "at or below" not in setup_rules_payoff, (
        "El Farol copy reintroduces the off-by-one: +1 happens only "
        "below the threshold, not at it."
    )
    assert "≤ capacity" not in setup_rules_payoff, (
        "El Farol copy reintroduces the off-by-one: payoff should be "
        "+1 for attendance strictly less than capacity_threshold."
    )

    # Must positively describe the correct boundary.
    assert "strictly below" in setup_rules_payoff
    assert (
        "reaches or exceeds" in setup_rules_payoff
        or "≥ capacity_threshold" in setup_rules_payoff
    )


@pytest.mark.parametrize("game_name", sorted(GAME_COPY.keys()))
def test_copy_fields_are_plain_text(game_name: str):
    """Copy text must not contain raw HTML — rendered without |safe.

    If copy ever needs emphasis, express it structurally in the template
    (separate fields, tables) rather than via inline HTML in a string —
    raw HTML in copy becomes an XSS footgun if the source ever opens
    up to non-code-owned edits.
    """
    copy = GAME_COPY[game_name]
    forbidden = ("<strong>", "<em>", "<br>", "<script>", "<code>")
    fields = [copy.setup, copy.point, copy.action_notes, copy.payoff_formula]
    fields.extend(copy.rules)
    for field_value in fields:
        for tag in forbidden:
            assert tag not in field_value, (
                f"{game_name}: raw HTML tag {tag!r} found in copy: {field_value[:80]!r}"
            )
