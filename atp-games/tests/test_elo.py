"""Tests for Elo rating system."""

import pytest

from atp_games.rating.elo import EloCalculator, EloRating


class TestEloCalculator:
    def test_initial_rating(self) -> None:
        calc = EloCalculator()
        rating = calc.create_rating("agent_a")
        assert rating.rating == 1500.0
        assert rating.games_played == 0

    def test_winner_gains_rating(self) -> None:
        calc = EloCalculator()
        ra = calc.create_rating("a")
        rb = calc.create_rating("b")
        calc.update(ra, rb, winner="a")
        assert ra.rating > 1500.0
        assert rb.rating < 1500.0

    def test_ratings_sum_preserved(self) -> None:
        calc = EloCalculator()
        ra = calc.create_rating("a")
        rb = calc.create_rating("b")
        total_before = ra.rating + rb.rating
        calc.update(ra, rb, winner="a")
        assert ra.rating + rb.rating == pytest.approx(total_before)

    def test_draw_moves_toward_equal(self) -> None:
        calc = EloCalculator()
        ra = EloRating(agent="a", rating=1600.0)
        rb = EloRating(agent="b", rating=1400.0)
        calc.update(ra, rb, winner=None)
        assert ra.rating < 1600.0
        assert rb.rating > 1400.0

    def test_upset_gives_more_points(self) -> None:
        calc = EloCalculator(k_factor=32)
        ra = EloRating(agent="a", rating=1200.0)
        rb = EloRating(agent="b", rating=1800.0)
        calc.update(ra, rb, winner="a")
        gain_a = ra.rating - 1200.0

        rc = EloRating(agent="c", rating=1800.0)
        rd = EloRating(agent="d", rating=1200.0)
        calc.update(rc, rd, winner="c")
        gain_c = rc.rating - 1800.0
        assert gain_a > gain_c

    def test_games_played_increments(self) -> None:
        calc = EloCalculator()
        ra = calc.create_rating("a")
        rb = calc.create_rating("b")
        calc.update(ra, rb, winner="a")
        assert ra.games_played == 1
        assert rb.games_played == 1

    def test_custom_k_factor(self) -> None:
        calc_low = EloCalculator(k_factor=16)
        calc_high = EloCalculator(k_factor=64)
        ra1 = calc_low.create_rating("a")
        rb1 = calc_low.create_rating("b")
        ra2 = calc_high.create_rating("a")
        rb2 = calc_high.create_rating("b")
        calc_low.update(ra1, rb1, winner="a")
        calc_high.update(ra2, rb2, winner="a")
        assert abs(ra2.rating - 1500) > abs(ra1.rating - 1500)

    def test_invalid_winner_raises(self) -> None:
        calc = EloCalculator()
        ra = calc.create_rating("a")
        rb = calc.create_rating("b")
        with pytest.raises(ValueError, match="neither"):
            calc.update(ra, rb, winner="nonexistent")

    def test_win_loss_draw_tracking(self) -> None:
        calc = EloCalculator()
        ra = calc.create_rating("a")
        rb = calc.create_rating("b")
        calc.update(ra, rb, winner="a")
        assert ra.wins == 1
        assert rb.losses == 1
        calc.update(ra, rb, winner=None)
        assert ra.draws == 1
        assert rb.draws == 1
