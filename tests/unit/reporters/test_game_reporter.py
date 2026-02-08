"""Tests for the game reporter."""

import json
from io import StringIO

import pytest

from atp.reporters.base import SuiteReport
from atp.reporters.game_reporter import (
    EpisodeResult,
    GameHTMLReporter,
    GameReport,
    GameReporter,
    MatchupResult,
    PlayerResult,
    TournamentStanding,
    export_game_csv,
    export_game_json,
)


@pytest.fixture
def basic_game_report() -> GameReport:
    """Create a basic game report for testing."""
    return GameReport(
        game_name="Prisoner's Dilemma",
        game_type="normal_form",
        num_players=2,
        num_rounds=10,
        num_episodes=5,
        players=[
            PlayerResult(
                player_id="player_1",
                strategy="tit_for_tat",
                average_payoff=3.0,
                total_payoff=15.0,
                cooperation_rate=0.8,
                exploitability=0.05,
            ),
            PlayerResult(
                player_id="player_2",
                strategy="always_cooperate",
                average_payoff=2.5,
                total_payoff=12.5,
                cooperation_rate=1.0,
                exploitability=0.25,
            ),
        ],
        episodes=[
            EpisodeResult(
                episode=0,
                payoffs={"player_1": 3.0, "player_2": 3.0},
                rounds=10,
                actions_summary={
                    "player_1": {"cooperate": 8, "defect": 2},
                    "player_2": {"cooperate": 10},
                },
            ),
            EpisodeResult(
                episode=1,
                payoffs={"player_1": 3.5, "player_2": 2.0},
                rounds=10,
            ),
        ],
    )


@pytest.fixture
def full_game_report(basic_game_report: GameReport) -> GameReport:
    """Create a game report with all optional fields populated."""
    basic_game_report.payoff_matrix = {
        "player_1": {"player_1": 0.0, "player_2": 3.0},
        "player_2": {"player_1": 2.5, "player_2": 0.0},
    }
    basic_game_report.strategy_timeline = [
        {"round": 1, "cooperate": 0.8, "defect": 0.2},
        {"round": 2, "cooperate": 0.9, "defect": 0.1},
        {"round": 3, "cooperate": 0.7, "defect": 0.3},
    ]
    basic_game_report.cooperation_dynamics = [
        {"round": 1, "cooperation_rate": 0.8},
        {"round": 2, "cooperation_rate": 0.9},
        {"round": 3, "cooperation_rate": 0.7},
    ]
    basic_game_report.matchups = [
        MatchupResult(
            player_1="tit_for_tat",
            player_2="always_cooperate",
            player_1_avg_payoff=3.0,
            player_2_avg_payoff=2.5,
            episodes=5,
            winner="tit_for_tat",
        ),
    ]
    basic_game_report.tournament_standings = [
        TournamentStanding(
            rank=1,
            agent="tit_for_tat",
            wins=3,
            losses=0,
            draws=1,
            total_payoff=15.0,
            average_payoff=3.75,
        ),
        TournamentStanding(
            rank=2,
            agent="always_defect",
            wins=2,
            losses=1,
            draws=1,
            total_payoff=12.0,
            average_payoff=3.0,
        ),
    ]
    basic_game_report.cross_play_matrix = {
        "tit_for_tat": {
            "tit_for_tat": 3.0,
            "always_defect": 1.5,
        },
        "always_defect": {
            "tit_for_tat": 4.0,
            "always_defect": 1.0,
        },
    }
    basic_game_report.metadata = {"seed": 42, "noise": 0.0}
    return basic_game_report


class TestGameReporter:
    """Tests for GameReporter."""

    @pytest.fixture
    def output(self) -> StringIO:
        return StringIO()

    @pytest.fixture
    def reporter(self, output: StringIO) -> GameReporter:
        return GameReporter(output=output)

    def test_name(self, reporter: GameReporter) -> None:
        assert reporter.name == "game"

    def test_report_suite(self, reporter: GameReporter, output: StringIO) -> None:
        """Test standard suite report output."""
        suite_report = SuiteReport(
            suite_name="game-suite",
            agent_name="test-agent",
            total_tests=3,
            passed_tests=2,
            failed_tests=1,
            success_rate=0.6667,
            tests=[],
        )
        reporter.report(suite_report)
        result = json.loads(output.getvalue())

        assert result["type"] == "suite_report"
        assert result["version"] == "1.0"
        assert result["summary"]["suite_name"] == "game-suite"
        assert result["summary"]["total_tests"] == 3
        assert result["summary"]["success_rate"] == 0.6667

    def test_report_game_basic(
        self,
        reporter: GameReporter,
        output: StringIO,
        basic_game_report: GameReport,
    ) -> None:
        """Test basic game report output."""
        reporter.report_game(basic_game_report)
        result = json.loads(output.getvalue())

        assert result["type"] == "game_report"
        assert result["version"] == "1.0"
        assert result["game"]["name"] == "Prisoner's Dilemma"
        assert result["game"]["num_players"] == 2
        assert result["game"]["num_rounds"] == 10
        assert result["game"]["num_episodes"] == 5
        assert len(result["players"]) == 2
        assert result["players"][0]["player_id"] == "player_1"
        assert result["players"][0]["strategy"] == "tit_for_tat"
        assert result["players"][0]["average_payoff"] == 3.0
        assert result["players"][0]["cooperation_rate"] == 0.8
        assert result["players"][0]["exploitability"] == 0.05
        assert len(result["episodes"]) == 2

    def test_report_game_full(
        self,
        reporter: GameReporter,
        output: StringIO,
        full_game_report: GameReport,
    ) -> None:
        """Test full game report with all optional fields."""
        reporter.report_game(full_game_report)
        result = json.loads(output.getvalue())

        assert "payoff_matrix" in result
        assert "strategy_timeline" in result
        assert "cooperation_dynamics" in result
        assert "matchups" in result
        assert "tournament_standings" in result
        assert "cross_play_matrix" in result
        assert result["metadata"]["seed"] == 42

    def test_report_game_excludes_episodes(
        self,
        output: StringIO,
        basic_game_report: GameReport,
    ) -> None:
        """Test that include_episodes=False excludes episodes."""
        reporter = GameReporter(output=output, include_episodes=False)
        reporter.report_game(basic_game_report)
        result = json.loads(output.getvalue())

        assert "episodes" not in result

    def test_report_game_no_optional_fields(
        self,
        reporter: GameReporter,
        output: StringIO,
    ) -> None:
        """Test report with no optional fields omits them."""
        report = GameReport(
            game_name="Simple Game",
            players=[
                PlayerResult(player_id="p1", average_payoff=1.0),
            ],
        )
        reporter.report_game(report)
        result = json.loads(output.getvalue())

        assert "payoff_matrix" not in result
        assert "strategy_timeline" not in result
        assert "cooperation_dynamics" not in result
        assert "matchups" not in result
        assert "tournament_standings" not in result
        assert "cross_play_matrix" not in result

    def test_player_without_optional_fields(
        self,
        reporter: GameReporter,
        output: StringIO,
    ) -> None:
        """Test player result without optional cooperation/exploitability."""
        report = GameReport(
            game_name="Test",
            players=[
                PlayerResult(
                    player_id="p1",
                    average_payoff=2.5,
                    total_payoff=10.0,
                ),
            ],
        )
        reporter.report_game(report)
        result = json.loads(output.getvalue())

        player = result["players"][0]
        assert "strategy" not in player
        assert "cooperation_rate" not in player
        assert "exploitability" not in player

    def test_report_to_file(
        self, tmp_path: object, basic_game_report: GameReport
    ) -> None:
        """Test writing report to file."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        output_file = tmp_path / "report.json"
        reporter = GameReporter(output_file=output_file)
        reporter.report_game(basic_game_report)

        assert output_file.exists()
        result = json.loads(output_file.read_text())
        assert result["type"] == "game_report"

    def test_episode_actions_summary(
        self,
        reporter: GameReporter,
        output: StringIO,
        basic_game_report: GameReport,
    ) -> None:
        """Test that actions_summary is included when present."""
        reporter.report_game(basic_game_report)
        result = json.loads(output.getvalue())

        ep0 = result["episodes"][0]
        assert "actions_summary" in ep0
        assert ep0["actions_summary"]["player_1"]["cooperate"] == 8

        ep1 = result["episodes"][1]
        assert "actions_summary" not in ep1


class TestGameHTMLReporter:
    """Tests for GameHTMLReporter."""

    def test_name(self) -> None:
        reporter = GameHTMLReporter()
        assert reporter.name == "game_html"

    def test_report_game_returns_html(self, basic_game_report: GameReport) -> None:
        reporter = GameHTMLReporter(title="Test Results")
        html = reporter.report_game(basic_game_report)

        assert "<!DOCTYPE html>" in html
        assert "Test Results" in html
        assert "Prisoner&#x27;s Dilemma" in html
        assert "2" in html  # num_players

    def test_report_game_with_payoff_matrix(self, full_game_report: GameReport) -> None:
        reporter = GameHTMLReporter()
        html = reporter.report_game(full_game_report)

        assert "Payoff Matrix" in html
        assert "heatmap-cell" in html

    def test_report_game_with_strategy_timeline(
        self, full_game_report: GameReport
    ) -> None:
        reporter = GameHTMLReporter()
        html = reporter.report_game(full_game_report)

        assert "Strategy Timeline" in html
        assert "strategyChart" in html

    def test_report_game_with_cooperation_dynamics(
        self, full_game_report: GameReport
    ) -> None:
        reporter = GameHTMLReporter()
        html = reporter.report_game(full_game_report)

        assert "Cooperation Dynamics" in html
        assert "cooperationChart" in html

    def test_report_game_with_tournament(self, full_game_report: GameReport) -> None:
        reporter = GameHTMLReporter()
        html = reporter.report_game(full_game_report)

        assert "Tournament Standings" in html
        assert "tit_for_tat" in html

    def test_report_game_with_crossplay(self, full_game_report: GameReport) -> None:
        reporter = GameHTMLReporter()
        html = reporter.report_game(full_game_report)

        assert "Cross-Play Matrix" in html

    def test_report_game_to_file(
        self, tmp_path: object, basic_game_report: GameReport
    ) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        output_file = tmp_path / "report.html"
        reporter = GameHTMLReporter(output_file=output_file)
        reporter.report_game(basic_game_report)

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_report_game_empty_sections(self) -> None:
        """Test HTML report with no optional sections."""
        reporter = GameHTMLReporter()
        report = GameReport(game_name="Minimal")
        html = reporter.report_game(report)

        assert "<!DOCTYPE html>" in html
        assert "Payoff Matrix" not in html
        assert "Strategy Timeline" not in html
        assert "Cooperation Dynamics" not in html
        assert "Tournament Standings" not in html
        assert "Cross-Play Matrix" not in html

    def test_suite_report_noop(self) -> None:
        """Test that standard suite report is a no-op."""
        reporter = GameHTMLReporter()
        suite_report = SuiteReport(
            suite_name="s",
            agent_name="a",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            success_rate=0.0,
        )
        reporter.report(suite_report)  # Should not raise


class TestExportGameCSV:
    """Tests for export_game_csv."""

    def test_basic_csv_export(self, basic_game_report: GameReport) -> None:
        csv_str = export_game_csv(basic_game_report)
        lines = csv_str.strip().splitlines()

        assert lines[0].strip() == "episode,player_id,payoff,strategy,cooperation_rate"
        assert len(lines) == 5  # header + 4 data rows (2 episodes x 2 players)

    def test_csv_export_to_stream(self, basic_game_report: GameReport) -> None:
        output = StringIO()
        csv_str = export_game_csv(basic_game_report, output=output)

        assert output.getvalue() == csv_str

    def test_csv_payoff_values(self, basic_game_report: GameReport) -> None:
        csv_str = export_game_csv(basic_game_report)
        lines = csv_str.strip().splitlines()

        # First episode, first player
        fields = lines[1].strip().split(",")
        assert fields[0] == "0"  # episode
        assert fields[1] == "player_1"
        assert fields[2] == "3.0"  # payoff
        assert fields[3] == "tit_for_tat"  # strategy
        assert fields[4] == "0.8"  # cooperation_rate

    def test_csv_empty_report(self) -> None:
        report = GameReport(game_name="Empty")
        csv_str = export_game_csv(report)
        lines = csv_str.strip().splitlines()
        assert len(lines) == 1  # header only


class TestExportGameJSON:
    """Tests for export_game_json."""

    def test_basic_json_export(self, basic_game_report: GameReport) -> None:
        json_str = export_game_json(basic_game_report)
        data = json.loads(json_str)

        assert data["game_name"] == "Prisoner's Dilemma"
        assert data["num_players"] == 2
        assert len(data["players"]) == 2
        assert len(data["episodes"]) == 2

    def test_json_export_to_stream(self, basic_game_report: GameReport) -> None:
        output = StringIO()
        json_str = export_game_json(basic_game_report, output=output)

        assert output.getvalue() == json_str

    def test_json_export_indent(self, basic_game_report: GameReport) -> None:
        json_str = export_game_json(basic_game_report, indent=4)
        # 4-space indent should produce different output than 2-space
        assert "    " in json_str


class TestGameReportModel:
    """Tests for GameReport data model."""

    def test_from_dict(self) -> None:
        data = {
            "game_name": "Test Game",
            "game_type": "repeated",
            "num_players": 3,
            "num_rounds": 20,
            "num_episodes": 10,
        }
        report = GameReport.from_dict(data)
        assert report.game_name == "Test Game"
        assert report.game_type == "repeated"
        assert report.num_players == 3

    def test_defaults(self) -> None:
        report = GameReport(game_name="Minimal")
        assert report.game_type == "normal_form"
        assert report.num_players == 2
        assert report.num_rounds == 1
        assert report.num_episodes == 1
        assert report.players == []
        assert report.episodes == []
        assert report.payoff_matrix is None
        assert report.metadata == {}
