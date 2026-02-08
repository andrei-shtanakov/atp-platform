"""Game reporter for game-theoretic evaluation results.

Extends the ATP reporter to handle game-specific data including
payoff matrices, strategy profiles, cooperation dynamics,
tournament standings, and cross-play heatmaps.
"""

import csv
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, Field

from atp.reporters.base import Reporter, SuiteReport


class PlayerResult(BaseModel):
    """Per-player result summary."""

    player_id: str
    strategy: str | None = None
    average_payoff: float = 0.0
    total_payoff: float = 0.0
    cooperation_rate: float | None = None
    exploitability: float | None = None


class EpisodeResult(BaseModel):
    """Result of a single game episode."""

    episode: int
    payoffs: dict[str, float]
    rounds: int = 1
    actions_summary: dict[str, dict[str, int]] = Field(default_factory=dict)


class MatchupResult(BaseModel):
    """Result of a specific agent matchup (for tournaments/cross-play)."""

    player_1: str
    player_2: str
    player_1_avg_payoff: float
    player_2_avg_payoff: float
    episodes: int
    winner: str | None = None


class TournamentStanding(BaseModel):
    """Tournament standing for one agent."""

    rank: int
    agent: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_payoff: float = 0.0
    average_payoff: float = 0.0


class GameReport(BaseModel):
    """Report data for a game evaluation."""

    game_name: str
    game_type: str = "normal_form"
    num_players: int = 2
    num_rounds: int = 1
    num_episodes: int = 1
    players: list[PlayerResult] = Field(default_factory=list)
    episodes: list[EpisodeResult] = Field(default_factory=list)
    payoff_matrix: dict[str, dict[str, float]] | None = None
    strategy_timeline: list[dict[str, Any]] | None = None
    cooperation_dynamics: list[dict[str, Any]] | None = None
    matchups: list[MatchupResult] | None = None
    tournament_standings: list[TournamentStanding] | None = None
    cross_play_matrix: dict[str, dict[str, float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameReport":
        """Create a GameReport from a dictionary."""
        return cls(**data)


class GameReporter(Reporter):
    """Reporter for game-theoretic evaluation results.

    Produces JSON output with game-specific fields including
    payoff matrices, strategy profiles, and tournament data.
    """

    FORMAT_VERSION = "1.0"

    def __init__(
        self,
        output_file: Path | str | None = None,
        output: TextIO | None = None,
        indent: int | None = 2,
        include_episodes: bool = True,
    ) -> None:
        """Initialize the game reporter.

        Args:
            output_file: Path to write JSON file.
            output: Output stream (defaults to sys.stdout).
            indent: JSON indentation level.
            include_episodes: Whether to include per-episode details.
        """
        self._output_file = Path(output_file) if output_file else None
        self._output = output
        self._indent = indent
        self._include_episodes = include_episodes

    @property
    def name(self) -> str:
        """Return the reporter name."""
        return "game"

    def report(self, report: SuiteReport) -> None:
        """Generate standard suite report output.

        For standard ATP suite reports, delegates to JSON format.

        Args:
            report: Suite report data to output.
        """
        json_data = {
            "version": self.FORMAT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "type": "suite_report",
            "summary": {
                "suite_name": report.suite_name,
                "agent_name": report.agent_name,
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "success_rate": round(report.success_rate, 4),
            },
        }
        self._write_json(json_data)

    def report_game(self, game_report: GameReport) -> None:
        """Generate game-specific report output.

        Args:
            game_report: Game report data to output.
        """
        json_data = self._build_game_json(game_report)
        self._write_json(json_data)

    def _build_game_json(self, game_report: GameReport) -> dict[str, Any]:
        """Build JSON structure from game report.

        Args:
            game_report: Game report data.

        Returns:
            Dictionary representing the JSON structure.
        """
        result: dict[str, Any] = {
            "version": self.FORMAT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "type": "game_report",
            "game": {
                "name": game_report.game_name,
                "type": game_report.game_type,
                "num_players": game_report.num_players,
                "num_rounds": game_report.num_rounds,
                "num_episodes": game_report.num_episodes,
            },
            "players": [self._build_player(p) for p in game_report.players],
        }

        if game_report.payoff_matrix is not None:
            result["payoff_matrix"] = game_report.payoff_matrix

        if game_report.strategy_timeline is not None:
            result["strategy_timeline"] = game_report.strategy_timeline

        if game_report.cooperation_dynamics is not None:
            result["cooperation_dynamics"] = game_report.cooperation_dynamics

        if self._include_episodes and game_report.episodes:
            result["episodes"] = [self._build_episode(e) for e in game_report.episodes]

        if game_report.matchups is not None:
            result["matchups"] = [m.model_dump() for m in game_report.matchups]

        if game_report.tournament_standings is not None:
            result["tournament_standings"] = [
                s.model_dump() for s in game_report.tournament_standings
            ]

        if game_report.cross_play_matrix is not None:
            result["cross_play_matrix"] = game_report.cross_play_matrix

        if game_report.metadata:
            result["metadata"] = game_report.metadata

        return result

    def _build_player(self, player: PlayerResult) -> dict[str, Any]:
        """Build player result dictionary."""
        result: dict[str, Any] = {
            "player_id": player.player_id,
            "average_payoff": round(player.average_payoff, 4),
            "total_payoff": round(player.total_payoff, 4),
        }
        if player.strategy is not None:
            result["strategy"] = player.strategy
        if player.cooperation_rate is not None:
            result["cooperation_rate"] = round(player.cooperation_rate, 4)
        if player.exploitability is not None:
            result["exploitability"] = round(player.exploitability, 4)
        return result

    def _build_episode(self, episode: EpisodeResult) -> dict[str, Any]:
        """Build episode result dictionary."""
        result: dict[str, Any] = {
            "episode": episode.episode,
            "payoffs": {k: round(v, 4) for k, v in episode.payoffs.items()},
            "rounds": episode.rounds,
        }
        if episode.actions_summary:
            result["actions_summary"] = episode.actions_summary
        return result

    def _write_json(self, data: dict[str, Any]) -> None:
        """Write JSON data to output."""
        json_str = json.dumps(data, indent=self._indent, default=str)
        if self._output_file:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            self._output_file.write_text(json_str + "\n")
        else:
            output = self._output or sys.stdout
            output.write(json_str + "\n")


class GameHTMLReporter(Reporter):
    """Reporter that produces HTML reports for game results.

    Generates a single-file HTML report with embedded CSS and
    Chart.js visualizations for game-theoretic evaluation results.
    """

    def __init__(
        self,
        output_file: Path | str | None = None,
        title: str = "Game Evaluation Results",
    ) -> None:
        """Initialize the HTML game reporter.

        Args:
            output_file: Path to write HTML file.
            title: Report title.
        """
        self._output_file = Path(output_file) if output_file else None
        self._title = title

    @property
    def name(self) -> str:
        """Return the reporter name."""
        return "game_html"

    def report(self, report: SuiteReport) -> None:
        """Generate standard suite report (minimal)."""
        pass

    def report_game(self, game_report: GameReport) -> str:
        """Generate HTML report for game results.

        Args:
            game_report: Game report data.

        Returns:
            HTML string.
        """
        html = self._render_html(game_report)
        if self._output_file:
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            self._output_file.write_text(html)
        return html

    def _render_html(self, report: GameReport) -> str:
        """Render the HTML report."""
        payoff_section = self._render_payoff_matrix(report)
        players_section = self._render_players_table(report)
        timeline_section = self._render_strategy_timeline(report)
        cooperation_section = self._render_cooperation_chart(report)
        tournament_section = self._render_tournament(report)
        crossplay_section = self._render_crossplay_heatmap(report)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self._title)}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
    <style>
        :root {{
            --color-success: #22c55e;
            --color-error: #ef4444;
            --color-warning: #f59e0b;
            --color-info: #3b82f6;
            --color-bg: #f9fafb;
            --color-card: #ffffff;
            --color-border: #e5e7eb;
            --color-text: #111827;
            --color-text-secondary: #6b7280;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, sans-serif;
            background: var(--color-bg);
            color: var(--color-text);
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 1.5rem; margin-bottom: 1rem; }}
        h2 {{ font-size: 1.2rem; margin: 1.5rem 0 0.75rem; }}
        .card {{
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .metric {{
            text-align: center;
            padding: 1rem;
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
        }}
        .metric .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--color-info);
        }}
        .metric .label {{
            font-size: 0.85rem;
            color: var(--color-text-secondary);
            margin-top: 0.25rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 0.5rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--color-border);
        }}
        th {{ font-weight: 600; background: #f3f4f6; }}
        .heatmap-cell {{
            text-align: center;
            font-weight: 500;
            padding: 0.5rem;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 1rem 0;
        }}
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-info {{ background: #dbeafe; color: #1e40af; }}
    </style>
</head>
<body>
<div class="container">
    <h1>{html.escape(self._title)}</h1>

    <div class="summary">
        <div class="metric">
            <div class="value">{html.escape(report.game_name)}</div>
            <div class="label">Game</div>
        </div>
        <div class="metric">
            <div class="value">{report.num_players}</div>
            <div class="label">Players</div>
        </div>
        <div class="metric">
            <div class="value">{report.num_rounds}</div>
            <div class="label">Rounds</div>
        </div>
        <div class="metric">
            <div class="value">{report.num_episodes}</div>
            <div class="label">Episodes</div>
        </div>
    </div>

    {players_section}
    {payoff_section}
    {timeline_section}
    {cooperation_section}
    {tournament_section}
    {crossplay_section}
</div>
</body>
</html>"""

    def _render_players_table(self, report: GameReport) -> str:
        """Render players summary table."""
        if not report.players:
            return ""
        rows = ""
        for p in report.players:
            coop = (
                f"{p.cooperation_rate:.2%}" if p.cooperation_rate is not None else "N/A"
            )
            expl = f"{p.exploitability:.4f}" if p.exploitability is not None else "N/A"
            strategy = html.escape(p.strategy or "N/A")
            rows += f"""<tr>
                <td>{html.escape(p.player_id)}</td>
                <td>{strategy}</td>
                <td>{p.average_payoff:.2f}</td>
                <td>{p.total_payoff:.2f}</td>
                <td>{coop}</td>
                <td>{expl}</td>
            </tr>"""
        return f"""
    <h2>Player Results</h2>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Player</th><th>Strategy</th>
                    <th>Avg Payoff</th><th>Total Payoff</th>
                    <th>Cooperation</th><th>Exploitability</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""

    def _render_payoff_matrix(self, report: GameReport) -> str:
        """Render payoff matrix as color-coded table."""
        if not report.payoff_matrix:
            return ""
        players = list(report.payoff_matrix.keys())
        header = "".join(f"<th>{html.escape(p)}</th>" for p in players)
        rows = ""
        all_values = [v for row in report.payoff_matrix.values() for v in row.values()]
        min_val = min(all_values) if all_values else 0
        max_val = max(all_values) if all_values else 1
        val_range = max_val - min_val if max_val != min_val else 1

        for row_player in players:
            cells = ""
            for col_player in players:
                val = report.payoff_matrix[row_player].get(col_player, 0)
                intensity = (val - min_val) / val_range
                r = int(239 - intensity * 100)
                g = int(68 + intensity * 140)
                b = int(68 + intensity * 50)
                cells += (
                    f'<td class="heatmap-cell" '
                    f'style="background:rgba({r},{g},{b},0.3)">'
                    f"{val:.2f}</td>"
                )
            rows += f"<tr><th>{html.escape(row_player)}</th>{cells}</tr>"

        return f"""
    <h2>Payoff Matrix</h2>
    <div class="card">
        <table>
            <thead><tr><th></th>{header}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""

    def _render_strategy_timeline(self, report: GameReport) -> str:
        """Render strategy distribution timeline chart."""
        if not report.strategy_timeline:
            return ""
        labels = json.dumps(
            [d.get("round", i) for i, d in enumerate(report.strategy_timeline)]
        )
        strategies: set[str] = set()
        for d in report.strategy_timeline:
            strategies.update(k for k in d if k != "round")
        colors = [
            "#3b82f6",
            "#ef4444",
            "#22c55e",
            "#f59e0b",
            "#8b5cf6",
            "#ec4899",
            "#14b8a6",
            "#f97316",
        ]
        datasets = []
        for i, s in enumerate(sorted(strategies)):
            data = json.dumps([d.get(s, 0) for d in report.strategy_timeline])
            color = colors[i % len(colors)]
            label = json.dumps(s)
            datasets.append(
                f'{{label:{label},data:{data},borderColor:"{color}",fill:false}}'
            )
        datasets_str = ",".join(datasets)
        return f"""
    <h2>Strategy Timeline</h2>
    <div class="card">
        <div class="chart-container">
            <canvas id="strategyChart"></canvas>
        </div>
        <script>
            new Chart(document.getElementById('strategyChart'), {{
                type: 'line',
                data: {{
                    labels: {labels},
                    datasets: [{datasets_str}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: true,
                            title: {{ display: true, text: 'Frequency' }}
                        }},
                        x: {{ title: {{ display: true, text: 'Round' }} }}
                    }}
                }}
            }});
        </script>
    </div>"""

    def _render_cooperation_chart(self, report: GameReport) -> str:
        """Render cooperation dynamics chart."""
        if not report.cooperation_dynamics:
            return ""
        labels = json.dumps(
            [d.get("round", i) for i, d in enumerate(report.cooperation_dynamics)]
        )
        rates = json.dumps(
            [d.get("cooperation_rate", 0) for d in report.cooperation_dynamics]
        )
        return f"""
    <h2>Cooperation Dynamics</h2>
    <div class="card">
        <div class="chart-container">
            <canvas id="cooperationChart"></canvas>
        </div>
        <script>
            new Chart(document.getElementById('cooperationChart'), {{
                type: 'line',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        label: 'Cooperation Rate',
                        data: {rates},
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34,197,94,0.1)',
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ min: 0, max: 1,
                            title: {{ display: true, text: 'Rate' }}
                        }},
                        x: {{ title: {{ display: true, text: 'Round' }} }}
                    }}
                }}
            }});
        </script>
    </div>"""

    def _render_tournament(self, report: GameReport) -> str:
        """Render tournament standings table."""
        if not report.tournament_standings:
            return ""
        rows = ""
        for s in report.tournament_standings:
            rows += f"""<tr>
                <td>{s.rank}</td>
                <td><strong>{html.escape(s.agent)}</strong></td>
                <td>{s.wins}</td>
                <td>{s.losses}</td>
                <td>{s.draws}</td>
                <td>{s.average_payoff:.2f}</td>
            </tr>"""
        return f"""
    <h2>Tournament Standings</h2>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Rank</th><th>Agent</th>
                    <th>Wins</th><th>Losses</th><th>Draws</th>
                    <th>Avg Payoff</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""

    def _render_crossplay_heatmap(self, report: GameReport) -> str:
        """Render cross-play matrix as heatmap table."""
        if not report.cross_play_matrix:
            return ""
        agents = list(report.cross_play_matrix.keys())
        header = "".join(f"<th>{html.escape(a)}</th>" for a in agents)

        all_values = [
            v for row in report.cross_play_matrix.values() for v in row.values()
        ]
        min_val = min(all_values) if all_values else 0
        max_val = max(all_values) if all_values else 1
        val_range = max_val - min_val if max_val != min_val else 1

        rows = ""
        for row_agent in agents:
            cells = ""
            for col_agent in agents:
                val = report.cross_play_matrix[row_agent].get(col_agent, 0)
                intensity = (val - min_val) / val_range
                r = int(239 - intensity * 100)
                g = int(68 + intensity * 140)
                b = int(68 + intensity * 50)
                cells += (
                    f'<td class="heatmap-cell" '
                    f'style="background:rgba({r},{g},{b},0.3)">'
                    f"{val:.2f}</td>"
                )
            rows += f"<tr><th>{html.escape(row_agent)}</th>{cells}</tr>"

        return f"""
    <h2>Cross-Play Matrix</h2>
    <div class="card">
        <table>
            <thead><tr><th>vs</th>{header}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""


def export_game_csv(game_report: GameReport, output: TextIO | None = None) -> str:
    """Export game results to CSV format.

    Args:
        game_report: Game report data.
        output: Optional output stream.

    Returns:
        CSV string.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header
    writer.writerow(
        [
            "episode",
            "player_id",
            "payoff",
            "strategy",
            "cooperation_rate",
        ]
    )

    # Build player lookup
    player_info = {p.player_id: p for p in game_report.players}

    for ep in game_report.episodes:
        for player_id, payoff in ep.payoffs.items():
            info = player_info.get(player_id)
            writer.writerow(
                [
                    ep.episode,
                    player_id,
                    round(payoff, 4),
                    info.strategy if info else "",
                    (
                        round(info.cooperation_rate, 4)
                        if info and info.cooperation_rate is not None
                        else ""
                    ),
                ]
            )

    csv_str = buf.getvalue()
    if output:
        output.write(csv_str)
    return csv_str


def export_game_json(
    game_report: GameReport,
    output: TextIO | None = None,
    indent: int = 2,
) -> str:
    """Export game results to JSON format for Jupyter analysis.

    Args:
        game_report: Game report data.
        output: Optional output stream.
        indent: JSON indentation.

    Returns:
        JSON string.
    """
    data = game_report.model_dump()
    json_str = json.dumps(data, indent=indent, default=str)
    if output:
        output.write(json_str)
    return json_str
