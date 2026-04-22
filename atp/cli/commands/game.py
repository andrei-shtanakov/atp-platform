"""CLI commands for game-theoretic evaluation suites."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click


@click.group(name="game")
def game_command() -> None:
    """Game-theoretic evaluation commands.

    Run game suites, list available games, and view game info.

    Examples:

      atp game run suites/prisoners_dilemma.yaml
      atp game list
      atp game info prisoners_dilemma
    """


@game_command.command(name="run")
@click.argument(
    "suite_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--episodes",
    type=int,
    default=None,
    help="Override number of episodes (default: from suite)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--output",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path (for json output)",
)
def game_run(
    suite_file: Path,
    episodes: int | None,
    verbose: bool,
    output: str,
    output_file: Path | None,
) -> None:
    """Run a game evaluation suite.

    SUITE_FILE is the path to a game suite YAML file.

    Examples:

      atp game run suites/prisoners_dilemma.yaml
      atp game run suites/auction.yaml --episodes=100
      atp game run suites/pd.yaml --output=json --output-file=results.json
    """
    try:
        result = asyncio.run(
            _run_game_suite(
                suite_file=suite_file,
                episodes=episodes,
                verbose=verbose,
                output_format=output,
                output_file=output_file,
            )
        )
        sys.exit(0 if result else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


async def _run_game_suite(
    suite_file: Path,
    episodes: int | None,
    verbose: bool,
    output_format: str,
    output_file: Path | None,
) -> bool:
    """Run a game suite asynchronously.

    Args:
        suite_file: Path to game suite YAML.
        episodes: Optional episode count override.
        verbose: Verbose output flag.
        output_format: Output format.
        output_file: Optional output file path.

    Returns:
        True if the suite ran successfully.
    """
    import json

    from atp_games.suites.game_suite_loader import (  # pyrefly: ignore[missing-import]
        GameSuiteLoader,
    )

    loader = GameSuiteLoader()
    suite = loader.load_file(suite_file)

    click.echo(f"Game Suite: {suite.name}")
    click.echo(f"Game: {suite.game.type} ({suite.game.variant})")
    click.echo(f"Agents: {len(suite.agents)}")

    # Resolve components
    game = loader.resolve_game(suite)
    agents = loader.resolve_agents(suite)
    run_config = loader.resolve_run_config(suite)

    if episodes is not None:
        from atp_games.models import (  # pyrefly: ignore[missing-import]
            GameRunConfig,
        )

        run_config = GameRunConfig(
            episodes=episodes,
            max_retries=run_config.max_retries,
            move_timeout=run_config.move_timeout,
        )

    click.echo(f"Episodes: {run_config.episodes}")
    click.echo()

    # Run the game
    from atp_games.runner.game_runner import (  # pyrefly: ignore[missing-import]
        GameRunner,
    )

    runner = GameRunner()
    result = await runner.run_game(
        game=game,
        agents=agents,
        config=run_config,
    )

    # Output results
    if output_format == "json":
        result_dict = result.to_dict()
        output_text = json.dumps(result_dict, indent=2)
        if output_file:
            output_file.write_text(output_text)
            click.echo(f"Results written to {output_file}")
        else:
            click.echo(output_text)
    else:
        _print_game_results(result, verbose)

    # Store results in dashboard DB (if available)
    await _store_game_result(result, suite, run_config, game)

    return True


def _print_game_results(result: Any, verbose: bool) -> None:
    """Print game results to console.

    Args:
        result: GameResult object.
        verbose: Whether to show detailed output.
    """
    click.echo(f"Game: {result.game_name}")
    click.echo(f"Episodes: {result.num_episodes}")
    click.echo()

    click.echo("Average Payoffs:")
    for pid, payoff in result.average_payoffs.items():
        agent_name = result.agent_names.get(pid, pid)
        click.echo(f"  {agent_name}: {payoff:.4f}")

    if verbose and result.episodes:
        click.echo()
        click.echo("Per-Episode Payoffs:")
        for ep in result.episodes:
            payoffs_str = ", ".join(f"{pid}={p:.4f}" for pid, p in ep.payoffs.items())
            click.echo(f"  Episode {ep.episode}: {payoffs_str}")


@game_command.command(name="tournament")
@click.argument(
    "suite_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--mode",
    type=click.Choice(["round_robin", "single_elimination", "double_elimination"]),
    default="round_robin",
    help="Tournament mode (default: round_robin)",
)
@click.option(
    "--episodes",
    type=int,
    default=None,
    help="Override episodes per matchup",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--output",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path (for json output)",
)
def game_tournament(
    suite_file: Path,
    mode: str,
    episodes: int | None,
    verbose: bool,
    output: str,
    output_file: Path | None,
) -> None:
    """Run a tournament between agents.

    SUITE_FILE is the path to a game suite YAML file defining
    the game and agents.

    Examples:

      atp game tournament suites/pd.yaml
      atp game tournament suites/pd.yaml --mode=single_elimination
      atp game tournament suites/pd.yaml --episodes=20
    """
    try:
        result = asyncio.run(
            _run_tournament(
                suite_file=suite_file,
                mode=mode,
                episodes=episodes,
                verbose=verbose,
                output_format=output,
                output_file=output_file,
            )
        )
        sys.exit(0 if result else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


@game_command.command(name="crossplay")
@click.argument(
    "suite_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--episodes",
    type=int,
    default=None,
    help="Override episodes per matchup",
)
@click.option(
    "--no-self-play",
    is_flag=True,
    help="Exclude self-play matchups",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--output",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path (for json output)",
)
def game_crossplay(
    suite_file: Path,
    episodes: int | None,
    no_self_play: bool,
    verbose: bool,
    output: str,
    output_file: Path | None,
) -> None:
    """Run cross-play matrix between agents.

    SUITE_FILE is the path to a game suite YAML file.

    Examples:

      atp game crossplay suites/pd.yaml
      atp game crossplay suites/pd.yaml --no-self-play
      atp game crossplay suites/pd.yaml --output=json
    """
    try:
        result = asyncio.run(
            _run_crossplay(
                suite_file=suite_file,
                episodes=episodes,
                include_self_play=not no_self_play,
                verbose=verbose,
                output_format=output,
                output_file=output_file,
            )
        )
        sys.exit(0 if result else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


async def _run_tournament(
    suite_file: Path,
    mode: str,
    episodes: int | None,
    verbose: bool,
    output_format: str,
    output_file: Path | None,
) -> bool:
    """Run a tournament asynchronously.

    Args:
        suite_file: Path to game suite YAML.
        mode: Tournament mode.
        episodes: Optional episode count override.
        verbose: Verbose output flag.
        output_format: Output format.
        output_file: Optional output file path.

    Returns:
        True if tournament ran successfully.
    """
    import json

    from atp_games.models import (  # pyrefly: ignore[missing-import]
        GameRunConfig,
    )
    from atp_games.runner.game_runner import (  # pyrefly: ignore[missing-import]
        GameRunner,
    )
    from atp_games.suites.game_suite_loader import (  # pyrefly: ignore[missing-import]
        GameSuiteLoader,
    )
    from atp_games.suites.tournament import (  # pyrefly: ignore[missing-import]
        run_double_elimination,
        run_round_robin,
        run_single_elimination,
    )

    loader = GameSuiteLoader()
    suite = loader.load_file(suite_file)

    click.echo(f"Tournament: {suite.name}")
    click.echo(f"Mode: {mode}")
    click.echo(f"Game: {suite.game.type} ({suite.game.variant})")
    click.echo(f"Agents: {len(suite.agents)}")

    game = loader.resolve_game(suite)
    agents = loader.resolve_agents(suite)
    run_config = loader.resolve_run_config(suite)

    if episodes is not None:
        run_config = GameRunConfig(
            episodes=episodes,
            max_retries=run_config.max_retries,
            move_timeout=run_config.move_timeout,
        )

    click.echo(f"Episodes per matchup: {run_config.episodes}")
    click.echo()

    runner = GameRunner()

    if mode == "round_robin":
        result = await run_round_robin(
            game=game, agents=agents, config=run_config, runner=runner
        )
    elif mode == "single_elimination":
        result = await run_single_elimination(
            game=game, agents=agents, config=run_config, runner=runner
        )
    elif mode == "double_elimination":
        result = await run_double_elimination(
            game=game, agents=agents, config=run_config, runner=runner
        )
    else:
        click.echo(f"Unknown mode: {mode}", err=True)
        return False

    if output_format == "json":
        output_text = json.dumps(result.to_dict(), indent=2)
        if output_file:
            output_file.write_text(output_text)
            click.echo(f"Results written to {output_file}")
        else:
            click.echo(output_text)
    else:
        _print_tournament_results(result, verbose)

    # Store in dashboard DB
    await _store_tournament_result(result, suite, mode, run_config)

    return True


async def _run_crossplay(
    suite_file: Path,
    episodes: int | None,
    include_self_play: bool,
    verbose: bool,
    output_format: str,
    output_file: Path | None,
) -> bool:
    """Run cross-play matrix asynchronously.

    Args:
        suite_file: Path to game suite YAML.
        episodes: Optional episode count override.
        include_self_play: Whether to include self-play.
        verbose: Verbose output flag.
        output_format: Output format.
        output_file: Optional output file path.

    Returns:
        True if cross-play ran successfully.
    """
    import json

    from atp_games.models import (  # pyrefly: ignore[missing-import]
        GameRunConfig,
    )
    from atp_games.runner.game_runner import (  # pyrefly: ignore[missing-import]
        GameRunner,
    )
    from atp_games.suites.cross_play import (  # pyrefly: ignore[missing-import]
        run_cross_play,
    )
    from atp_games.suites.game_suite_loader import (  # pyrefly: ignore[missing-import]
        GameSuiteLoader,
    )

    loader = GameSuiteLoader()
    suite = loader.load_file(suite_file)

    click.echo(f"Cross-Play Matrix: {suite.name}")
    click.echo(f"Game: {suite.game.type} ({suite.game.variant})")
    click.echo(f"Agents: {len(suite.agents)}")

    game = loader.resolve_game(suite)
    agents = loader.resolve_agents(suite)
    run_config = loader.resolve_run_config(suite)

    if episodes is not None:
        run_config = GameRunConfig(
            episodes=episodes,
            max_retries=run_config.max_retries,
            move_timeout=run_config.move_timeout,
        )

    click.echo(f"Episodes per matchup: {run_config.episodes}")
    click.echo(f"Self-play: {'yes' if include_self_play else 'no'}")
    click.echo()

    runner = GameRunner()
    result = await run_cross_play(
        game=game,
        agents=agents,
        config=run_config,
        runner=runner,
        include_self_play=include_self_play,
    )

    if output_format == "json":
        output_text = json.dumps(result.to_dict(), indent=2)
        if output_file:
            output_file.write_text(output_text)
            click.echo(f"Results written to {output_file}")
        else:
            click.echo(output_text)
    else:
        _print_crossplay_results(result, verbose)

    return True


def _print_tournament_results(result: Any, verbose: bool) -> None:
    """Print tournament results to console.

    Args:
        result: TournamentResult object.
        verbose: Whether to show detailed output.
    """
    click.echo(f"Tournament Mode: {result.mode}")
    click.echo(f"Total Matches: {len(result.matches)}")
    click.echo()

    click.echo("Standings:")
    click.echo(f"  {'Agent':<20} {'W':>4} {'L':>4} {'D':>4} {'Pts':>6} {'Payoff':>10}")
    click.echo("  " + "-" * 52)
    for i, standing in enumerate(result.standings, 1):
        click.echo(
            f"  {standing.agent:<20} "
            f"{standing.wins:>4} "
            f"{standing.losses:>4} "
            f"{standing.draws:>4} "
            f"{standing.points:>6.0f} "
            f"{standing.total_payoff:>10.4f}"
        )

    if verbose:
        click.echo()
        click.echo("Match Details:")
        for match in result.matches:
            winner_str = match.winner or "draw"
            click.echo(
                f"  {match.agent_a} vs {match.agent_b}: "
                f"{match.score_a:.4f} - {match.score_b:.4f} "
                f"(winner: {winner_str})"
            )


def _print_crossplay_results(result: Any, verbose: bool) -> None:
    """Print cross-play results to console.

    Args:
        result: CrossPlayResult object.
        verbose: Whether to show detailed output.
    """
    click.echo("Cross-Play Payoff Matrix:")
    click.echo()

    # Header
    max_name = max(len(a) for a in result.agents) if result.agents else 10
    header = " " * (max_name + 2) + "  ".join(f"{a:>{max_name}}" for a in result.agents)
    click.echo(header)
    click.echo(" " * (max_name + 2) + "-" * (len(result.agents) * (max_name + 2)))

    for row_agent in result.agents:
        values = "  ".join(
            f"{result.matrix[row_agent].get(col, 0.0):>{max_name}.4f}"
            for col in result.agents
        )
        click.echo(f"{row_agent:>{max_name}}  {values}")

    if result.dominance:
        click.echo()
        click.echo("Dominance Relations:")
        for dom in result.dominance:
            kind = "strictly" if dom.strict else "weakly"
            click.echo(f"  {dom.dominator} {kind} dominates {dom.dominated}")

    if result.pareto_frontier:
        click.echo()
        click.echo(f"Pareto Frontier: {', '.join(result.pareto_frontier)}")

    if verbose and result.clusters:
        click.echo()
        click.echo("Clusters:")
        for i, cluster in enumerate(result.clusters, 1):
            click.echo(f"  Cluster {i}: {', '.join(cluster)}")


@game_command.command(name="benchmark")
@click.option(
    "--suite",
    "-s",
    "suite_name",
    type=str,
    default="alympics",
    help="Benchmark suite to run (default: alympics)",
)
@click.option(
    "--episodes",
    type=int,
    default=None,
    help="Override episode count per game",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--output",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Output file path (for json output)",
)
def game_benchmark(
    suite_name: str,
    episodes: int | None,
    verbose: bool,
    output: str,
    output_file: Path | None,
) -> None:
    """Run a game-theoretic benchmark suite.

    Evaluates agents across all 5 canonical games and computes
    composite scores across strategic reasoning, cooperation,
    fairness, and robustness categories.

    Examples:

      atp game benchmark
      atp game benchmark --suite=alympics
      atp game benchmark --episodes=10 --output=json
    """
    try:
        result = asyncio.run(
            _run_benchmark(
                suite_name=suite_name,
                episodes=episodes,
                verbose=verbose,
                output_format=output,
                output_file=output_file,
            )
        )
        sys.exit(0 if result else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


async def _run_benchmark(
    suite_name: str,
    episodes: int | None,
    verbose: bool,
    output_format: str,
    output_file: Path | None,
) -> bool:
    """Run an Alympics benchmark asynchronously.

    Args:
        suite_name: Name of the benchmark suite.
        episodes: Optional episode count override.
        verbose: Verbose output flag.
        output_format: Output format (console or json).
        output_file: Optional output file path.

    Returns:
        True if benchmark completed successfully.
    """
    import json as json_mod

    if suite_name != "alympics":
        click.echo(
            f"Unknown benchmark suite: {suite_name}. Available: alympics",
            err=True,
        )
        return False

    from atp_games.suites.alympics import (  # pyrefly: ignore[missing-import]
        run_alympics,
    )

    click.echo("Alympics Benchmark Suite")
    click.echo("=" * 50)
    click.echo("Running all 5 canonical games...")
    click.echo()

    result = await run_alympics(
        agent_name="builtin",
        episodes_override=episodes,
        verbose=verbose,
    )

    if output_format == "json":
        output_text = json_mod.dumps(result.to_dict(), indent=2)
        if output_file:
            output_file.write_text(output_text)
            click.echo(f"Results written to {output_file}")
        else:
            click.echo(output_text)
    else:
        _print_benchmark_results(result)

    return True


def _print_benchmark_results(result: Any) -> None:
    """Print Alympics benchmark results to console.

    Args:
        result: AlympicsResult object.
    """
    click.echo(result.summary())
    click.echo()

    click.echo("Category Breakdown:")
    click.echo(f"  {'Category':<15} {'Score':>8} {'Weight':>8}")
    click.echo("  " + "-" * 33)
    for cat in result.categories.values():
        click.echo(f"  {cat.name:<15} {cat.score:>7.1f} {cat.weight:>7.0%}")
        for game, score in cat.game_scores.items():
            click.echo(f"    {game:<20} {score:>5.1f}")

    click.echo()
    click.echo("Per-Game Average Payoffs:")
    for game_type, game_result in result.game_results.items():
        click.echo(f"  {game_type}:")
        for pid, payoff in game_result.average_payoffs.items():
            agent_name = game_result.agent_names.get(pid, pid)
            click.echo(f"    {agent_name}: {payoff:.4f}")


def _build_game_result_kwargs(
    result: Any,
    suite: Any,
    game: Any,
) -> dict[str, Any]:
    """Build ORM ``GameResult(...)`` kwargs from a run result + suite + game.

    Populates both the legacy JSON-blob fields (``players_json``,
    ``episodes_json``, ``metadata_json``) and the Phase 7 El Farol
    additive columns (``match_id``, ``game_version``, ``num_days``,
    ``num_slots``, ``max_intervals``, ``max_total_slots``,
    ``capacity_ratio``, ``capacity_threshold``, ``actions_json``,
    ``day_aggregates_json``, ``round_payoffs_json``, ``agents_json``).

    Keys whose source data is absent are omitted entirely so the ORM
    stores ``NULL`` for legacy rows / non-El-Farol games. This is a pure
    function — no DB, no I/O — so it is easy to unit-test.
    """
    from dataclasses import asdict
    from datetime import datetime as _dt

    # Legacy players_json
    players: list[dict[str, Any]] = []
    for pid, payoff in result.average_payoffs.items():
        name = result.agent_names.get(pid, pid)
        players.append(
            {
                "player_id": pid,
                "name": name,
                "strategy": name,
                "average_payoff": payoff,
            }
        )

    episodes_data: list[dict[str, Any]] = []
    for ep in result.episodes:
        episodes_data.append(
            {
                "episode": ep.episode,
                "payoffs": ep.payoffs,
                "seed": ep.seed,
            }
        )

    # ``num_rounds`` / ``num_days`` come from the game's typed config
    # (authoritative) rather than the YAML suite blob, so users who leave
    # suite.game.config empty still get accurate values.
    game_cfg = getattr(game, "config", None)
    num_rounds = getattr(game_cfg, "num_rounds", None)

    kwargs: dict[str, Any] = {
        "game_name": result.game_name,
        "game_type": suite.game.variant or "one_shot",
        "num_players": len(result.agent_names),
        "num_rounds": num_rounds if num_rounds is not None else 1,
        "num_episodes": len(result.episodes),
        "status": "completed",
        "completed_at": _dt.now(),
        "players_json": players,
        "episodes_json": episodes_data,
        "metadata_json": {
            "suite_name": suite.name,
            "config": {
                "episodes": result.config.episodes,
            },
        },
    }

    # ------------------------------------------------------------------
    # Phase 7 additive columns — keys omitted when source data is absent
    # so legacy rows continue to read as NULL.
    # ------------------------------------------------------------------

    # match_id = run_id — distinguishes independent runs of the same config.
    run_id = getattr(result, "run_id", None)
    if run_id:
        kwargs["match_id"] = run_id

    # game_version from the suite envelope (YAML-declared).
    suite_version = getattr(suite, "version", None)
    if suite_version:
        kwargs["game_version"] = suite_version

    # num_days mirrors num_rounds when present on the game config.
    if num_rounds is not None:
        kwargs["num_days"] = num_rounds

    # El Farol-only config columns: populated when the typed game config
    # exposes the attribute. ``hasattr`` gate keeps non-El-Farol games
    # (PD, etc.) from leaking stray ``None`` defaults into the row.
    for attr in (
        "num_slots",
        "max_intervals",
        "max_total_slots",
        "capacity_ratio",
        "capacity_threshold",
    ):
        if game_cfg is not None and hasattr(game_cfg, attr):
            kwargs[attr] = getattr(game_cfg, attr)

    # Typed agent roster — list[AgentRecord] → list[dict].
    agents = list(getattr(result, "agents", None) or [])
    if agents:
        kwargs["agents_json"] = [asdict(a) for a in agents]

    # Flatten typed ActionRecords across all episodes. Gate for
    # day_aggregates: episode 0 having any ActionRecord marks the run as
    # El-Farol-shaped and therefore worth deriving per-day attendance.
    actions_json = _serialize_action_records(result.episodes)
    if actions_json:
        kwargs["actions_json"] = actions_json

    # Per-round payoffs — episode 0 only. Multi-episode runs still emit
    # all action records above; here we persist the canonical series.
    if result.episodes:
        rp = result.episodes[0].round_payoffs
        if rp:
            kwargs["round_payoffs_json"] = [dict(r) for r in rp]

    # Day aggregates — computed from episode 0's step history when the
    # El Farol gate (typed actions on episode 0) is open. Use
    # ``game.name`` (not ``result.game_name``) so the synthesised
    # ``match_id`` matches the runner's per-ActionRecord ``match_id``
    # format: ``{game.name}#{run_id}#ep{episode}``.
    if result.episodes and result.episodes[0].actions:
        aggregates = _compute_day_aggregates(
            game_name=getattr(game, "name", result.game_name),
            run_id=run_id,
            episode_index=result.episodes[0].episode,
            history=result.episodes[0].history,
            capacity_threshold=getattr(game_cfg, "capacity_threshold", None),
        )
        if aggregates:
            kwargs["day_aggregates_json"] = aggregates

    return kwargs


def _serialize_action_records(episodes: list[Any]) -> list[dict[str, Any]]:
    """Flatten typed ``ActionRecord`` lists across episodes.

    Converts each record with :func:`dataclasses.asdict` and replaces any
    ``submitted_at`` datetime with its ISO 8601 string form so the
    payload is directly ``json.dumps``-safe.
    """
    from dataclasses import asdict as _asdict
    from datetime import datetime as _dt

    out: list[dict[str, Any]] = []
    for ep in episodes:
        for rec in getattr(ep, "actions", None) or []:
            payload = _asdict(rec)
            submitted_at = payload.get("submitted_at")
            if isinstance(submitted_at, _dt):
                payload["submitted_at"] = submitted_at.isoformat()
            out.append(payload)
    return out


def _compute_day_aggregates(
    *,
    game_name: str,
    run_id: str | None,
    episode_index: int,
    history: list[dict[str, Any]],
    capacity_threshold: int | None,
) -> list[dict[str, Any]]:
    """Derive per-day attendance aggregates from El Farol step history.

    Each step in ``history`` is the serialised form of a ``StepResult``
    whose ``state.public_state.attendance_history`` is the cumulative
    list of per-slot occupancy arrays. The last element is the day that
    step resolved. Returns an empty list when the history does not carry
    the expected El Farol public-state shape (e.g. PD) so callers can
    decide to omit the dashboard column entirely.
    """
    rid = run_id if run_id is not None else "run"
    match_id = f"{game_name}#{rid}#ep{episode_index}"

    aggregates: list[dict[str, Any]] = []
    for idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        state = entry.get("state")
        if not isinstance(state, dict):
            continue
        public = state.get("public_state")
        if not isinstance(public, dict):
            continue
        attendance_history = public.get("attendance_history")
        if not isinstance(attendance_history, list) or not attendance_history:
            continue
        last = attendance_history[-1]
        if not isinstance(last, (list, tuple)):
            continue
        try:
            slot_attendance = [int(x) for x in last]
        except (TypeError, ValueError):
            continue
        if capacity_threshold is not None:
            over_slots = sum(1 for occ in slot_attendance if occ >= capacity_threshold)
        else:
            over_slots = 0
        aggregates.append(
            {
                "match_id": match_id,
                "day": idx + 1,
                "slot_attendance": slot_attendance,
                "over_slots": over_slots,
                "total_attendances": sum(slot_attendance),
            }
        )
    return aggregates


async def _store_game_result(
    result: Any,
    suite: Any,
    config: Any,
    game: Any,
) -> None:
    """Store game result in dashboard database (if available).

    Silently skips if dashboard is not installed. The ``config`` argument
    is kept for backward compatibility; authoritative run metadata is
    read from ``result.config``.
    """
    try:
        from atp.dashboard import init_database
        from atp.dashboard.models import GameResult

        db = await init_database()
        async with db.session() as session:
            kwargs = _build_game_result_kwargs(result, suite, game)
            game_result = GameResult(**kwargs)
            session.add(game_result)
            await session.commit()
            click.echo(f"Results saved to dashboard (game_id={game_result.id})")

    except ImportError:
        pass  # Dashboard not installed
    except Exception as e:
        click.echo(f"Warning: failed to save to dashboard: {e}", err=True)


async def _store_tournament_result(
    result: Any,
    suite: Any,
    mode: str,
    config: Any,
) -> None:
    """Store tournament result in dashboard database (if available).

    Silently skips if dashboard is not installed.
    """
    try:
        from atp.dashboard import init_database
        from atp.dashboard.models import TournamentResult

        db = await init_database()
        async with db.session() as session:
            from datetime import datetime

            standings = [s.to_dict() for s in result.standings]
            matchups = [m.to_dict() for m in result.matches]

            tournament = TournamentResult(
                name=suite.name,
                game_name=suite.game.type,
                tournament_type=mode,
                num_agents=len(result.standings),
                episodes_per_matchup=config.episodes,
                status="completed",
                completed_at=datetime.now(),
                standings_json=standings,
                matchups_json=matchups,
            )
            session.add(tournament)
            await session.commit()
            click.echo(f"Results saved to dashboard (tournament_id={tournament.id})")

    except ImportError:
        pass  # Dashboard not installed
    except Exception as e:
        click.echo(f"Warning: failed to save to dashboard: {e}", err=True)


@game_command.command(name="list")
def game_list() -> None:
    """List available game types.

    Examples:

      atp game list
    """
    try:
        from game_envs import GameRegistry  # pyrefly: ignore[missing-import]

        games = GameRegistry.list_games(with_metadata=True)
        click.echo("Available Games:")
        click.echo("-" * 50)
        for info in games:
            if isinstance(info, dict):
                click.echo(f"  {info['name']:<20} {info.get('description', '')}")
            else:
                click.echo(f"  {info}")
    except ImportError:
        click.echo(
            "Error: game-environments package not installed",
            err=True,
        )
        sys.exit(2)


@game_command.command(name="info")
@click.argument("game_name")
def game_info(game_name: str) -> None:
    """Show detailed information about a game.

    Examples:

      atp game info prisoners_dilemma
      atp game info auction
    """
    try:
        from game_envs import GameRegistry  # pyrefly: ignore[missing-import]

        info = GameRegistry.game_info(game_name)
        click.echo(f"Game: {info['name']}")
        click.echo(f"Type: {info['game_type']}")
        click.echo(f"Move Order: {info['move_order']}")
        click.echo(f"Players: {', '.join(info['player_ids'])}")
        click.echo()

        if info.get("description"):
            click.echo(f"Description:\n  {info['description']}")
            click.echo()

        click.echo("Action Spaces:")
        for pid, desc in info.get("action_spaces", {}).items():
            click.echo(f"  {pid}: {desc}")

        click.echo()
        click.echo("Config Schema:")
        for field_name, field_info in info.get("config_schema", {}).items():
            default = field_info.get("default")
            ftype = field_info.get("type", "unknown")
            click.echo(f"  {field_name}: {ftype} (default: {default})")

    except ImportError:
        click.echo(
            "Error: game-environments package not installed",
            err=True,
        )
        sys.exit(2)
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
