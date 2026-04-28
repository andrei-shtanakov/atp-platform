"""GameRunner orchestrates multi-agent game execution."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from typing import Any

from atp.adapters.base import AgentAdapter
from atp.protocol.models import ATPRequest, Task
from game_envs.core.game import Game, MoveOrder

from atp_games.mapping.action_mapper import ActionMapper
from atp_games.mapping.observation_mapper import ObservationMapper
from atp_games.models import (
    ActionRecord,
    AgentRecord,
    EpisodeResult,
    GameResult,
    GameRunConfig,
    IntervalPair,
)
from atp_games.runner.action_validator import ActionValidator
from atp_games.runner.builtin_adapter import BuiltinAdapter

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Reports progress of multi-episode game execution.

    Logs episode completion with timing and ETA.
    """

    def __init__(self, total_episodes: int) -> None:
        self.total = total_episodes
        self.completed = 0
        self._start_time = time.monotonic()
        self._lock = asyncio.Lock()

    async def report_complete(self, episode: int) -> None:
        """Report that an episode has completed."""
        async with self._lock:
            self.completed += 1
            elapsed = time.monotonic() - self._start_time
            remaining = self.total - self.completed
            if self.completed > 0 and remaining > 0:
                avg_time = elapsed / self.completed
                eta = avg_time * remaining
                logger.info(
                    "Episode %d/%d complete (%.1fs elapsed, ETA %.1fs)",
                    self.completed,
                    self.total,
                    elapsed,
                    eta,
                )
            else:
                logger.info(
                    "Episode %d/%d complete (%.1fs elapsed)",
                    self.completed,
                    self.total,
                    elapsed,
                )

    @property
    def elapsed(self) -> float:
        """Total elapsed time in seconds."""
        return time.monotonic() - self._start_time


def _action_to_interval_pair(
    action: Any,
    *,
    num_slots: int,
    max_total_slots: int,
) -> IntervalPair | None:
    """Convert a canonical El Farol action to an IntervalPair.

    Accepts either ``{"intervals": [[start, end], ...]}`` or a bare list
    of ``[start, end]`` pairs. Returns ``None`` when the action is not
    interval-shaped (e.g. PD's string ``"cooperate"``) or when
    ``IntervalPair`` rejects the values (out-of-range, total slots
    exceeded, more than two runs).

    ``max_total_slots`` is stored on the returned ``IntervalPair`` so the
    action metadata stays faithful to the match's configured cap.
    """
    if isinstance(action, dict):
        action = action.get("intervals")
    if action is None:
        return None
    if not isinstance(action, (list, tuple)):
        return None

    pairs: list[tuple[int, int]] = []
    for pair in action:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        start, end = pair[0], pair[1]
        if not (isinstance(start, int) and not isinstance(start, bool)):
            return None
        if not (isinstance(end, int) and not isinstance(end, bool)):
            return None
        pairs.append((int(start), int(end)))

    if len(pairs) > 2:
        return None

    ordered = sorted(pairs, key=lambda p: p[0])
    first: tuple[int, int] | tuple[()] = ordered[0] if ordered else ()
    second: tuple[int, int] | tuple[()] = ordered[1] if len(ordered) == 2 else ()
    try:
        return IntervalPair(
            first=first,
            second=second,
            num_slots=num_slots,
            max_total_slots=max_total_slots,
        )
    except ValueError:
        return None


def _make_game_for_episode(
    game: Game,
    seed: int | None,
) -> Game:
    """Create a game instance configured for a specific episode.

    Uses deep copy to ensure independent state, then applies
    the episode-specific seed via dataclasses.replace to
    preserve the config subclass (e.g. PDConfig).

    Args:
        game: Template game instance.
        seed: Seed to use for this episode, or None.

    Returns:
        A fresh game instance ready for play.
    """
    import dataclasses
    import random

    episode_game = copy.deepcopy(game)
    if seed is not None:
        new_config = dataclasses.replace(episode_game.config, seed=seed)
        episode_game.config = new_config  # type: ignore[misc]
        episode_game._rng = random.Random(seed)
    return episode_game


class GameRunner:
    """Orchestrates multi-agent game execution via ATP.

    Manages the game loop: reset -> observe -> map ->
    send -> validate -> step -> repeat until terminal.

    Supports simultaneous and sequential move games,
    parallel episode execution, and deterministic seeding.
    """

    def __init__(
        self,
        observation_mapper: ObservationMapper | None = None,
        action_mapper: ActionMapper | None = None,
        action_validator: ActionValidator | None = None,
    ) -> None:
        self.observation_mapper = observation_mapper or ObservationMapper()
        self.action_mapper = action_mapper or ActionMapper()
        self.action_validator = action_validator or ActionValidator()

    async def run_game(
        self,
        game: Game,
        agents: dict[str, AgentAdapter],
        config: GameRunConfig | None = None,
    ) -> GameResult:
        """Run a complete game with multiple episodes.

        Supports parallel episode execution via config.parallel
        and deterministic seeding via config.base_seed.

        Args:
            game: The game environment instance.
            agents: Mapping of player_id to agent adapter.
            config: Run configuration (episodes, retries,
                parallel, base_seed).

        Returns:
            GameResult with per-episode results and
            aggregated statistics.
        """
        config = config or GameRunConfig()
        self.action_validator.max_retries = config.max_retries

        run_id = str(uuid.uuid4())

        agent_names = {
            pid: self._get_agent_name(pid, adapter) for pid, adapter in agents.items()
        }
        agent_records = [
            AgentRecord.from_legacy(agent_id=pid, display_name=name)
            for pid, name in agent_names.items()
        ]

        progress = ProgressReporter(config.episodes)

        if config.parallel <= 1:
            episodes = await self._run_sequential(
                game, agents, config, progress, run_id
            )
        else:
            episodes = await self._run_parallel(game, agents, config, progress, run_id)

        logger.info(
            "Game '%s' complete: %d episodes in %.1fs",
            game.name,
            len(episodes),
            progress.elapsed,
        )

        return GameResult(
            game_name=game.name,
            config=config,
            episodes=episodes,
            agent_names=agent_names,
            agents=agent_records,
            run_id=run_id,
        )

    async def _run_sequential(
        self,
        game: Game,
        agents: dict[str, AgentAdapter],
        config: GameRunConfig,
        progress: ProgressReporter,
        run_id: str,
    ) -> list[EpisodeResult]:
        """Run episodes sequentially."""
        episodes: list[EpisodeResult] = []
        for idx in range(config.episodes):
            seed = config.episode_seed(idx)
            episode_game = _make_game_for_episode(game, seed)
            episode_agents = self._copy_agents(agents)
            result = await self._run_episode(
                episode_game, episode_agents, config, idx, seed, run_id
            )
            episodes.append(result)
            await progress.report_complete(idx)
        return episodes

    async def _run_parallel(
        self,
        game: Game,
        agents: dict[str, AgentAdapter],
        config: GameRunConfig,
        progress: ProgressReporter,
        run_id: str,
    ) -> list[EpisodeResult]:
        """Run episodes with bounded parallelism."""
        semaphore = asyncio.Semaphore(config.parallel)
        results: list[EpisodeResult | None] = [None] * config.episodes

        async def run_one(idx: int) -> None:
            async with semaphore:
                seed = config.episode_seed(idx)
                episode_game = _make_game_for_episode(game, seed)
                episode_agents = self._copy_agents(agents)
                result = await self._run_episode(
                    episode_game,
                    episode_agents,
                    config,
                    idx,
                    seed,
                    run_id,
                )
                results[idx] = result
                await progress.report_complete(idx)

        tasks = [run_one(i) for i in range(config.episodes)]
        await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    def _copy_agents(
        self,
        agents: dict[str, AgentAdapter],
    ) -> dict[str, AgentAdapter]:
        """Create independent copies of agents for an episode.

        BuiltinAdapters are deep-copied to ensure independent
        strategy state. External adapters are shared.
        """
        result: dict[str, AgentAdapter] = {}
        for pid, adapter in agents.items():
            if isinstance(adapter, BuiltinAdapter):
                result[pid] = copy.deepcopy(adapter)
            else:
                result[pid] = adapter
        return result

    async def _run_episode(
        self,
        game: Game,
        agents: dict[str, AgentAdapter],
        config: GameRunConfig,
        episode: int,
        seed: int | None = None,
        run_id: str | None = None,
    ) -> EpisodeResult:
        """Run a single game episode.

        Args:
            game: The game environment (episode-specific copy).
            agents: Agent adapters keyed by player_id.
            config: Run configuration.
            episode: Episode index.
            seed: Seed used for this episode.

        Returns:
            EpisodeResult for this episode.
        """
        # Reset strategies for builtin adapters
        for adapter in agents.values():
            if isinstance(adapter, BuiltinAdapter):
                adapter.reset()

        game.reset()
        history: list[dict[str, Any]] = []
        actions_log: list[dict[str, Any]] = []
        action_records: list[ActionRecord] = []
        round_payoffs: list[dict[str, float]] = []

        # Include run_id so independent runs of the same config don't
        # collide on match_id. Fall back to a fresh uuid4 if a caller
        # invokes _run_episode directly (only happens in older tests).
        resolved_run_id = run_id if run_id is not None else str(uuid.uuid4())
        match_id = f"{game.name}#{resolved_run_id}#ep{episode}"
        day = 0

        while not game.is_terminal:
            observations = {pid: game.observe(pid) for pid in game.player_ids}

            if game.move_order == MoveOrder.SIMULTANEOUS:
                actions, intents = await self._parallel_moves(
                    observations,
                    agents,
                    game,
                    episode,
                    config,
                )
            else:
                actions, intents = await self._sequential_moves(
                    observations,
                    agents,
                    game,
                    episode,
                    config,
                )

            step_result = game.step(actions)
            history.append(step_result.to_dict())
            actions_log.append(dict(actions))
            round_payoffs.append(
                {pid: float(v) for pid, v in step_result.payoffs.items()}
            )

            day += 1
            self._append_action_records(
                records=action_records,
                match_id=match_id,
                day=day,
                actions=actions,
                intents=intents,
                step_result=step_result,
                game=game,
            )

        return EpisodeResult(
            episode=episode,
            payoffs=game.get_payoffs(),
            history=history,
            actions_log=actions_log,
            actions=action_records,
            round_payoffs=round_payoffs,
            seed=seed,
        )

    def _append_action_records(
        self,
        *,
        records: list[ActionRecord],
        match_id: str,
        day: int,
        actions: dict[str, Any],
        intents: dict[str, str | None],
        step_result: Any,
        game: Game,
    ) -> None:
        """Build per-agent ActionRecord for this day when possible.

        ActionRecords are built for El Farol whose canonical action shape
        is ``{"intervals": [[start, end], ...]}`` (or the bare pair-list
        form). Non-interval games (e.g. PD) bail out entirely and produce
        an empty ``EpisodeResult.actions``.
        """
        num_slots = self._infer_num_slots(game, actions)
        max_total_slots = self._infer_max_total_slots(game, num_slots)
        crowded_slots = self._extract_crowded_slots(step_result)

        # First action determines whether this game is interval-shaped at all.
        # If the inaugural action isn't interval-convertible, bail entirely.
        first_intervals: IntervalPair | None = None
        first_pid: str | None = None
        for pid, action in actions.items():
            first_intervals = _action_to_interval_pair(
                action,
                num_slots=num_slots,
                max_total_slots=max_total_slots,
            )
            first_pid = pid
            break
        if first_intervals is None:
            return  # non-interval game (PD etc.)

        for pid, action in actions.items():
            if pid == first_pid:
                intervals = first_intervals
            else:
                intervals = _action_to_interval_pair(
                    action,
                    num_slots=num_slots,
                    max_total_slots=max_total_slots,
                )
                if intervals is None:
                    continue  # malformed action for an otherwise-interval game

            picks = intervals.covered_slots()
            payoff = float(step_result.payoffs.get(pid, 0.0))
            if crowded_slots is not None:
                num_over = sum(1 for s in picks if s in crowded_slots)
                num_under = len(picks) - num_over
            else:
                num_under = len(picks)
                num_over = 0

            records.append(
                ActionRecord(
                    match_id=match_id,
                    day=day,
                    agent_id=pid,
                    intervals=intervals,
                    picks=picks,
                    num_visits=intervals.num_visits(),
                    total_slots=intervals.total_slots(),
                    payoff=payoff,
                    num_under=num_under,
                    num_over=num_over,
                    intent=intents.get(pid),
                )
            )

    @staticmethod
    def _infer_num_slots(game: Game, actions: dict[str, Any]) -> int:
        """Best-effort extraction of num_slots from game config or actions."""
        cfg = getattr(game, "config", None)
        num_slots = getattr(cfg, "num_slots", None)
        if isinstance(num_slots, int) and num_slots > 0:
            return num_slots
        # Fallback: infer from the largest slot index observed across
        # canonical interval-shaped actions.
        max_slot = -1
        for action in actions.values():
            payload = (
                action.get("intervals") if isinstance(action, dict) else action
            )
            if not isinstance(payload, (list, tuple)):
                continue
            for pair in payload:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                end = pair[1]
                if isinstance(end, int) and end > max_slot:
                    max_slot = end
        return max_slot + 1 if max_slot >= 0 else 16

    @staticmethod
    def _infer_max_total_slots(game: Game, num_slots: int) -> int:
        """Read the match's configured slot cap, falling back to num_slots.

        Prefers ``game.config.max_total_slots`` so persisted
        :class:`IntervalPair` instances carry the real rule that produced
        them. When the game config does not expose the attribute (e.g.
        non-El-Farol slot games), falls back to ``num_slots`` which is a
        valid upper bound for any action the space can accept.
        """
        cfg = getattr(game, "config", None)
        value = getattr(cfg, "max_total_slots", None)
        if isinstance(value, int) and value > 0:
            return value
        return num_slots

    @staticmethod
    def _extract_crowded_slots(step_result: Any) -> set[int] | None:
        """Pull ``crowded_slots_today`` from the step-result public state."""
        state = getattr(step_result, "state", None)
        public = getattr(state, "public_state", None)
        if not isinstance(public, dict):
            return None
        crowded = public.get("crowded_slots_today")
        if crowded is None:
            return None
        try:
            return {int(s) for s in crowded}
        except (TypeError, ValueError):
            return None

    async def _parallel_moves(
        self,
        observations: dict[str, Any],
        agents: dict[str, AgentAdapter],
        game: Game,
        episode: int,
        config: GameRunConfig,
    ) -> tuple[dict[str, Any], dict[str, str | None]]:
        """Send requests to all agents in parallel.

        For simultaneous-move games, all agents are queried
        concurrently. Returns both the resolved actions and
        the per-agent Tier-1 intent strings (``None`` when absent).
        """
        tasks = {
            pid: self._get_validated_action(
                pid,
                agents[pid],
                observations[pid],
                game,
                episode,
                config,
            )
            for pid in observations
        }

        results = await asyncio.gather(*tasks.values())
        actions: dict[str, Any] = {}
        intents: dict[str, str | None] = {}
        for pid, (action, intent) in zip(tasks.keys(), results):
            actions[pid] = action
            intents[pid] = intent
        return actions, intents

    async def _sequential_moves(
        self,
        observations: dict[str, Any],
        agents: dict[str, AgentAdapter],
        game: Game,
        episode: int,
        config: GameRunConfig,
    ) -> tuple[dict[str, Any], dict[str, str | None]]:
        """Query agents sequentially in player order.

        For sequential-move games, each agent is queried
        one at a time in the order of player_ids. Returns both
        the resolved actions and the per-agent Tier-1 intent
        strings (``None`` when absent).
        """
        actions: dict[str, Any] = {}
        intents: dict[str, str | None] = {}
        for pid in game.player_ids:
            if pid in observations:
                action, intent = await self._get_validated_action(
                    pid,
                    agents[pid],
                    observations[pid],
                    game,
                    episode,
                    config,
                )
                actions[pid] = action
                intents[pid] = intent
        return actions, intents

    async def _get_validated_action(
        self,
        player_id: str,
        agent: AgentAdapter,
        observation: Any,
        game: Game,
        episode: int,
        config: GameRunConfig,
    ) -> tuple[Any, str | None]:
        """Get a validated action from an agent.

        Sends the observation to the agent, validates the
        response, and retries on invalid actions up to
        max_retries times. Falls back to a default action
        if all retries fail. Returns the resolved action
        alongside the agent's Tier-1 ``intent`` self-report
        (``None`` when absent or when the fallback path is
        taken).
        """
        action_space = game.action_space(player_id)
        request = self.observation_mapper.to_atp_request(
            observation, game.name, episode
        )

        for attempt in range(config.max_retries + 1):
            try:
                response = await agent.execute(request)
                game_action = self.action_mapper.from_atp_response(response)
                result = self.action_validator.validate(game_action, action_space)

                if result.valid:
                    return game_action.action, game_action.intent

                if attempt < config.max_retries:
                    error_msg = result.errors[0]
                    logger.warning(
                        "Player %s attempt %d: %s",
                        player_id,
                        attempt + 1,
                        error_msg,
                    )
                    request = self._add_error_context(request, error_msg, action_space)
            except ValueError as e:
                logger.warning(
                    "Player %s attempt %d error: %s",
                    player_id,
                    attempt + 1,
                    e,
                )
                if attempt < config.max_retries:
                    request = self._add_error_context(
                        request,
                        str(e),
                        action_space,
                    )

        # Fallback to default action
        default = self.action_validator.get_default_action(action_space, game._rng)
        logger.warning(
            "Player %s: all retries exhausted, using default action: %s",
            player_id,
            default.action,
        )
        return default.action, None

    def _add_error_context(
        self,
        request: ATPRequest,
        error: str,
        action_space: Any,
    ) -> ATPRequest:
        """Create a new request with error context for retry.

        Appends the validation error and valid actions
        to the task description.
        """
        retry_prompt = self.action_validator.build_retry_prompt(error, action_space)
        new_description = f"{request.task.description}\n\n--- RETRY ---\n{retry_prompt}"
        return ATPRequest(
            task_id=request.task_id,
            task=Task(description=new_description),
            context=request.context,
            constraints=request.constraints,
            metadata=request.metadata,
        )

    def _get_agent_name(
        self,
        player_id: str,
        adapter: AgentAdapter,
    ) -> str:
        """Get a human-readable name for an agent."""
        if isinstance(adapter, BuiltinAdapter):
            return adapter.strategy.name
        return f"{adapter.adapter_type}:{player_id}"
