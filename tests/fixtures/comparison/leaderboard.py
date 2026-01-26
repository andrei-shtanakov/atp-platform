"""Mock data generator for leaderboard tests.

This module provides utilities for generating realistic leaderboard data
including multiple agents, tests, and score matrices for testing the
leaderboard matrix visualization.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, TypedDict

from atp.dashboard.models import Agent, RunResult, SuiteExecution, TestExecution
from tests.fixtures.comparison.events import generate_realistic_event_sequence
from tests.fixtures.comparison.factories import (
    AgentFactory,
    RunResultFactory,
    SuiteExecutionFactory,
    TestExecutionFactory,
)


class TestDefinition(TypedDict):
    """Typed dict for test definition."""

    test_id: str
    test_name: str
    tags: list[str]
    difficulty_modifier: float


@dataclass
class LeaderboardScenario:
    """Container for a complete leaderboard test scenario."""

    agents: list[Agent]
    suite_executions: list[SuiteExecution]
    test_executions: list[TestExecution]
    run_results: list[RunResult]
    score_matrix: dict[str, dict[str, float | None]]  # test_id -> agent_name -> score


def create_leaderboard_scenario(
    suite_name: str = "benchmark-suite",
    num_agents: int = 3,
    num_tests: int = 5,
    runs_per_test: int = 1,
    include_events: bool = True,
    event_steps: int = 5,
    seed: int | None = None,
) -> LeaderboardScenario:
    """Create a complete leaderboard scenario with all related data.

    This function generates a realistic set of test data for leaderboard
    testing, including:
    - Multiple agents with different performance characteristics
    - Suite executions for each agent
    - Test executions with varying scores
    - Run results with optional events

    Args:
        suite_name: Name of the test suite.
        num_agents: Number of agents to create.
        num_tests: Number of tests in the suite.
        runs_per_test: Number of runs per test.
        include_events: Whether to include events_json in run results.
        event_steps: Number of steps in event sequences.
        seed: Random seed for reproducibility.

    Returns:
        LeaderboardScenario with all generated data.
    """
    if seed is not None:
        random.seed(seed)

    agents: list[Agent] = []
    suite_executions: list[SuiteExecution] = []
    test_executions: list[TestExecution] = []
    run_results: list[RunResult] = []
    score_matrix: dict[str, dict[str, float | None]] = {}

    # Create agents with different "skill levels"
    agent_skill = {}
    for i in range(num_agents):
        agent = AgentFactory.create(
            name=f"agent-{chr(65 + i)}",  # agent-A, agent-B, etc.
            agent_type="http",
            description=f"Test agent {chr(65 + i)} for leaderboard testing",
        )
        agents.append(agent)
        # Assign skill level (affects average scores)
        agent_skill[agent.name] = 0.6 + (i * 0.15)  # 0.6, 0.75, 0.9

    # Create test definitions with difficulty
    test_definitions: list[TestDefinition] = []
    for i in range(num_tests):
        test_id = f"test-{i + 1:03d}"
        # Vary difficulty
        difficulty = ["easy", "medium", "hard", "very_hard"][i % 4]
        difficulty_modifier = {
            "easy": 0.2,
            "medium": 0.0,
            "hard": -0.15,
            "very_hard": -0.3,
        }[difficulty]
        test_def: TestDefinition = {
            "test_id": test_id,
            "test_name": (f"Test Case {i + 1}: {difficulty.replace('_', ' ').title()}"),
            "tags": [difficulty, f"category-{i % 3 + 1}"],
            "difficulty_modifier": difficulty_modifier,
        }
        test_definitions.append(test_def)
        score_matrix[test_id] = {}

    # Create suite executions and test results for each agent
    base_time = datetime.now() - timedelta(hours=num_agents)

    for agent_idx, agent in enumerate(agents):
        suite_start = base_time + timedelta(hours=agent_idx)
        passed_tests = 0
        failed_tests = 0

        suite_exec = SuiteExecutionFactory.create(
            suite_name=suite_name,
            agent_id=agent_idx + 1,  # Assuming sequential IDs
            started_at=suite_start,
            runs_per_test=runs_per_test,
            total_tests=num_tests,
            status="running",  # Will update after processing tests
        )
        suite_executions.append(suite_exec)

        for test_def in test_definitions:
            test_start = suite_start + timedelta(minutes=5 * len(test_executions))

            # Calculate score based on agent skill and test difficulty
            base_score = agent_skill[agent.name] + test_def["difficulty_modifier"]
            # Add some randomness
            score = max(0.0, min(1.0, base_score + random.gauss(0, 0.1)))
            score_100 = score * 100

            # Determine success (score > 50%)
            success = score > 0.5
            if success:
                passed_tests += 1
            else:
                failed_tests += 1

            score_matrix[test_def["test_id"]][agent.name] = score_100

            test_exec = TestExecutionFactory.create(
                suite_execution_id=len(suite_executions),
                test_id=test_def["test_id"],
                test_name=test_def["test_name"],
                tags=test_def["tags"],
                started_at=test_start,
                total_runs=runs_per_test,
                successful_runs=runs_per_test if success else 0,
                success=success,
                score=score_100,
                status="completed",
            )
            test_executions.append(test_exec)

            # Create run results
            for run_num in range(1, runs_per_test + 1):
                run_start = test_start + timedelta(minutes=run_num)
                events_json = None

                if include_events:
                    events_json = generate_realistic_event_sequence(
                        task_id=test_def["test_id"],
                        num_steps=event_steps,
                        include_error=not success,
                        start_time=run_start,
                    )

                run = RunResultFactory.create(
                    test_execution_id=len(test_executions),
                    run_number=run_num,
                    started_at=run_start,
                    success=success,
                    events_json=events_json,
                    total_tokens=1000 + random.randint(0, 500),
                    cost_usd=0.01 + random.random() * 0.02,
                )
                run_results.append(run)

        # Update suite execution summary
        suite_exec.passed_tests = passed_tests
        suite_exec.failed_tests = failed_tests
        suite_exec.success_rate = passed_tests / num_tests if num_tests > 0 else 0.0
        suite_exec.status = "completed"
        completed_at = datetime.now()
        suite_exec.completed_at = completed_at
        suite_exec.duration_seconds = (completed_at - suite_start).total_seconds()

    return LeaderboardScenario(
        agents=agents,
        suite_executions=suite_executions,
        test_executions=test_executions,
        run_results=run_results,
        score_matrix=score_matrix,
    )


def generate_leaderboard_data(
    suite_name: str = "benchmark-suite",
    agent_configs: list[dict[str, Any]] | None = None,
    test_configs: list[dict[str, Any]] | None = None,
    seed: int | None = None,
) -> LeaderboardScenario:
    """Generate leaderboard data from specific configurations.

    Allows fine-grained control over agent and test configurations
    for testing specific scenarios (e.g., all agents failing a test).

    Args:
        suite_name: Name of the test suite.
        agent_configs: List of agent configuration dicts:
            - name: Agent name
            - skill: Base skill level (0-1)
        test_configs: List of test configuration dicts:
            - test_id: Test identifier
            - test_name: Human-readable name
            - difficulty: "easy", "medium", "hard", "very_hard"
            - tags: List of tags
        seed: Random seed for reproducibility.

    Returns:
        LeaderboardScenario with generated data.

    Example:
        >>> scenario = generate_leaderboard_data(
        ...     agent_configs=[
        ...         {"name": "claude", "skill": 0.9},
        ...         {"name": "gpt-4", "skill": 0.85},
        ...     ],
        ...     test_configs=[
        ...         {"test_id": "code-001", "difficulty": "hard"},
        ...         {"test_id": "text-001", "difficulty": "easy"},
        ...     ],
        ... )
    """
    if seed is not None:
        random.seed(seed)

    # Default configurations if not provided
    if agent_configs is None:
        agent_configs = [
            {"name": "agent-A", "skill": 0.85},
            {"name": "agent-B", "skill": 0.75},
            {"name": "agent-C", "skill": 0.65},
        ]

    if test_configs is None:
        test_configs = [
            {"test_id": "test-001", "test_name": "Simple Task", "difficulty": "easy"},
            {"test_id": "test-002", "test_name": "Medium Task", "difficulty": "medium"},
            {"test_id": "test-003", "test_name": "Hard Task", "difficulty": "hard"},
        ]

    difficulty_modifiers = {
        "easy": 0.2,
        "medium": 0.0,
        "hard": -0.15,
        "very_hard": -0.3,
    }

    agents: list[Agent] = []
    suite_executions: list[SuiteExecution] = []
    test_executions: list[TestExecution] = []
    run_results: list[RunResult] = []
    score_matrix: dict[str, dict[str, float | None]] = {}

    # Create agents
    for config in agent_configs:
        agent = AgentFactory.create(
            name=config["name"],
            agent_type=config.get("type", "http"),
            description=config.get("description"),
        )
        agents.append(agent)

    # Initialize score matrix
    for test_config in test_configs:
        score_matrix[test_config["test_id"]] = {}

    # Create suite executions and test results
    base_time = datetime.now() - timedelta(hours=len(agents))

    for agent_idx, (agent, agent_config) in enumerate(zip(agents, agent_configs)):
        suite_start = base_time + timedelta(hours=agent_idx)
        passed_tests = 0
        failed_tests = 0
        skill = agent_config.get("skill", 0.75)

        suite_exec = SuiteExecutionFactory.create(
            suite_name=suite_name,
            agent_id=agent_idx + 1,
            started_at=suite_start,
            total_tests=len(test_configs),
            status="running",
        )
        suite_executions.append(suite_exec)

        for test_config in test_configs:
            test_start = suite_start + timedelta(minutes=5 * len(test_executions))

            difficulty = test_config.get("difficulty", "medium")
            diff_mod = difficulty_modifiers.get(difficulty, 0.0)

            # Calculate score
            base_score = skill + diff_mod
            # Check for forced score in config
            if "forced_score" in test_config:
                score = test_config["forced_score"].get(agent.name)
                if score is None:
                    score = max(0.0, min(1.0, base_score + random.gauss(0, 0.1)))
            else:
                score = max(0.0, min(1.0, base_score + random.gauss(0, 0.1)))

            score_100 = score * 100
            success = score > 0.5

            if success:
                passed_tests += 1
            else:
                failed_tests += 1

            score_matrix[test_config["test_id"]][agent.name] = score_100

            test_exec = TestExecutionFactory.create(
                suite_execution_id=len(suite_executions),
                test_id=test_config["test_id"],
                test_name=test_config.get(
                    "test_name", f"Test {test_config['test_id']}"
                ),
                tags=test_config.get("tags", [difficulty]),
                started_at=test_start,
                success=success,
                score=score_100,
            )
            test_executions.append(test_exec)

            run = RunResultFactory.create(
                test_execution_id=len(test_executions),
                run_number=1,
                started_at=test_start,
                success=success,
                events_json=generate_realistic_event_sequence(
                    task_id=test_config["test_id"],
                    num_steps=3,
                    include_error=not success,
                    start_time=test_start,
                ),
            )
            run_results.append(run)

        # Update suite
        suite_exec.passed_tests = passed_tests
        suite_exec.failed_tests = failed_tests
        suite_exec.success_rate = (
            passed_tests / len(test_configs) if test_configs else 0.0
        )
        suite_exec.status = "completed"
        suite_exec.completed_at = datetime.now()

    return LeaderboardScenario(
        agents=agents,
        suite_executions=suite_executions,
        test_executions=test_executions,
        run_results=run_results,
        score_matrix=score_matrix,
    )


# Pre-built scenarios for common test cases
LEADERBOARD_SCENARIOS = {
    "basic": lambda: create_leaderboard_scenario(
        num_agents=3,
        num_tests=5,
        seed=42,
    ),
    "large": lambda: create_leaderboard_scenario(
        num_agents=5,
        num_tests=10,
        runs_per_test=3,
        seed=42,
    ),
    "hard_for_all": lambda: generate_leaderboard_data(
        agent_configs=[
            {"name": "agent-A", "skill": 0.4},
            {"name": "agent-B", "skill": 0.35},
            {"name": "agent-C", "skill": 0.3},
        ],
        test_configs=[
            {"test_id": "hard-001", "difficulty": "very_hard"},
            {"test_id": "hard-002", "difficulty": "very_hard"},
            {"test_id": "hard-003", "difficulty": "hard"},
        ],
        seed=42,
    ),
    "easy_suite": lambda: generate_leaderboard_data(
        agent_configs=[
            {"name": "agent-A", "skill": 0.9},
            {"name": "agent-B", "skill": 0.85},
        ],
        test_configs=[
            {"test_id": "easy-001", "difficulty": "easy"},
            {"test_id": "easy-002", "difficulty": "easy"},
            {"test_id": "easy-003", "difficulty": "medium"},
        ],
        seed=42,
    ),
    "mixed_performance": lambda: generate_leaderboard_data(
        agent_configs=[
            {"name": "specialist", "skill": 0.95},  # Very good
            {"name": "generalist", "skill": 0.7},  # Average
            {"name": "beginner", "skill": 0.4},  # Poor
        ],
        test_configs=[
            {"test_id": "test-001", "test_name": "Easy Task", "difficulty": "easy"},
            {"test_id": "test-002", "test_name": "Medium Task", "difficulty": "medium"},
            {"test_id": "test-003", "test_name": "Hard Task", "difficulty": "hard"},
            {
                "test_id": "test-004",
                "test_name": "Expert Task",
                "difficulty": "very_hard",
            },
        ],
        seed=42,
    ),
}
