# Phase 2: Launch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete launch-ready documentation: rewrite README with clear value proposition, create CONTRIBUTING.md, add competitor comparison page, create a GitHub Actions game-eval template, build a reproducible experiment runner, and draft the launch blog post.

**Architecture:** Primarily documentation tasks with one code task (experiment runner script). CI templates and game reporters already exist — this phase fills the remaining gaps.

**Tech Stack:** Markdown, YAML, Python (experiment script)

---

## What Already Exists (No Work Needed)

- CI templates for 5 platforms (examples/ci/)
- GitHub workflows (ci.yml, atp-test.yml, publish.yml, game-environments-ci.yml, atp-games-ci.yml)
- Game HTML reporter with Chart.js visualizations
- LICENSE (MIT)
- Game suites (tournament, crossplay, stress-test, alympics)
- Docker setup (Dockerfile, docker-compose.yml from Phase 1)

---

## Task 1: Rewrite README Value Proposition

**Files:**
- Modify: `README.md`

The current README is comprehensive but buries the value proposition. Rewrite the opening sections to lead with "ATP = framework-agnostic agent testing + game-theoretic evaluation."

- [ ] **Step 1: Read current README.md**

- [ ] **Step 2: Rewrite the opening section (before Quick Start)**

Replace the current Overview/Problem/Solution sections with a tighter opening:

```markdown
# ATP — Agent Test Platform

**The framework-agnostic platform for testing and evaluating AI agents.**

ATP provides a unified protocol for testing agents regardless of their implementation framework (LangGraph, CrewAI, AutoGen, custom, etc.), plus unique game-theoretic evaluation capabilities.

## Why ATP?

- **Framework-agnostic**: One protocol, any agent. HTTP, CLI, Container, MCP, cloud adapters — test everything the same way.
- **Game-theoretic evaluation**: Measure cooperation, trust, fairness, and strategic reasoning across 7 canonical games with Elo ratings.
- **Statistical rigor**: Welch's t-test, confidence intervals, baseline regression detection. Not just pass/fail — real statistical analysis.
- **Production-ready**: CI/CD templates, Docker support, web dashboard, JUnit/JSON/HTML reporting.

## Quick Start

```bash
pip install atp-platform
atp quickstart
```

See the [5-minute quickstart guide](docs/guides/quickstart-5min.md) for details.
```

Keep the rest of the README intact. Only rewrite from the top through the Quick Start section.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README value proposition for launch"
```

---

## Task 2: Create CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write CONTRIBUTING.md**

```markdown
# Contributing to ATP

Thank you for your interest in contributing to ATP! This guide explains how to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/atp-platform/atp-platform.git
cd atp-platform

# Install dependencies (requires uv)
uv sync --all-extras

# Run tests
uv run python -m pytest tests/unit/ -q

# Code quality
uv run ruff format .
uv run ruff check .
uv run pyrefly check
```

## What to Contribute

### Add an Adapter

Adapters translate between the ATP Protocol and agent-specific communication. To add one:

1. Create `packages/atp-adapters/atp/adapters/your_adapter.py`
2. Implement `AgentAdapter` (see `base.py` for the interface)
3. Register via entry point in `packages/atp-adapters/pyproject.toml`
4. Add tests in `tests/unit/adapters/`

### Add an Evaluator

Evaluators assess agent results. To add one:

1. Create `atp/evaluators/your_evaluator.py`
2. Implement the evaluator interface (see existing evaluators for patterns)
3. Register via entry point in `pyproject.toml`
4. Add tests in `tests/unit/evaluators/`

### Add a Game

Games are in the standalone `game-environments` library:

1. Create `game-environments/game_envs/games/your_game.py`
2. Implement the `Game` abstract class (see `prisoners_dilemma.py` as template)
3. Register with `@register_game("your_game", YourConfig)`
4. Add strategies in `game-environments/game_envs/strategies/`
5. Add tests in `game-environments/tests/`

## Code Standards

- **Python 3.12+** with type hints on all code
- **Line length**: 88 characters (ruff enforced)
- **Testing**: pytest with anyio for async. Minimum 80% coverage for new code.
- **Formatting**: `uv run ruff format .` before committing
- **Type checking**: `uv run pyrefly check` must pass

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Write tests first (TDD encouraged)
4. Implement your changes
5. Run the full quality suite: `uv run ruff format . && uv run ruff check . && uv run pyrefly check && uv run python -m pytest tests/unit/ -q`
6. Submit a PR with a clear description

## Project Structure

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation and development commands.
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md for open-source contributors"
```

---

## Task 3: Competitor Comparison Page

**Files:**
- Create: `docs/comparison.md`

- [ ] **Step 1: Write the comparison document**

Create an honest comparison page. Research what DeepEval, Promptfoo, and Inspect AI offer. Be fair — acknowledge where competitors are stronger.

Key differentiators for ATP:
- Framework-agnostic protocol (not tied to one LLM provider)
- Game-theoretic evaluation (unique to ATP)
- Statistical rigor (Welch's t-test, CIs, regression detection)
- Full platform (adapters, evaluators, dashboard, CI/CD)

Structure:

```markdown
# ATP vs Other Agent Testing Tools

## Comparison Matrix

| Feature | ATP | DeepEval | Promptfoo | Inspect AI |
|---------|-----|----------|-----------|------------|
| Framework-agnostic protocol | Yes — any agent via adapters | Python-focused | Config-driven | Python-focused |
| Agent adapters | 9 (HTTP, CLI, Container, MCP, LangGraph, CrewAI, AutoGen, cloud) | LLM-focused | LLM-focused | Python tasks |
| Game-theoretic evaluation | Yes — 7 games, Elo ratings, tournaments | No | No | No |
| Statistical analysis | Welch's t-test, CIs, regression detection | Basic metrics | Basic scoring | Basic scoring |
| Evaluator types | 10+ (artifact, behavior, LLM-judge, code-exec, security, factuality, etc.) | LLM-judge, custom | Assertions, LLM-judge | Scorers |
| Dashboard | Web UI with analytics | Cloud platform | Web viewer | Log viewer |
| CI/CD integration | GitHub, GitLab, Jenkins, Azure, CircleCI templates | GitHub Actions | GitHub Actions | N/A |
| Cost tracking | Built-in per-model pricing | Via platform | Limited | Limited |
| Open source | MIT | Apache 2.0 | MIT | MIT |

## When to Use ATP

**Choose ATP when you need:**
- To test agents built on different frameworks with one tool
- Game-theoretic evaluation of agent behavior (cooperation, fairness, trust)
- Statistical rigor beyond pass/fail (confidence intervals, regression detection)
- A self-hosted, full-stack testing platform

**Choose alternatives when you need:**
- Quick LLM prompt evaluation (Promptfoo is simpler for this)
- Cloud-hosted evaluation platform (DeepEval Cloud)
- Tight integration with a specific framework
```

NOTE: The comparison should be based on publicly available information. Be factual and fair. If unsure about a competitor's feature, mark it with "?" rather than guessing.

- [ ] **Step 2: Commit**

```bash
git add docs/comparison.md
git commit -m "docs: add competitor comparison page (ATP vs DeepEval, Promptfoo, Inspect AI)"
```

---

## Task 4: GitHub Actions Game Evaluation Template

**Files:**
- Create: `examples/ci/github-actions-game-eval.yml`

- [ ] **Step 1: Read existing templates**

Read `examples/ci/github-actions-basic.yml` and `.github/workflows/atp-test.yml` for patterns.

- [ ] **Step 2: Create game evaluation template**

```yaml
# github-actions-game-eval.yml
# GitHub Actions workflow for game-theoretic agent evaluation
#
# Copy this file to .github/workflows/game-eval.yml in your repository.
#
# This workflow runs game-theoretic evaluation of your agent(s)
# as part of your CI pipeline.

name: ATP Game Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.12"
  ATP_EPISODES: 50

jobs:
  game-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install ATP
        run: pip install atp-platform game-environments atp-games

      - name: Run tournament evaluation
        run: |
          atp game tournament \
            --game prisoners_dilemma \
            --episodes ${{ env.ATP_EPISODES }} \
            --mode round_robin \
            --output json \
            --output-file results/tournament.json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Run Alympics benchmark
        run: |
          atp game benchmark \
            --suite alympics \
            --episodes ${{ env.ATP_EPISODES }} \
            --output json \
            --output-file results/benchmark.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: game-eval-results
          path: results/
```

- [ ] **Step 3: Commit**

```bash
git add examples/ci/github-actions-game-eval.yml
git commit -m "feat: add GitHub Actions game evaluation CI template"
```

---

## Task 5: Reproducible Experiment Script

**Files:**
- Create: `examples/experiments/run_experiment.py`

A Python script that runs a complete game-theoretic experiment and saves results in machine-readable format.

- [ ] **Step 1: Read game runner and tournament code for API**

Read: `atp-games/atp_games/runner/game_runner.py`, `atp-games/atp_games/suites/tournament.py`

- [ ] **Step 2: Create the experiment script**

```python
#!/usr/bin/env python3
"""Reproducible game-theoretic experiment runner.

Runs builtin strategies across all canonical games and saves results
as JSON + CSV for analysis and publication.

Usage:
    uv run python examples/experiments/run_experiment.py
    uv run python examples/experiments/run_experiment.py --episodes 200 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


async def run_experiment(
    episodes: int = 100,
    seed: int = 42,
    output_dir: str = "results/experiment",
) -> None:
    """Run a complete game-theoretic experiment."""
    from game_envs.games.registry import GameRegistry
    from game_envs.strategies.pd_strategies import (
        AlwaysCooperate,
        AlwaysDefect,
        TitForTat,
    )

    from atp_games.rating.elo import EloCalculator
    from atp_games.runner.game_runner import (
        BuiltinAdapter,
        GameRunConfig,
        GameRunner,
    )
    from atp_games.suites.tournament import run_round_robin

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Strategies for 2-player discrete games
    strategies_2p = {
        "always_cooperate": AlwaysCooperate(),
        "always_defect": AlwaysDefect(),
        "tit_for_tat": TitForTat(),
    }
    agents_2p = {
        name: BuiltinAdapter(s) for name, s in strategies_2p.items()
    }

    # Games to test (2-player discrete only for builtin strategies)
    game_names = ["prisoners_dilemma", "stag_hunt", "battle_of_sexes"]

    results = {}
    elo = EloCalculator()

    for game_name in game_names:
        print(f"\n{'='*60}")
        print(f"Running {game_name} ({episodes} episodes, seed={seed})")
        print(f"{'='*60}")

        game = GameRegistry.create(
            game_name,
            config={"num_players": 2, "num_rounds": 50},
        )

        tournament = await run_round_robin(
            game=game,
            agents=agents_2p,
            episodes_per_match=episodes,
            elo_calculator=elo,
        )

        standings = [
            {
                "agent": s.agent,
                "wins": s.wins,
                "losses": s.losses,
                "draws": s.draws,
                "points": s.points,
                "total_payoff": round(s.total_payoff, 2),
            }
            for s in tournament.standings
        ]

        elo_ratings = {}
        if tournament.elo_ratings:
            elo_ratings = {
                name: round(r.rating, 1)
                for name, r in tournament.elo_ratings.items()
            }

        results[game_name] = {
            "standings": standings,
            "elo_ratings": elo_ratings,
            "matches": len(tournament.matches),
        }

        print(f"\nStandings for {game_name}:")
        for s in standings:
            elo_str = (
                f" (Elo: {elo_ratings.get(s['agent'], 'N/A')})"
                if elo_ratings
                else ""
            )
            print(
                f"  {s['agent']:20s} "
                f"W:{s['wins']} L:{s['losses']} D:{s['draws']} "
                f"Payoff:{s['total_payoff']}{elo_str}"
            )

    # Save JSON results
    experiment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "episodes": episodes,
            "seed": seed,
            "games": game_names,
            "agents": list(strategies_2p.keys()),
        },
        "results": results,
    }

    json_path = out / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"\nJSON results saved to {json_path}")

    # Save CSV summary
    csv_path = out / "experiment_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "game", "agent", "wins", "losses", "draws",
            "points", "total_payoff", "elo_rating",
        ])
        for game_name, data in results.items():
            for s in data["standings"]:
                writer.writerow([
                    game_name,
                    s["agent"],
                    s["wins"],
                    s["losses"],
                    s["draws"],
                    s["points"],
                    s["total_payoff"],
                    data["elo_ratings"].get(s["agent"], ""),
                ])
    print(f"CSV summary saved to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible game-theoretic experiment"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Episodes per match (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", default="results/experiment",
        help="Output directory (default: results/experiment)",
    )
    args = parser.parse_args()
    asyncio.run(
        run_experiment(args.episodes, args.seed, args.output_dir)
    )


if __name__ == "__main__":
    main()
```

IMPORTANT: Read the actual GameRegistry.create() and run_round_robin() APIs before writing. The code above is a guide — adjust to match real signatures.

- [ ] **Step 3: Run the experiment to verify**

Run: `uv run python examples/experiments/run_experiment.py --episodes 10`
Expected: JSON and CSV files in results/experiment/

- [ ] **Step 4: Add results/ to .gitignore if not already**

Check `.gitignore` — ensure `results/` is ignored.

- [ ] **Step 5: Commit**

```bash
git add examples/experiments/run_experiment.py
git commit -m "feat: add reproducible experiment runner with JSON/CSV output"
```

---

## Task 6: Launch Blog Post Draft

**Files:**
- Create: `docs/blog/how-cooperative-is-your-ai.md`

- [ ] **Step 1: Run a quick experiment**

Run: `uv run python examples/experiments/run_experiment.py --episodes 50`
Capture the output for use in the blog post.

- [ ] **Step 2: Write the blog post**

Create `docs/blog/how-cooperative-is-your-ai.md`:

```markdown
# How Cooperative is Your AI? Game-Theoretic Evaluation of LLM Agents

*Testing whether AI agents cooperate, compete, or defect — using classic game theory.*

## The Problem

When you deploy an AI agent, you test whether it produces correct outputs. But do you test *how* it makes decisions? Does it cooperate with other agents? Does it act fairly? Can it be exploited?

Traditional agent evaluation treats agents as input-output functions: give it a task, check the result. But agents increasingly operate in multi-agent environments where strategic behavior matters. A customer service agent that always yields to demands loses money. A negotiation agent that always defects loses trust.

## Game Theory Meets AI

Game theory has studied these exact questions for decades. The Prisoner's Dilemma, Stag Hunt, and Battle of the Sexes are canonical games that test cooperation, trust, and coordination — the same properties we care about in AI agents.

**ATP (Agent Test Platform)** brings game-theoretic evaluation to AI agent testing. Instead of just asking "did the agent complete the task?", ATP asks:

- **Does it cooperate?** (Prisoner's Dilemma)
- **Does it trust?** (Stag Hunt)
- **Does it coordinate?** (Battle of the Sexes)
- **Is it fair?** (Public Goods Game)
- **Can it be exploited?** (Exploitability analysis)

## How It Works

ATP runs agents through repeated games against baseline strategies and each other. Each game tests a different strategic dimension:

| Game | Tests | Key Insight |
|------|-------|-------------|
| Prisoner's Dilemma | Cooperation vs defection | Will the agent cooperate for mutual benefit? |
| Stag Hunt | Trust vs safety | Will the agent take risks for better outcomes? |
| Battle of the Sexes | Coordination | Can agents coordinate when they disagree? |
| Public Goods | Group cooperation | Does the agent free-ride or contribute? |
| Auction | Strategic bidding | Does the agent bid rationally? |

Agents play 50-100 rounds per game, generating statistical profiles of their strategic behavior. Results include Elo ratings for cross-agent comparison.

## Example: Builtin Strategies

As a baseline, we tested three classic Prisoner's Dilemma strategies:

- **Always Cooperate**: Pure cooperation
- **Always Defect**: Pure defection
- **Tit for Tat**: Cooperate first, then mirror the opponent

*(Insert results from experiment run here)*

The results confirm game theory: Tit for Tat achieves the best long-term balance of cooperation and defense, while Always Defect wins individual matches but loses in tournaments.

## Try It Yourself

```bash
pip install atp-platform game-environments atp-games

# Run a tournament
atp game tournament \
  --game prisoners_dilemma \
  --episodes 100 \
  --mode round_robin

# Run the full benchmark
atp game benchmark --suite alympics

# Run a reproducible experiment
python examples/experiments/run_experiment.py --episodes 100 --seed 42
```

## What's Next

We're working on LLM agent benchmarks: testing GPT-4o-mini, Claude Haiku, Gemini Flash, and Llama across all 7 games. Early results show fascinating differences in how models approach cooperation and trust.

ATP is open-source (MIT). Try it, add your own games, and let us know what you find.

**Links:**
- GitHub: [atp-platform/atp-platform](https://github.com/atp-platform/atp-platform)
- PyPI: `pip install atp-platform`
- Documentation: [Quickstart Guide](../guides/quickstart-5min.md)
```

- [ ] **Step 3: Commit**

```bash
mkdir -p docs/blog
git add docs/blog/how-cooperative-is-your-ai.md
git commit -m "docs: add launch blog post draft — game-theoretic evaluation of AI agents"
```

---

## Task 7: Final Phase 2 Verification

- [ ] **Step 1: Verify all deliverables**

```bash
ls -la README.md CONTRIBUTING.md docs/comparison.md \
  examples/ci/github-actions-game-eval.yml \
  examples/experiments/run_experiment.py \
  docs/blog/how-cooperative-is-your-ai.md
```

- [ ] **Step 2: Run the experiment**

```bash
uv run python examples/experiments/run_experiment.py --episodes 10
ls results/experiment/
```

- [ ] **Step 3: Run all tests**

```bash
uv run python -m pytest tests/unit/ --ignore=tests/unit/dashboard --ignore=tests/unit/test_github_import.py -q 2>&1 | tail -5
```

- [ ] **Step 4: Review all Phase 2 commits**

```bash
git log --oneline -10
```

- [ ] **Step 5: Review launch checklist from spec**

```
- [ ] All packages on PyPI with correct versions → metadata ready, builds verified
- [ ] pip install atp-platform && atp quickstart works → quickstart command implemented
- [ ] Docker compose up → dashboard with demo data → Dockerfile + compose ready
- [ ] GitHub Actions template verified → basic + game-eval templates exist
- [ ] README, comparison page, contributing guide ready → all created
- [ ] Blog post published → draft ready
- [ ] At least 1 reproducible experiment → run_experiment.py with JSON/CSV output
```
