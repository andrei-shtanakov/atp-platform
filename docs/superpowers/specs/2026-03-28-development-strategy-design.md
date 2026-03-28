# ATP Platform Development Strategy: Dual-Track Launch

**Date:** 2026-03-28
**Status:** Approved
**Timeline:** ~10 weeks (3 phases)
**Team:** 2-3 people

## Context

ATP (Agent Test Platform) is a framework-agnostic platform for testing and evaluating AI agents. The project has completed Phases 1-5 (decomposition, plugin ecosystem, operations). It consists of 5 workspace packages (atp-core, atp-adapters, atp-dashboard, game-environments, atp-games), 9+ adapters, 10+ evaluator types, ~250 tests, and comprehensive documentation.

### Goals

- **Public release and adoption** — publish to PyPI, polish DX, attract users
- **Research instrument** — game-theoretic evaluation of LLM agents, reproducible experiments

### Target Audience

Both AI researchers and AI engineers in product teams, equally.

### Competitive Advantage

The combination of protocol-agnostic testing (single protocol for any agent framework) and game-theoretic evaluation is unique. No existing tool (DeepEval, Promptfoo, Inspect AI) offers both.

### Constraints

- Protocol stability needs audit before public commitment
- Team of 2-3 people — must parallelize effectively
- 29 pre-existing pyrefly type errors to resolve

---

## Strategy: Dual-Track with 3 Phases

Two parallel work tracks converging in a unified launch.

| Phase | Duration | Track 1: Platform | Track 2: Research |
|-------|----------|-------------------|-------------------|
| **Phase 0: Stabilize** | 3 weeks | Protocol audit, pyrefly fixes, API surface audit | Game theory audit, edge cases, gap identification |
| **Phase 1: Publish** | 4 weeks | PyPI publishing, CLI/DX polish, quickstart, Docker | New games, multi-round tournaments, LLM benchmarks |
| **Phase 2: Launch** | 3 weeks | CI/CD templates, Docker images, docs, licensing | Visualizations, reproducible experiments, blog post |

**Sync point:** End of Phase 1 — integration test with PyPI-published packages + tournament via CLI + dashboard.

---

## Phase 0: Stabilize (3 weeks)

### Track 1: Protocol Audit & Stabilization

- **ATP Protocol audit:** review ATPRequest/ATPResponse/ATPEvent for completeness and consistency. Check: are all fields necessary, no duplication, correct defaults.
- **Semver contract:** freeze Protocol v1.0. Document what constitutes a breaking change. Add protocol version field if absent.
- **Pyrefly cleanup:** fix 29 existing type errors, configure CI to block new ones.
- **API surface audit:** review public API of each package (atp-core, atp-adapters, atp-dashboard). Ensure `__all__` exports are correct, internal modules marked `_private`.
- **Deprecation sweep:** find and remove dead code, unused features, TODO stubs.

### Track 2: Game Theory Audit

- **Game coverage:** audit 5 existing games and 19 strategies — verify payoff matrices, Nash equilibria correctness.
- **Edge cases:** review hypothesis property-based test coverage for boundary conditions (zero rounds, identical strategies, extreme payoffs).
- **LLM agent stability:** verify GPT-4o-mini agent reliability in Prisoner's Dilemma — retry logic, timeout handling, cost tracking.
- **Gap identification:** which classic games are missing and needed for a compelling research story.

### Deliverables

- "ATP v1.0 Protocol Specification" document in `docs/`
- "Game Theory Coverage Report" document in `docs/`

---

## Phase 1: Publish (4 weeks)

### Track 1: PyPI & Developer Experience

**Publishing order:**
1. `game-environments` (standalone, no ATP dependency)
2. `atp-games` (depends on game-environments + pydantic)
3. `atp-core`, `atp-adapters`, `atp-dashboard` (workspace members)
4. `atp-platform` (meta-package — update dependencies to PyPI versions of sub-packages)

**CLI/DX polish:**
- `atp init` — generates a working project in 1 command (test suite + config + example agent)
- `atp quickstart` — interactive wizard: choose adapter, generate test suite, first run
- Improve error messages — clear hints instead of tracebacks
- Consistent `--verbose` / `--quiet` flags across all commands

**Quickstart guide:** "From zero to first test in 5 minutes" — install via pip, run demo agent, view results.

**Docker:** official `Dockerfile` + `docker-compose.yml` with dashboard + demo agent for instant start.

### Track 2: Game Theory Expansion

**New games (2-3):**
- **Public Goods Game** — tests group cooperation (>2 agents)
- **Battle of the Sexes** — tests coordination under preference conflict
- **Stag Hunt** — tests trust vs safety (complements Prisoner's Dilemma)

**Multi-round tournaments:**
- Round-robin between N agents x M strategies
- Elo rating of agents based on tournament results
- Persistence: tournament history saved to dashboard DB

**LLM Agent Benchmarks:**
- Standard set: 3-5 models (GPT-4o-mini, Claude Haiku, Gemini Flash, Llama) x 5 games x 100 rounds
- Metrics: cooperation rate, adaptation speed, consistency, cost per game
- Reproducibility: seed-based randomization, fixed prompts

### Sync Point

Integration test at end of Phase 1: install all packages from PyPI, run tournament via CLI, verify results appear in dashboard.

---

## Phase 2: Launch (3 weeks)

### Track 1: CI/CD & Production Readiness

**GitHub Actions templates:**
- `atp-test.yml` — run test suite in CI, baseline regression check
- `atp-game-eval.yml` — game-theoretic evaluation as CI step
- Ready-to-copy examples for any repository

**Docker images:**
- `ghcr.io/*/atp-runner` — minimal image for CI
- `ghcr.io/*/atp-dashboard` — dashboard with preloaded demo data
- Multi-arch (amd64 + arm64)

**Documentation for launch:**
- Rewrite README: clear value proposition — "ATP = framework-agnostic agent testing + game-theoretic evaluation"
- Comparison page: ATP vs DeepEval vs Promptfoo vs Inspect AI — honest table, emphasis on uniqueness
- Contributing guide: how to add an adapter, evaluator, game

**Licensing:** ensure LICENSE file exists, choose license (MIT/Apache 2.0), verify dependency compatibility.

### Track 2: Research Content & Visualizations

**Visualizations:**
- Interactive tournament charts (cooperation heatmap, Elo progression, strategy distribution)
- Export to PNG/SVG for publications
- Dashboard integration: new "Game Analytics" page

**Reproducible experiments:**
- `examples/experiments/` — full pipeline: config, run, results, visualization
- `atp experiment run examples/experiments/llm_prisoners_dilemma.yaml` — one command
- Results in machine-readable format (JSON + CSV)

**Blog post / article:**
- "How Cooperative is Your AI? Game-Theoretic Evaluation of LLM Agents"
- Content: motivation, methodology, benchmark results, how to reproduce
- Target platforms: dev.to, Medium, arxiv (if deep enough)

### Launch Checklist

- [ ] All packages on PyPI with correct versions
- [ ] `pip install atp-platform` then `atp quickstart` works
- [ ] Docker compose up shows dashboard with demo data
- [ ] GitHub Actions template verified on real repo
- [ ] README, comparison page, contributing guide ready
- [ ] Blog post published
- [ ] At least 1 reproducible experiment with results

---

## Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Protocol breaking changes after publication | Medium | High | Phase 0 audit + protocol versioning. Fix issues BEFORE PyPI publication |
| LLM API costs during benchmarks | High | Medium | Use cheap models (Haiku, Flash, GPT-4o-mini). Budget caps via atp cost. Cache results for reruns |
| Focus dilution between tracks | Medium | Medium | Weekly sync. Clear ownership split. Phase 1 sync point is mandatory |
| PyPI dependency hell | Low | High | Test installation in clean venv at each publishing stage. Minimize transitive dependencies |
| Low adoption after launch | Medium | High | Blog post + Hacker News, Reddit r/MachineLearning. Examples for popular frameworks (LangGraph, CrewAI) in quickstart |

---

## Success Metrics

### At launch (~10 weeks):

- **Platform:** all 5 packages on PyPI, `pip install atp-platform && atp quickstart` works in < 5 minutes
- **Research:** at least 3 models tested across 5 games, results published and reproducible
- **Documentation:** quickstart, comparison page, contributing guide, 1 blog post
- **Quality:** 0 pyrefly errors, test coverage >= 80%, CI green

### 3 months post-launch:

- PyPI downloads > 500/month
- GitHub stars > 100
- At least 2 external contributors
- At least 1 external publication/blog post about ATP
