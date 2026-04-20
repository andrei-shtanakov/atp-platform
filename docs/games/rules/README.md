# Game rules

Human-readable rules for the games shipped in `game-environments/`. Each file is derived from the Python implementation and intended for humans (agent operators, players, reviewers) — not as a machine-readable spec.

## Available rule sets

| Game registry id | Rules |
|---|---|
| `el_farol` | [English](el-farol-bar.en.md) · [Русский](el-farol-bar.ru.md) |

## Coverage plan

The game registry (`game-environments/game_envs/games/registry.py`) currently exposes 8 games. Rule documents still pending:

- [ ] `prisoners_dilemma`
- [ ] `auction`
- [ ] `stag_hunt`
- [ ] `battle_of_sexes`
- [ ] `congestion`
- [ ] `colonel_blotto`
- [ ] `public_goods`

## Conventions

- One file per game per language, named `<game-id>.<lang>.md` (e.g. `el-farol-bar.en.md`, `el-farol-bar.ru.md`).
- Keep in sync with the implementation: anything describing default values, action space, or payoff formulas must match the code. The source of truth is the `Game` subclass and its `GameConfig`.
- Prefer linking to the exact file (e.g. `game-environments/game_envs/games/el_farol.py`) rather than duplicating large code blocks.
