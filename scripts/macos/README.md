# macOS launcher scripts

Double-clickable `.command` files for macOS Finder. They open Terminal.app, run one `uv run atp game ...` command, and leave the window open so the output is visible. Intended for demos and non-CLI users on a Mac.

## Files

| File | What it does |
|---|---|
| `00_setup_deps.command` | `uv sync --group dev` — install project dependencies |
| `10_run_el_farol.command` | Run the El Farol suite (`examples/test_suites/13_game_el_farol.yaml`) |
| `11_run_el_farol_verbose.command` | Same as above with `--episodes=20 -v` |
| `20_run_prisoners_dilemma.command` | Run Prisoner's Dilemma suite |
| `21_run_auction.command` | Run Auction suite |
| `30_run_all_games.command` | Run PD → Auction → El Farol in sequence |

Prefixes (`00`, `10`, `20`, `30`) are only used to sort the files in Finder.

## Usage

1. Install [uv](https://docs.astral.sh/uv/) on your Mac (`brew install uv` or the official installer).
2. Double-click `00_setup_deps.command` once to install dependencies.
3. Double-click any `NN_run_*.command` to run a game suite.

## Notes

- macOS only — Linux/Windows do not recognise `.command` files.
- On first launch Gatekeeper may block the script. Open via right-click → **Open**, or clear the quarantine attribute:
  ```
  xattr -d com.apple.quarantine scripts/macos/*.command
  ```
- Each script resolves its own path with `$SCRIPT_DIR` and then `cd`s two levels up to the repo root before calling `uv run`, so the scripts work regardless of what the current working directory is when double-clicked.
- If you move the repo, the scripts keep working as long as this directory remains `scripts/macos/` relative to the repo root.
