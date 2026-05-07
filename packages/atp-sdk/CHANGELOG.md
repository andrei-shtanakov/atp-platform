# Changelog — atp-platform-sdk

This package follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Routing rule: only changes that affect this package's public API land here.
Platform-level changes go in the root `CHANGELOG.md`.

## [Unreleased]

## [2.0.0] - YYYY-MM-DD

### Added

- `AsyncATPClient` and synchronous `ATPClient` wrapper.
- `BenchmarkRun` async/sync iteration with `next_batch(n)`.
- `emit()` / `emit_sync()` for streaming benchmark-run events.
- Sync convenience methods (`submit_sync`, `status_sync`, `cancel_sync`,
  `leaderboard_sync`, `next_batch_sync`).
- Exponential-backoff retry on transient HTTP errors.

### Changed

- See root `CHANGELOG.md` `[2.0.0]` for the MCP `/mcp` purpose gating
  contract. The SDK itself accepts whatever bearer token the caller
  supplies; the server decides eligibility based on the token's
  `agent_purpose` snapshot. No SDK-side knob to set.

### Migration

For users upgrading from 1.x, see the root `CHANGELOG.md` `[2.0.0]` entry
and the migration guides under `docs/migrations/`.
