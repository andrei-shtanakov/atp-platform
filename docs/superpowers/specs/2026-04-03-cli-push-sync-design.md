# CLI Push, Pull & Sync

**Date:** 2026-04-03
**Status:** Approved

## Overview

Three new CLI commands for managing test suites between local filesystem and remote ATP server:
- `atp push` — upload YAML files to server
- `atp pull` — download suites from server to local YAML files
- `atp sync` — synchronize a directory (push changed/new files)

## Commands

### `atp push`

Upload one or more YAML test suite files to the remote server.

```bash
atp push suite.yaml --server https://atp.pr0sto.space
atp push tests/*.yaml --server https://atp.pr0sto.space
atp push suite.yaml --api-key sk-abc123
```

- Uses `POST /api/suite-definitions/upload` endpoint (already implemented)
- Shows validation result per file (errors, warnings) in terminal
- Exit code 0 on success, 1 if any file failed validation
- Skips files that already exist on server (409) with a warning, unless `--force` re-uploads
- Updates `.atp-sync.json` manifest if it exists in the directory

### `atp pull`

Download suite definitions from server to local YAML files.

```bash
atp pull --server https://atp.pr0sto.space --dir ./tests/
atp pull --id 5 --server https://atp.pr0sto.space
atp pull --all --server https://atp.pr0sto.space --dir ./tests/
```

- `--dir` — output directory (default: current directory)
- `--id N` — pull a specific suite by ID
- `--all` — pull all suites from server (default behavior when no --id)
- Uses `GET /api/suite-definitions` to list suites, then `GET /api/suite-definitions/{id}/yaml` to export each
- Saves as `{suite_name}.yaml` (sanitized filename — replace spaces/special chars with hyphens)
- Overwrites existing files with `--force`, otherwise skips with warning
- Updates `.atp-sync.json` manifest

### `atp sync`

Synchronize a local directory with the remote server. Push only changed/new files.

```bash
atp sync ./tests/ --server https://atp.pr0sto.space
atp sync . --server https://atp.pr0sto.space
```

- Scans directory recursively for `.yaml`/`.yml` files
- Compares SHA256 hashes with local manifest `.atp-sync.json`
- **New files** (not in manifest) → push to server
- **Changed files** (hash differs) → push to server (creates new version or errors on duplicate name — user must delete old one first)
- **Unchanged files** → skip
- **Deleted files** (in manifest but not on disk) → warning in terminal, does NOT delete from server
- Creates/updates `.atp-sync.json` after successful sync

## Authentication

Resolution order:
1. `--api-key KEY` flag
2. `ATP_API_KEY` environment variable
3. Token from `~/.atp/config.json` (from Device Flow login)

All three commands use the same auth resolution.

## Server URL

Resolution order:
1. `--server URL` flag
2. `ATP_SERVER` environment variable
3. `server` field from `.atp-sync.json` (if exists in current directory)

## Manifest `.atp-sync.json`

Created/updated by push, pull, and sync commands. Stored in the synced directory.

```json
{
  "server": "https://atp.pr0sto.space",
  "last_sync": "2026-04-03T12:00:00Z",
  "files": {
    "suite.yaml": {
      "sha256": "abc123def456...",
      "suite_id": 1,
      "synced_at": "2026-04-03T12:00:00Z"
    },
    "reasoning.yaml": {
      "sha256": "789ghi012jkl...",
      "suite_id": 2,
      "synced_at": "2026-04-03T12:00:00Z"
    }
  }
}
```

## Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `atp/cli/push.py` | `atp push` command implementation |
| Create | `atp/cli/pull.py` | `atp pull` command implementation |
| Create | `atp/cli/sync_cmd.py` | `atp sync` command implementation |
| Create | `atp/cli/remote.py` | Shared: auth resolution, server URL, HTTP client, manifest I/O |
| Modify | `atp/cli/main.py` | Register push, pull, sync commands |
| Create | `tests/unit/cli/test_push.py` | Tests for push command |
| Create | `tests/unit/cli/test_pull.py` | Tests for pull command |
| Create | `tests/unit/cli/test_sync.py` | Tests for sync command |
| Create | `tests/unit/cli/test_remote.py` | Tests for shared utilities |

## Output Format

Terminal output uses simple text with status indicators:

```
$ atp push tests/*.yaml --server https://atp.pr0sto.space
Pushing 3 files to https://atp.pr0sto.space...
  ✓ suite.yaml → created (id=1)
  ✓ reasoning.yaml → created (id=2, 1 warning)
  ✗ broken.yaml → 2 validation errors
    - test-1: unknown assertion type 'foo'
    - YAML parse error at line 15

2 succeeded, 1 failed
```

```
$ atp pull --all --server https://atp.pr0sto.space --dir ./tests/
Pulling suites from https://atp.pr0sto.space...
  ✓ suite.yaml (id=1)
  ✓ reasoning.yaml (id=2)
  - skipped advanced.yaml (already exists, use --force)

2 pulled, 1 skipped
```

```
$ atp sync ./tests/ --server https://atp.pr0sto.space
Syncing ./tests/ with https://atp.pr0sto.space...
  ✓ suite.yaml → unchanged, skipped
  ✓ reasoning.yaml → updated (id=2)
  ✓ new-suite.yaml → created (id=3)
  ⚠ old-suite.yaml → removed locally, still on server

1 created, 1 updated, 1 unchanged, 1 removed locally
```

## Scope

- Push uses the existing upload endpoint — no new server-side code needed
- Pull uses existing suite-definitions YAML export endpoint
- No server-side delete (sync only pushes, never deletes remotely)
- No conflict resolution — if server has a different version, push fails with 409
- `.atp-sync.json` should be committed to git (tracks sync state for the team)
- No watch mode / continuous sync (future feature)
