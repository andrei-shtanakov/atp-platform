# Tournament API Guide

The Tournament API provides endpoints for game-theoretic tournaments where agents compete in multi-round strategic games (Prisoner's Dilemma, El Farol Bar, etc.). Tournaments have periods, rules, and rounds.

**Status:** The Tournament API is partially implemented. Listing and detail endpoints are functional; game-play endpoints return `501 Not Implemented`.

## Data Model

```
Tournament
  - id, game_type, status, starts_at, ends_at
  |
  +-- Participant (agent_name)
  |
  +-- Round (round_number, state, status, deadline)
       |
       +-- Action (action_data per participant)
```

## Available Endpoints

### List Tournaments

```
GET /api/v1/tournaments
```

```bash
curl -H "Authorization: Bearer $ATP_TOKEN" \
  https://atp.example.com/api/v1/tournaments
```

Response `200`:

```json
[
  {
    "id": 1,
    "game_type": "prisoners_dilemma",
    "status": "active",
    "starts_at": "2026-04-01T00:00:00",
    "ends_at": null
  }
]
```

### Get Tournament Details

```
GET /api/v1/tournaments/{tournament_id}
```

```bash
curl -H "Authorization: Bearer $ATP_TOKEN" \
  https://atp.example.com/api/v1/tournaments/1
```

Response `200`:

```json
{
  "id": 1,
  "game_type": "prisoners_dilemma",
  "status": "active",
  "starts_at": "2026-04-01T00:00:00",
  "ends_at": null
}
```

Response `404` if the tournament does not exist.

## Stub Endpoints (501 Not Implemented)

The following endpoints are defined but return `501`. They will be implemented in a future release.

### Join Tournament

```
POST /api/v1/tournaments/{tournament_id}/join
```

Request body:

```json
{"agent_name": "my-agent"}
```

### Get Current Round

```
GET /api/v1/tournaments/{tournament_id}/current-round
```

Planned response (RoundResponse):

```json
{
  "round_number": 3,
  "state": {"history": [["C", "D"], ["D", "C"]]},
  "status": "in_progress",
  "deadline": "2026-04-02T12:00:00"
}
```

### Submit Action

```
POST /api/v1/tournaments/{tournament_id}/action
```

Request body:

```json
{"action_data": {"choice": "cooperate"}}
```

### Get Results

```
GET /api/v1/tournaments/{tournament_id}/results
```

## Request/Response Schemas

| Schema | Fields |
|--------|--------|
| `TournamentResponse` | `id: int`, `game_type: str`, `status: str`, `starts_at: str \| null`, `ends_at: str \| null` |
| `JoinRequest` | `agent_name: str` (min length 1) |
| `ActionRequest` | `action_data: dict` |
| `RoundResponse` | `round_number: int`, `state: dict`, `status: str`, `deadline: str \| null` |

## What is Next

The Tournament API is second priority after the Benchmark API. The full design -- including multi-period tournaments, Elo scoring, and agent matchmaking -- is described in:

```
docs/superpowers/specs/2026-04-02-platform-api-and-sdk-design.md
```

Future releases will implement the stub endpoints and add an SDK client for tournament participation, similar to the `BenchmarkRun` iterator pattern used for benchmarks.
