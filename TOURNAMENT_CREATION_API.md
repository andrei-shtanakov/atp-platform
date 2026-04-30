# Tournament Creation API & Flow

## Overview

The ATP platform provides both REST API and admin UI for creating tournaments. Tournaments support multiple game types (Prisoner's Dilemma, El Farol, Stag Hunt, Battle of the Sexes, Public Goods) with configurable parameters.

---

## 1. REST API Endpoint

### Endpoint Details

**POST** `/api/v1/tournaments`

**Authentication:** Bearer JWT token (via `JWTUserStateMiddleware`)

**Status Code:** 201 (Created)

**Location:** [packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py](packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py#L309)

---

## 2. Request Format

### Request Body Schema

```python
class CreateTournamentRequest(BaseModel):
    """Payload for creating a new tournament."""
    
    name: str                                  # Tournament display name
    game_type: str = "prisoners_dilemma"      # Game type (see supported below)
    num_players: int = Field(ge=2)            # Number of participants
    total_rounds: int = Field(ge=1)           # Total rounds to play
    round_deadline_s: int = Field(ge=1)       # Seconds per round deadline
    private: bool = False                     # Private (requires join_token) or public
    roster: list[BuiltinRosterEntry] = Field(default_factory=list)  # Optional builtin AI players
```

### Supported Game Types & Player Requirements

| Game Type | Min Players | Max Players | Notes |
|-----------|-------------|-------------|-------|
| `prisoners_dilemma` | 2 | 2 | Exactly 2 players |
| `stag_hunt` | 2 | 2 | Exactly 2 players |
| `battle_of_sexes` | 2 | 2 | Exactly 2 players |
| `el_farol` | 2 | 20 | N-player (2-20) |
| `public_goods` | 2 | 20 | N-player (2-20) |

### Builtin Roster (Optional)

Pre-populate tournament with deterministic AI strategies:

```python
class BuiltinRosterEntry(BaseModel):
    builtin_strategy: str  # Format: "game_type/strategy_name" (e.g., "prisoners_dilemma/tit_for_tat")
```

Examples:
- `"prisoners_dilemma/always_cooperate"`
- `"prisoners_dilemma/always_defect"`
- `"prisoners_dilemma/tit_for_tat"`
- `"el_farol/traditionalist"`

---

## 3. Response Format

### Success Response (201 Created)

```json
{
  "id": 42,
  "name": "Tournament Name",
  "status": "pending",
  "game_type": "prisoners_dilemma",
  "num_players": 2,
  "total_rounds": 100,
  "round_deadline_s": 30,
  "has_join_token": true,
  "cancelled_reason": null,
  "cancelled_reason_detail": null,
  "join_token": "generated_secret_token_string_here_only_returned_once"
}
```

**Key Points:**
- `join_token` is returned **only once** during creation
- For **private tournaments** only (when `private=true`)
- For **public tournaments**, `join_token` is `null`
- Store the token securely if needed for private tournaments

### Error Responses

#### 400 Bad Request - Roster Validation
```json
{
  "detail": "unknown builtin strategy 'prisoners_dilemma/nonexistent' for game prisoners_dilemma (namespace mismatch)"
}
```

#### 422 Unprocessable Entity - Validation Error
```json
{
  "detail": "max duration 100 rounds at 30s deadline; got 200 rounds. Reduce total_rounds or round_deadline_s."
}
```

#### 429 Too Many Requests - Private Tournament Limit Exceeded
```json
{
  "detail": "concurrent private tournament limit exceeded (3/3)"
}
```

#### 401 Unauthorized
```json
{
  "detail": "unauthenticated"
}
```

---

## 4. Key Constraints & Validations

### Duration Cap (AD-9)

The platform enforces a hard duration cap to prevent JWT token expiry mid-game:

```
max_duration_s = TOURNAMENT_PENDING_MAX_WAIT_S + (total_rounds × round_deadline_s)
max_allowed = (ATP_TOKEN_EXPIRE_MINUTES - 10) × 60
```

**Default calculation:**
- `ATP_TOKEN_EXPIRE_MINUTES = 60` (1 hour)
- `round_deadline_s = 30`
- **Max rounds: 100**

**Example error:**
```
max duration 100 rounds at 30s deadline; got 200 rounds. 
Reduce total_rounds or round_deadline_s.
```

### Private Tournament Requirements

When `private=true`:

1. **Creator commitment check**: Creator must have at least one tournament-purpose agent registered, OR roster must fill all slots
   ```
   RosterValidationError: private tournament needs 2 participants; you have 0 tournament agents 
   and 0 builtin(s), leaving 2 slot(s) unfilled. Register a tournament-purpose agent 
   via /ui/agents, or extend the builtin roster until len(roster) == num_players (2).
   ```

2. **Concurrent private tournament cap**: Per-user limit on active private tournaments (default: 3)
   ```
   ConcurrentPrivateCapExceededError: concurrent private tournament limit exceeded (3/3)
   ```

### Roster Validation

When providing builtin strategies:

1. Each entry must use format `"game_type/strategy_name"` (namespaced)
2. Game type in entry must match tournament's `game_type`
3. Roster size cannot exceed `num_players`
4. For private tournaments, if roster doesn't fill all slots, creator must have tournament agents

---

## 5. API Code Examples

### Python - Using `httpx` or `requests`

```python
import httpx

# Create tournament
response = httpx.post(
    "https://atp.example.com/api/v1/tournaments",
    headers={"Authorization": f"Bearer {jwt_token}"},
    json={
        "name": "My PD Tournament",
        "game_type": "prisoners_dilemma",
        "num_players": 2,
        "total_rounds": 100,
        "round_deadline_s": 30,
        "private": False,
    }
)

data = response.json()
tournament_id = data["id"]
join_token = data.get("join_token")  # Only for private tournaments

print(f"Tournament {tournament_id} created")
print(f"Status: {data['status']}")
print(f"Join token: {join_token}")
```

### Python - With Optional Builtin Players

```python
import httpx

response = httpx.post(
    "https://atp.example.com/api/v1/tournaments",
    headers={"Authorization": f"Bearer {jwt_token}"},
    json={
        "name": "PD vs. AI Tournament",
        "game_type": "prisoners_dilemma",
        "num_players": 2,
        "total_rounds": 50,
        "round_deadline_s": 30,
        "private": True,
        "roster": [
            {"builtin_strategy": "prisoners_dilemma/tit_for_tat"}
        ]
    }
)

data = response.json()
print(f"Tournament created with builtin AI: {data}")
```

### Python - Private Tournament with Creator Agent

```python
import httpx

response = httpx.post(
    "https://atp.example.com/api/v1/tournaments",
    headers={"Authorization": f"Bearer {jwt_token}"},
    json={
        "name": "Research Group Tournament",
        "game_type": "prisoners_dilemma",
        "num_players": 2,
        "total_rounds": 100,
        "round_deadline_s": 30,
        "private": True
    }
)

if response.status_code == 201:
    data = response.json()
    tournament_id = data["id"]
    join_token = data["join_token"]
    
    # Securely store/distribute join_token to authorized participants
    print(f"Private tournament {tournament_id} created")
    print(f"Share this token with participants: {join_token}")
else:
    print(f"Error: {response.text}")
```

### cURL Examples

#### Basic Tournament Creation

```bash
curl -X POST "https://atp.example.com/api/v1/tournaments" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "PD Tournament",
    "game_type": "prisoners_dilemma",
    "num_players": 2,
    "total_rounds": 100,
    "round_deadline_s": 30,
    "private": false
  }'
```

#### Private Tournament with Builtin AI

```bash
curl -X POST "https://atp.example.com/api/v1/tournaments" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AI Tournament",
    "game_type": "prisoners_dilemma",
    "num_players": 2,
    "total_rounds": 50,
    "round_deadline_s": 30,
    "private": true,
    "roster": [
      {"builtin_strategy": "prisoners_dilemma/tit_for_tat"}
    ]
  }'
```

#### El Farol Tournament (N-player)

```bash
curl -X POST "https://atp.example.com/api/v1/tournaments" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "El Farol Bar Problem",
    "game_type": "el_farol",
    "num_players": 10,
    "total_rounds": 50,
    "round_deadline_s": 30,
    "private": false
  }'
```

---

## 6. Admin UI for Tournament Creation

### UI Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/ui/admin/tournaments` | GET | List all tournaments |
| `/ui/admin/tournaments/new` | GET | Show creation form |
| `/ui/admin/tournaments/new` | POST | Submit creation form |
| `/ui/admin/tournaments/{id}` | GET | View tournament details |

### Form Fields

The admin UI (at `/ui/admin/tournaments/new`) accepts:

- **Name** (text) - Tournament display name
- **Game Type** (select) - Choose from supported games
- **Number of Players** (number) - Based on game type constraints
- **Total Rounds** (number) - Round count
- **Round Deadline (seconds)** (number) - Timeout per round

### Form Submission Code

Location: [packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py](packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py#L115)

```python
@router.post("/tournaments/new")
async def admin_tournament_new_submit(
    request: Request,
    session: DBSession,
    service: TournamentService = Depends(get_tournament_service),
    name: str = Form(...),
    game_type: str = Form(...),
    num_players: int = Form(...),
    total_rounds: int = Form(...),
    round_deadline_s: int = Form(...),
):
    """Create a tournament and redirect to its admin detail page."""
    user = await _require_admin_ui_user(request, session)
    try:
        tournament, _join_token = await service.create_tournament(
            user,
            name=name,
            game_type=game_type,
            num_players=num_players,
            total_rounds=total_rounds,
            round_deadline_s=round_deadline_s,
        )
    except TournamentValidationError as exc:
        return _templates(request).TemplateResponse(
            request=request,
            name="ui/admin/tournament_new.html",
            context={
                "user": user,
                "error": str(exc),
                "active_page": "admin",
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return RedirectResponse(
        url=f"/ui/admin/tournaments/{tournament.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
```

---

## 7. Tournament Creation Flow (Backend)

### Service Layer Implementation

Location: [packages/atp-dashboard/atp/dashboard/tournament/service.py](packages/atp-dashboard/atp/dashboard/tournament/service.py#L195)

Key steps:

1. **Validate game type** - Check against `_SUPPORTED_GAMES`
2. **Validate player count** - Enforce game-specific constraints
3. **Validate duration** - Check AD-9 cap
4. **Validate roster** (if provided)
   - Check namespacing (format: `game/name`)
   - Verify each strategy exists
   - Verify roster size ≤ num_players
5. **Private tournament checks** (if `private=true`)
   - Creator must have tournament agent or roster fills slots
   - Check concurrent-private cap per user
6. **Create Tournament record**
   - Generate join_token if private: `secrets.token_urlsafe(32)`
   - Set status to `PENDING`
   - Set pending_deadline to `now + TOURNAMENT_PENDING_MAX_WAIT_S`
7. **Insert builtin participants** (if roster provided)
   - Create `Participant` rows with `builtin_strategy` set
   - If roster fills all slots, auto-transition to `ACTIVE`
8. **Return** `(tournament, join_token_plaintext)`

### Pseudo-code Flow

```python
async def create_tournament(
    self,
    creator: User,
    *,
    name: str,
    game_type: str,
    num_players: int,
    total_rounds: int,
    round_deadline_s: int,
    private: bool = False,
    roster: list[str] | None = None,
) -> tuple[Tournament, str | None]:
    # 1. Validate inputs
    validate_game_type(game_type)
    validate_player_count(game_type, num_players)
    validate_duration_cap(total_rounds, round_deadline_s)
    
    # 2. Validate roster if provided
    if roster:
        validate_roster(roster, game_type, num_players)
    
    # 3. Private tournament checks
    if private:
        check_creator_has_agent_or_roster_full(creator, roster, num_players)
        check_concurrent_private_cap(creator)
    
    # 4. Create tournament
    join_token = None
    if private:
        join_token = secrets.token_urlsafe(32)
    
    tournament = Tournament(
        game_type=game_type,
        status=TournamentStatus.PENDING,
        num_players=num_players,
        total_rounds=total_rounds,
        round_deadline_s=round_deadline_s,
        created_by=creator.id,
        config={"name": name},
        pending_deadline=now + TOURNAMENT_PENDING_MAX_WAIT_S,
        join_token=join_token,
    )
    session.add(tournament)
    await session.flush()
    
    # 5. Insert builtin participants
    for strategy in roster:
        session.add(Participant(
            tournament_id=tournament.id,
            agent_name=strategy,
            builtin_strategy=strategy,
        ))
    
    # Auto-transition if roster fills all slots
    if len(roster) == num_players:
        await _start_tournament(tournament)
    
    return tournament, join_token
```

---

## 8. Tournament Status Lifecycle

After creation, tournaments follow this lifecycle:

```
PENDING → ACTIVE → COMPLETED/CANCELLED
```

### Status Transitions

- **PENDING**: Created, waiting for participants to join (max: `TOURNAMENT_PENDING_MAX_WAIT_S` seconds)
- **ACTIVE**: Game in progress, participants are playing rounds
- **COMPLETED**: All rounds finished
- **CANCELLED**: Tournament cancelled by admin/owner

### Auto-transitions

1. When `num_players` join → `PENDING` → `ACTIVE` (triggers round 1)
2. After last round completes → `ACTIVE` → `COMPLETED`
3. If pending expires → `PENDING` → `CANCELLED`

---

## 9. Related Endpoints

### List Tournaments

**GET** `/api/v1/tournaments?status_filter=pending`

```python
async def list_tournaments_endpoint(
    user: TournamentUser,
    service: TournamentSvc,
    status_filter: str | None = None,
) -> dict[str, Any]:
    """List tournaments visible to the calling user."""
    tournaments = await service.list_tournaments(user=user, status=filt)
    return {"tournaments": [...]}
```

### Get Tournament Details

**GET** `/api/v1/tournaments/{tournament_id}`

```python
async def get_tournament_endpoint(
    tournament_id: int,
    user: TournamentUser,
    service: TournamentSvc,
) -> dict[str, Any]:
    """Return tournament details (visibility-filtered)."""
    t = await service.get_tournament(tournament_id, user)
    return _serialize(t, user.is_admin)
```

### Cancel Tournament

**POST** `/api/v1/tournaments/{tournament_id}/cancel`

```python
@router.post("/{tournament_id}/cancel")
async def cancel_tournament_endpoint(
    tournament_id: int,
    user: TournamentUser,
    service: TournamentSvc,
) -> dict[str, Any]:
    """Cancel a tournament (admin or owner only)."""
    await service.cancel_tournament(user=user, tournament_id=tournament_id)
```

---

## 10. Configuration & Environment Variables

### Relevant Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATP_TOKEN_EXPIRE_MINUTES` | 60 | JWT expiry (affects duration cap) |
| `ATP_ADMIN_TOKEN_EXPIRE_MINUTES` | 720 (12h) | Admin JWT expiry |
| `ATP_TOURNAMENT_PENDING_MAX_WAIT_S` | 300 (5m) | Max wait for pending tournament |
| `ATP_MAX_CONCURRENT_PRIVATE_TOURNAMENTS_PER_USER` | 3 | Limit on active private tournaments |
| `ATP_TOURNAMENT_REASONING_MAX_CHARS` | 8000 | Max reasoning text length per action |

---

## 11. Error Handling Reference

### Common Error Scenarios

| Scenario | Status | Error |
|----------|--------|-------|
| Missing auth header | 401 | `"unauthenticated"` |
| Non-admin creating tournament | 403 | `"Admin access required"` (UI only) |
| Invalid game type | 422 | `"unsupported game_type 'foo'; supports: [...]"` |
| Wrong player count for game | 422 | `"prisoners_dilemma requires exactly 2 players, got 3"` |
| Tournament too long | 422 | `"max duration 100 rounds at 30s deadline; got 200 rounds..."` |
| Unknown builtin strategy | 400 | `"unknown builtin strategy 'pd/nonexistent'..."` |
| Creator has no tournament agent | 400 | `"private tournament needs 2 participants; you have 0..."` |
| Concurrent private cap exceeded | 429 | `"concurrent private tournament limit exceeded (3/3)"` |

---

## 12. Documentation References

- **Main spec**: [docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md](docs/superpowers/specs/2026-04-10-mcp-tournament-server-design.md)
- **Service implementation**: [packages/atp-dashboard/atp/dashboard/tournament/service.py](packages/atp-dashboard/atp/dashboard/tournament/service.py)
- **API routes**: [packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py](packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py)
- **Admin UI**: [packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py](packages/atp-dashboard/atp/dashboard/v2/routes/admin_ui.py)
- **Schemas**: [packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py](packages/atp-dashboard/atp/dashboard/v2/routes/tournament_api.py) (lines ~100-130)
- **Models**: [packages/atp-dashboard/atp/dashboard/tournament/models.py](packages/atp-dashboard/atp/dashboard/tournament/models.py)

---

## 13. Quick Start Example

### Complete End-to-End Example

```python
import httpx
import json

# Step 1: Create a tournament
response = httpx.post(
    "https://atp.example.com/api/v1/tournaments",
    headers={"Authorization": f"Bearer {jwt_token}"},
    json={
        "name": "Research Tournament",
        "game_type": "prisoners_dilemma",
        "num_players": 2,
        "total_rounds": 100,
        "round_deadline_s": 30,
        "private": False,
    }
)

assert response.status_code == 201, f"Failed: {response.text}"

tournament = response.json()
tournament_id = tournament["id"]
print(f"✓ Tournament {tournament_id} created")
print(f"  Status: {tournament['status']}")
print(f"  Game: {tournament['game_type']}")
print(f"  Rounds: {tournament['total_rounds']}")

# Step 2: List tournaments to verify
list_response = httpx.get(
    "https://atp.example.com/api/v1/tournaments",
    headers={"Authorization": f"Bearer {jwt_token}"},
)
tournaments = list_response.json()["tournaments"]
print(f"✓ Total tournaments: {len(tournaments)}")

# Step 3: Get tournament details
detail_response = httpx.get(
    f"https://atp.example.com/api/v1/tournaments/{tournament_id}",
    headers={"Authorization": f"Bearer {jwt_token}"},
)
details = detail_response.json()
print(f"✓ Tournament details: {json.dumps(details, indent=2)}")
```

---

## 14. Key Design Decisions (AD-1 through AD-10)

The tournament creation API reflects several key architectural decisions:

- **AD-9**: Hard duration cap prevents JWT expiry mid-game
- **AD-10**: Open-join with optional `join_token` for private tournaments
- Private tournaments require creator to have a tournament-purpose agent or provide AI roster
- Concurrent private tournament limit (default 3) prevents resource abuse
- Builtin AI players can pre-fill tournament slots for testing/research

