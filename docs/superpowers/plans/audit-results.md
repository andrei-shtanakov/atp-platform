# v2.0.0 Audit Results

**Audit run on:** 2026-05-07
**HEAD commit:** 5abb2e1f28817a2da4af7b4e6be7a2474d07af71
**Base tag:** v1.0.0 (commit 5ac531843f231cb79b8d60909e3639878eeb0338, 2026-02-13)

## Ancestor check

- PR #105 (El Farol intervals) — commit `6d480b27`: OK
  - Note: the spec listed `f6c14a6` for PR #105, but that SHA is **not** a valid
    git object on this checkout. The squash-merge for PR #105 on this branch is
    `6d480b27 El Farol: drop legacy flat-slot move format (#105)` (2026-04-28).
    Confirmed via `git log --grep='#105'`. The CHANGELOG / migration guide must
    cite `6d480b27`, not `f6c14a6`.
- PR #121 (El Farol scoring) — commit `89f000e4`: OK
- MCP gating (LABS-TSA PR-3) — commit `d0f11e26`: OK

## Breaking-change discovery

Total commits in v1.0.0..HEAD: **612**

Commits matching `BREAKING|breaking|breaking-change` (case-insensitive): 9 hits,
all of which are spec / plan / docs commits or commit bodies that mention "not
breaking" — none are actual breaking-change announcements distinct from the
items below.

Area queries inspected:
- `packages/atp-sdk/**` — 17 commits.
- `atp/dashboard/v2/routes/**` (incl. `packages/atp-dashboard/.../v2/routes/**`) — ~95 commits.
- `atp/dashboard/mcp/**` — 25 commits.
- `atp/dashboard/tournament/**` — ~70 commits.
- `game-environments/game_envs/games/**` — 20 commits.
- `atp/protocol/**` — 3 commits.

### Confirmed breaking changes

1. **El Farol action format** — PR #105, commit `6d480b27` (2026-04-28).
   Public-contract impact: the wire shape for El Farol moves changed from flat
   `{"slots": [...]}` to interval-based `{"intervals": [[start, end], ...]}`.
   Old format is no longer accepted; `sanitize` coerces invalid input to a
   safe action. External consumers via the participant-kit-el-farol-en wire
   contract are affected.
2. **El Farol default scoring** — PR #121, commit `89f000e4` (2026-05-02).
   Public-contract impact: `ElFarolConfig.scoring_mode` default flipped from
   `happy_minus_crowded` (per-day `happy − crowded`; final
   `t_happy / max(t_crowded, 0.1)`) to `happy_only` (per-day count of happy
   slots; final `t_happy`, no crowded penalty). Tournaments inherit the new
   default through the `_el_farol_for()` factory; legacy mode remains opt-in
   via `ElFarolConfig(scoring_mode=...)` for tests / atp-games standalone
   only and is **not exposed via the tournament API**.
3. **MCP `purpose` claim gating** — LABS-TSA PR-3, commit `d0f11e26`
   (2026-04-23). Public-contract impact:
   `JWTUserStateMiddleware` + `MCPAuthMiddleware` now reject any request to
   `/mcp` whose `agent_purpose` is NULL (user-level / admin / legacy token)
   or not equal to `"tournament"`. Same gating on `/api/v1/benchmarks/*` for
   tournament-purpose tokens. Existing API tokens get an
   `agent_purpose` snapshot column on `APIToken` populated from
   `Agent.purpose` at issuance; legacy tokens fall back to a lazy join cached
   in-process by `token_hash`.
4. **`POST /api/agents` → `410 Gone`** — PR #53 / LABS-54 Phase 2, commit
   `e1ab0436` (2026-04-18). Public-contract impact: the legacy ownerless
   agent-creation endpoint that worked at v1.0.0 now returns HTTP 410 with
   `Deprecation` + `Sunset` + `Link: ...; rel="successor-version"` headers
   pointing at `POST /api/v1/agents`. Stale clients fail loudly. The
   replacement endpoint resolves ownership from the caller's JWT and enforces
   per-user, per-purpose quotas. v1.0.0 shipped this endpoint as a working
   `201 Created` route, so this is a real backward-incompatible change for
   v1.0.0..HEAD and must be documented.

### Inspected but NOT breaking (audit log)

- **SDK rewrite to async-first (commits `4ee148ee` → `4c54cf4f`):** the
  atp-sdk package did not exist at v1.0.0 — first scaffold commit was
  `4ee148ee` (2026-04-02). The async-first rewrite, batch API, retry, sync
  wrapper, etc. are all internal evolution of a feature added between
  v1.0.0 and v2.0.0. Per spec §3, sub-packages have **independent versions
  and CHANGELOGs** — `atp-platform-sdk` already documents its own 0.1.0 →
  2.0.0 break in `packages/atp-sdk/CHANGELOG.md`. Root v2.0.0 lists the
  SDK only as a non-breaking highlight.
- **Benchmark API (`b90a58fc` and the entire route surface):** post-v1.0.0
  feature; the routes did not exist at v1.0.0. The auth requirement /
  ownership / `Run.user_id NOT NULL` security fix (`e46a98b0`) is internal
  evolution within the v1→v2 window — no v1.0.0 client could have been
  calling these endpoints. Listed as non-breaking highlight.
- **MCP server itself (`19f3a4fe` and the rest of `atp/dashboard/mcp/`):**
  post-v1.0.0 feature. The whole `/mcp` surface, all 8 tools, and
  `MCPAuthMiddleware` are additions. Only the **`purpose`-gating** layer
  on top is treated as breaking (item 3 above) because clients that joined
  during the post-v1.0.0 / pre-d0f11e26 window need to add `purpose` now.
- **API token prefixes `atp_u_` / `atp_a_` (commits `3ef07e30`, `00db0d30`):**
  additive — `JWTUserStateMiddleware` branches on prefix. JWT bearer tokens
  still work. Listed as non-breaking highlight.
- **Tournament API (`atp/dashboard/tournament/**`):** post-v1.0.0 feature.
  The wire-shape adjustments (server-side `game_type` injection in
  `f2a4801f`, `RoundState` discriminated union in `7fd58422`,
  `pending_submission` / `your_turn` injection in `464a2d72`) are all
  refinements of a feature that did not exist at v1.0.0; the
  `to_dict()` method preserves the on-the-wire contract for MCP callers
  introduced in the same release window.
- **New tournament games (battle_of_sexes `ff4e0958`, stag_hunt
  `4b0f4efa`, public_goods `17b7de37`):** purely additive — register new
  `game_type` values via discriminated union; existing PD / El Farol
  tournaments unaffected.
- **Shrink-on-deadline (`7180ea2b`, PR #117):** opt-in via
  `min_participants` field on `ElFarolBar` / `PublicGoods` tournaments;
  default behavior unchanged for tournaments that don't set it.
- **Per-purpose agent quota (`64dcb724`, LABS-TSA PR-2):** changes the
  HTTP status from 409 to 429 on quota exhaustion, but the endpoint
  (`POST /api/v1/agents`) is itself post-v1.0.0. No v1.0.0 client.
- **Participant uniqueness shift `tournament_id+user_id` →
  `tournament_id+agent_id` (`0c2226f3`, LABS-TSA PR-6):** internal
  database-level migration to support multi-agent users; tournament API
  is post-v1.0.0.
- **Protocol version validation (`c7c1e5df`):** `PROTOCOL_VERSION = "1.0"`,
  `SUPPORTED_VERSIONS = {"1.0"}`. v1.0.0 messages either omit the field
  (uses default `"1.0"`) or send `"1.0"`. Effectively transparent;
  rejection only fires for unknown values that v1.0.0 clients never
  sent. Not breaking.
- **Rate limiting (`3e58fe8b` and follow-ups):** new HTTP middleware on a
  surface that's largely post-v1.0.0; all limits are tunable via
  `ATP_RATE_LIMIT_*`. The default `"60/minute"` is loose enough that
  no v1.0.0 caller pattern would have been broken; not surfaced as a
  breaking change in the existing project tracking.
- **Container-isolated code-exec evaluator (`atp/evaluators/container.py`):**
  additive evaluator; not enabled by default. Not breaking.
- **Auth state store consolidation (`e88db055`, `adc19855`):** internal
  refactor (DeviceFlowStore → AuthStateStore); no public-API surface
  change. Not breaking.

## Decision

Audit complete — found **1 additional breaking change** beyond the three known
items: `POST /api/agents` returns `410 Gone` (PR #53, commit `e1ab0436`,
LABS-54 Phase 2). It gets a bullet in CHANGELOG `[2.0.0] / Breaking Changes`
and a `docs/migrations/2026-04-legacy-agents-endpoint.md` guide.

Final v2.0.0 breaking-change list:
1. El Farol action format (PR #105, `6d480b27`).
2. El Farol scoring default (PR #121, `89f000e4`).
3. MCP `purpose` claim gating (LABS-TSA PR-3, `d0f11e26`).
4. `POST /api/agents` returns 410 Gone (PR #53 / LABS-54 Phase 2,
   `e1ab0436`).
