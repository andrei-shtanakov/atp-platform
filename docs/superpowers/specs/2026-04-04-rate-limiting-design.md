# Rate Limiting Design

## Summary

Add HTTP rate limiting to the ATP Dashboard API using slowapi (built on the `limits` library). Protects against brute-force attacks on auth endpoints and abuse of the benchmark API. Per-endpoint limits with user/IP-based keys.

## Context

- Production runs at https://atp.pr0sto.space behind nginx reverse proxy
- Single uvicorn worker currently (`--workers=1`)
- SDK already handles 429 with `Retry-After` header (retry.py:109-124)
- No existing rate limiting; only CORS middleware in factory.py
- Quota middleware exists (tenancy) but handles business-level limits, not HTTP rate limiting

## Design

### Dependencies

- `slowapi>=0.1.9` — FastAPI rate limiting, wraps `limits` library
- In-memory backend by default; `limits` supports Redis via `redis://` URI when needed

### Key Function

Rate limit keys resolve in this order:
1. If request has valid JWT → `user:{user_id}`
2. Else → `ip:{client_ip}`

**Proxy handling**: Use `X-Forwarded-For` header when behind reverse proxy. slowapi's `get_remote_address` handles this, but we'll configure `trusted_hosts` to prevent spoofing. Nginx already sets `X-Real-IP` / `X-Forwarded-For`.

### Limits

| Group | Endpoints | Limit | Key | Rationale |
|-------|-----------|-------|-----|-----------|
| Auth | `/api/auth/login`, `/api/auth/register`, `/api/auth/device*` | 5/minute | IP | Brute-force protection |
| SSO/SAML | `/api/sso/init`, `/api/sso/callback`, `/api/saml/init`, `/api/saml/acs` | 10/minute | IP | IdP retry tolerance |
| Benchmark API | `/api/v1/benchmarks/*`, `/api/v1/runs/*` | 120/minute | User | SDK polling: next_task + submit cycles |
| Suite upload | `POST /api/suite-definitions/upload` | 10/minute | User | Heavy validation |
| UI pages | `/ui/*` | 120/minute | IP | Browser navigation |
| Default | Everything else | 60/minute | IP | Reasonable fallback |

**Note on Benchmark API limit**: SDK `next_batch()` + `submit()` pattern does 2 requests per task. With 10-task benchmark and 5s per task, that's ~24 req/min. 120/min gives 5x headroom. Raised from initial 60/min based on SDK usage analysis.

### 429 Response

```
HTTP/1.1 429 Too Many Requests
Retry-After: 42
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 0
Content-Type: application/json

{
  "error": "rate_limit_exceeded",
  "detail": "Rate limit exceeded: 5 per 1 minute",
  "retry_after": 42
}
```

SDK retry.py already parses `Retry-After` header and backs off accordingly.

### Configuration

New fields in `DashboardConfig` (env vars):

```python
rate_limit_enabled: bool = True          # ATP_RATE_LIMIT_ENABLED
rate_limit_default: str = "60/minute"    # ATP_RATE_LIMIT_DEFAULT
rate_limit_auth: str = "5/minute"        # ATP_RATE_LIMIT_AUTH
rate_limit_api: str = "120/minute"       # ATP_RATE_LIMIT_API
rate_limit_upload: str = "10/minute"     # ATP_RATE_LIMIT_UPLOAD
rate_limit_storage: str = "memory://"    # ATP_RATE_LIMIT_STORAGE (redis:// for prod)
```

### Implementation

**New file**: `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py`
- `create_limiter(config)` — creates slowapi `Limiter` instance
- `get_rate_limit_key(request)` — extracts user_id from JWT or falls back to IP
- `rate_limit_exceeded_handler(request, exc)` — custom 429 JSON response

**Modified**: `packages/atp-dashboard/atp/dashboard/v2/factory.py`
- Add `SlowAPIMiddleware` after CORS
- Register `RateLimitExceeded` exception handler
- Pass limiter to app state

**Modified**: Route files — add `@limiter.limit()` decorators to endpoint groups

### Multi-Worker Note

In-memory backend (`memory://`) does not share state between workers. With `--workers=N`, effective limit is N times the configured value. For multi-worker production, switch to `redis://` via `ATP_RATE_LIMIT_STORAGE`. Current production runs single-worker, so this is acceptable for MVP.

### Testing

- **Unit**: key function returns user_id vs IP correctly, config parsing
- **Integration**: middleware returns 429 after limit exceeded, Retry-After header present, counter resets after window
- **Edge cases**: missing JWT, malformed X-Forwarded-For, disabled rate limiting

### Files to Create/Modify

| File | Action |
|------|--------|
| `packages/atp-dashboard/pyproject.toml` | Add `slowapi>=0.1.9` |
| `packages/atp-dashboard/atp/dashboard/v2/rate_limit.py` | New: limiter setup, key function, 429 handler |
| `packages/atp-dashboard/atp/dashboard/v2/config.py` | Add rate limit config fields |
| `packages/atp-dashboard/atp/dashboard/v2/factory.py` | Wire middleware + exception handler |
| `packages/atp-dashboard/atp/dashboard/v2/routes/auth.py` | Add limit decorators |
| `packages/atp-dashboard/atp/dashboard/v2/routes/device_auth.py` | Add limit decorators |
| `packages/atp-dashboard/atp/dashboard/v2/routes/sso.py` | Add limit decorators |
| `packages/atp-dashboard/atp/dashboard/v2/routes/saml.py` | Add limit decorators |
| `packages/atp-dashboard/atp/dashboard/v2/routes/benchmark_api.py` | Add limit decorators |
| `packages/atp-dashboard/atp/dashboard/v2/routes/upload.py` | Add limit decorators |
| `packages/atp-dashboard/atp/dashboard/v2/routes/ui.py` | Add limit decorators |
| `tests/unit/dashboard/test_rate_limit.py` | New: unit + integration tests |
