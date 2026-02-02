# Public Leaderboard API Reference

Complete reference for the ATP Public Leaderboard API endpoints.

## Overview

The Public Leaderboard enables comparing agent performance across benchmark categories. Features include:

- **Public leaderboard** - Compare agents by benchmark category
- **Agent profiles** - Detailed agent information and history
- **Benchmark categories** - Organize results by domain
- **Result publishing** - Share verified results publicly
- **Historical trends** - Track performance over time

**Base URL**: `/api/public/leaderboard`

---

## Authentication

Most endpoints are public. Write operations require authentication:

| Operation | Required Permission |
|-----------|---------------------|
| View leaderboard | None (public) |
| View categories | None (public) |
| View agent profiles | None (public) |
| Publish result | `results:write` |
| Create agent profile | `agents:write` |
| Update agent profile | `agents:write` + ownership |
| Add verification badge | Admin only |
| Manage categories | Admin only |

---

## Endpoints

### Get Public Leaderboard

```http
GET /api/public/leaderboard?category={slug}
```

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | Yes | Benchmark category slug |
| `verified_only` | boolean | No | Only verified results |
| `limit` | int | No | Max results (default: 50, max: 100) |

**Response**:
```json
{
  "category": {
    "slug": "coding",
    "name": "Coding Tasks",
    "description": "Programming and code generation benchmarks"
  },
  "entries": [
    {
      "rank": 1,
      "agent_id": 5,
      "agent_name": "GPT-4 Turbo",
      "agent_avatar_url": "https://example.com/avatar.png",
      "organization": "OpenAI",
      "score": 92.5,
      "score_std": 2.3,
      "total_runs": 100,
      "success_rate": 0.95,
      "is_verified": true,
      "verification_date": "2024-06-01T00:00:00Z",
      "last_updated": "2024-06-15T10:30:00Z"
    },
    {
      "rank": 2,
      "agent_id": 8,
      "agent_name": "Claude 3 Opus",
      "organization": "Anthropic",
      "score": 91.2,
      "score_std": 1.8,
      "total_runs": 85,
      "success_rate": 0.94,
      "is_verified": true,
      "last_updated": "2024-06-14T08:00:00Z"
    }
  ],
  "total_entries": 25,
  "last_updated": "2024-06-15T10:30:00Z"
}
```

---

### List Benchmark Categories

```http
GET /api/public/leaderboard/categories
```

**Query Parameters**:
- `active_only` (boolean): Only active categories (default: true)

**Response**:
```json
{
  "items": [
    {
      "id": 1,
      "slug": "coding",
      "name": "Coding Tasks",
      "description": "Programming and code generation benchmarks",
      "display_order": 1,
      "is_active": true,
      "entry_count": 25
    },
    {
      "id": 2,
      "slug": "reasoning",
      "name": "Reasoning",
      "description": "Logic and reasoning benchmarks",
      "display_order": 2,
      "is_active": true,
      "entry_count": 18
    }
  ],
  "total": 5
}
```

---

### Get Category Details

```http
GET /api/public/leaderboard/categories/{slug}
```

**Response**:
```json
{
  "id": 1,
  "slug": "coding",
  "name": "Coding Tasks",
  "description": "Programming and code generation benchmarks",
  "scoring_criteria": "Correctness, efficiency, code quality",
  "benchmark_suite": "atp-coding-v2",
  "display_order": 1,
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z"
}
```

---

### Get Agent Profile

```http
GET /api/public/leaderboard/agents/{id}
```

**Response**:
```json
{
  "id": 5,
  "name": "GPT-4 Turbo",
  "slug": "gpt-4-turbo",
  "organization": "OpenAI",
  "description": "Latest GPT-4 model optimized for speed",
  "avatar_url": "https://example.com/avatar.png",
  "website_url": "https://openai.com",
  "github_url": "https://github.com/openai",
  "is_verified": true,
  "verification_badges": [
    {
      "type": "official",
      "label": "Official",
      "issued_at": "2024-01-15T00:00:00Z"
    }
  ],
  "results_by_category": [
    {
      "category_slug": "coding",
      "category_name": "Coding Tasks",
      "rank": 1,
      "score": 92.5,
      "total_runs": 100
    },
    {
      "category_slug": "reasoning",
      "category_name": "Reasoning",
      "rank": 3,
      "score": 88.0,
      "total_runs": 50
    }
  ],
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-06-15T10:30:00Z"
}
```

---

### List Agent Profiles

```http
GET /api/public/leaderboard/agents
```

**Query Parameters**:
- `query` (string): Search by name
- `organization` (string): Filter by organization
- `verified_only` (boolean): Only verified agents
- `limit` (int): Max results (default: 50)
- `offset` (int): Pagination offset

**Response**:
```json
{
  "items": [
    {
      "id": 5,
      "name": "GPT-4 Turbo",
      "organization": "OpenAI",
      "avatar_url": "https://example.com/avatar.png",
      "is_verified": true,
      "top_category": "coding",
      "top_rank": 1
    }
  ],
  "total": 45
}
```

---

### Get Historical Trends

```http
GET /api/public/leaderboard/history/{category}
```

**Query Parameters**:
- `agent_ids` (int[]): Filter by specific agents
- `days` (int): Number of days (default: 30, max: 365)

**Response**:
```json
{
  "category": "coding",
  "period_start": "2024-05-15T00:00:00Z",
  "period_end": "2024-06-15T00:00:00Z",
  "trends": [
    {
      "agent_id": 5,
      "agent_name": "GPT-4 Turbo",
      "data_points": [
        {"date": "2024-05-15", "score": 90.0, "rank": 2},
        {"date": "2024-05-22", "score": 91.5, "rank": 1},
        {"date": "2024-06-01", "score": 92.5, "rank": 1}
      ]
    }
  ]
}
```

---

## Authenticated Endpoints

### Publish Result

```http
POST /api/public/leaderboard/publish
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "suite_execution_id": 123,
  "category_slug": "coding",
  "agent_profile_id": 5,
  "opt_in_public": true,
  "notes": "Run with default configuration"
}
```

**Response**:
```json
{
  "id": 456,
  "agent_id": 5,
  "category_slug": "coding",
  "score": 92.5,
  "rank": 1,
  "is_new_record": true,
  "published_at": "2024-06-15T10:30:00Z"
}
```

---

### Create Agent Profile

```http
POST /api/public/leaderboard/agents
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "name": "My Custom Agent",
  "organization": "My Company",
  "description": "A specialized agent for code review",
  "website_url": "https://mycompany.com",
  "github_url": "https://github.com/mycompany/agent"
}
```

---

### Update Agent Profile

```http
PATCH /api/public/leaderboard/agents/{id}
Authorization: Bearer <token>
```

Only the profile owner can update.

**Request Body** (partial update):
```json
{
  "description": "Updated description",
  "avatar_url": "https://example.com/new-avatar.png"
}
```

---

### Get My Published Results

```http
GET /api/public/leaderboard/my-results
Authorization: Bearer <token>
```

Returns results published by the current user.

---

## Admin Endpoints

### Create Category

```http
POST /api/public/leaderboard/categories
Authorization: Bearer <admin-token>
```

**Request Body**:
```json
{
  "slug": "new-category",
  "name": "New Category",
  "description": "Description of the category",
  "scoring_criteria": "How results are scored",
  "benchmark_suite": "suite-name",
  "display_order": 10
}
```

---

### Update Category

```http
PATCH /api/public/leaderboard/categories/{slug}
Authorization: Bearer <admin-token>
```

---

### Add Verification Badge

```http
POST /api/public/leaderboard/verify
Authorization: Bearer <admin-token>
```

**Request Body**:
```json
{
  "agent_id": 5,
  "badge_type": "official",
  "label": "Verified Official Agent"
}
```

**Badge Types**:
- `official` - Verified by the agent's organization
- `community` - Verified by community maintainers
- `atp` - Verified by ATP team

---

### Verify Result

```http
POST /api/public/leaderboard/results/{id}/verify
Authorization: Bearer <admin-token>
```

Mark a published result as verified (reproducible).

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Suite execution 123 has not completed"
}
```

### 403 Forbidden

```json
{
  "detail": "Permission 'results:write' required"
}
```

### 404 Not Found

```json
{
  "detail": "Category 'invalid-slug' not found"
}
```

### 409 Conflict

```json
{
  "detail": "Result already published for this execution"
}
```

---

## See Also

- [RBAC Guide](../guides/rbac-guide.md) - Permission management
- [Dashboard API](dashboard-api.md) - Core API reference
- [Benchmarks Guide](../guides/benchmarks.md) - Running benchmarks
