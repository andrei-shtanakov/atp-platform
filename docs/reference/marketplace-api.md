# Test Suite Marketplace API Reference

Complete reference for the ATP Test Suite Marketplace API endpoints.

## Overview

The Marketplace enables sharing, discovering, and installing test suites. It supports:

- **Publishing** test suites with versioning
- **Search and discovery** with filtering
- **Ratings and reviews** from the community
- **GitHub import** for easy suite publishing
- **Installation tracking** for usage analytics

**Base URL**: `/api/marketplace`

---

## Authentication

Most read endpoints are public. Write operations require authentication:

| Operation | Required Permission |
|-----------|---------------------|
| List/Search suites | None (public) |
| View suite details | None (public) |
| Publish suite | `marketplace:write` |
| Update/Delete suite | `marketplace:write` + ownership |
| Add review | `marketplace:write` |
| Feature/Verify suite | `marketplace:admin` |

---

## Endpoints

### List Marketplace Suites

```http
GET /api/marketplace
```

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | - | Search in name and description |
| `category` | string | - | Filter by category |
| `tags` | string[] | - | Filter by tags (any match) |
| `license_type` | string | - | Filter by license |
| `verified_only` | boolean | false | Only verified suites |
| `featured_only` | boolean | false | Only featured suites |
| `min_rating` | float | - | Minimum average rating (0-5) |
| `sort_by` | string | downloads | Sort field |
| `sort_order` | string | desc | asc or desc |
| `limit` | int | 20 | Max results (≤100) |
| `offset` | int | 0 | Pagination offset |

**Sort Options**: `downloads`, `rating`, `updated`, `name`, `created`

**Response**:
```json
{
  "items": [
    {
      "id": 1,
      "slug": "api-testing-suite",
      "name": "API Testing Suite",
      "short_description": "Comprehensive API testing",
      "category": "api",
      "tags": ["rest", "graphql"],
      "license_type": "MIT",
      "author_id": 5,
      "author_name": "johndoe",
      "is_verified": true,
      "is_featured": false,
      "download_count": 1250,
      "average_rating": 4.5,
      "review_count": 23,
      "latest_version": "2.1.0",
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-06-15T10:30:00Z"
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

---

### Get Suite Details

```http
GET /api/marketplace/{slug}
```

**Response**:
```json
{
  "id": 1,
  "slug": "api-testing-suite",
  "name": "API Testing Suite",
  "short_description": "Comprehensive API testing",
  "full_description": "# API Testing Suite\n\nThis suite provides...",
  "category": "api",
  "tags": ["rest", "graphql", "openapi"],
  "license_type": "MIT",
  "license_url": "https://opensource.org/licenses/MIT",
  "repository_url": "https://github.com/example/api-tests",
  "documentation_url": "https://example.com/docs",
  "author_id": 5,
  "author_name": "johndoe",
  "is_verified": true,
  "is_featured": false,
  "download_count": 1250,
  "install_count": 890,
  "average_rating": 4.5,
  "review_count": 23,
  "latest_version": "2.1.0",
  "versions": [
    {
      "version": "2.1.0",
      "release_notes": "Added GraphQL support",
      "created_at": "2024-06-15T10:30:00Z"
    }
  ],
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-06-15T10:30:00Z"
}
```

---

### Publish Suite

```http
POST /api/marketplace
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "name": "My Test Suite",
  "short_description": "A brief description",
  "full_description": "# My Test Suite\n\nDetailed markdown...",
  "category": "integration",
  "tags": ["database", "postgres"],
  "license_type": "Apache-2.0",
  "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
  "repository_url": "https://github.com/me/my-tests",
  "documentation_url": "https://my-tests.docs.com",
  "suite_definition": {
    "name": "My Test Suite",
    "tests": [...]
  },
  "initial_version": "1.0.0"
}
```

**Response**: `201 Created`
```json
{
  "id": 42,
  "slug": "my-test-suite",
  "name": "My Test Suite",
  ...
}
```

---

### Update Suite

```http
PATCH /api/marketplace/{slug}
Authorization: Bearer <token>
```

Only the suite owner can update. Updates the published suite metadata.

**Request Body** (partial update):
```json
{
  "short_description": "Updated description",
  "tags": ["database", "postgres", "mysql"]
}
```

---

### Delete (Unpublish) Suite

```http
DELETE /api/marketplace/{slug}
Authorization: Bearer <token>
```

Unpublishes the suite. Only owner can delete.

---

### List Versions

```http
GET /api/marketplace/{slug}/versions
```

**Response**:
```json
{
  "items": [
    {
      "id": 10,
      "version": "2.1.0",
      "release_notes": "Added GraphQL support",
      "suite_definition": {...},
      "download_count": 450,
      "created_at": "2024-06-15T10:30:00Z"
    },
    {
      "id": 8,
      "version": "2.0.0",
      "release_notes": "Major refactoring",
      "download_count": 800,
      "created_at": "2024-03-01T12:00:00Z"
    }
  ],
  "total": 5
}
```

---

### Create Version

```http
POST /api/marketplace/{slug}/versions
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "version": "2.2.0",
  "release_notes": "Bug fixes and improvements",
  "suite_definition": {
    "name": "My Test Suite",
    "tests": [...]
  }
}
```

---

### List Reviews

```http
GET /api/marketplace/{slug}/reviews
```

**Query Parameters**:
- `limit` (int): Max results (default: 20)
- `offset` (int): Pagination offset

**Response**:
```json
{
  "items": [
    {
      "id": 15,
      "user_id": 10,
      "username": "reviewer1",
      "rating": 5,
      "title": "Excellent suite!",
      "content": "Very comprehensive and well documented.",
      "created_at": "2024-06-01T15:00:00Z",
      "updated_at": "2024-06-01T15:00:00Z"
    }
  ],
  "total": 23,
  "average_rating": 4.5
}
```

---

### Add Review

```http
POST /api/marketplace/{slug}/reviews
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "rating": 5,
  "title": "Great suite!",
  "content": "Very helpful for our testing needs."
}
```

---

### Update Review

```http
PATCH /api/marketplace/reviews/{id}
Authorization: Bearer <token>
```

Only the review author can update.

---

### Delete Review

```http
DELETE /api/marketplace/reviews/{id}
Authorization: Bearer <token>
```

Only the review author or admin can delete.

---

### Install Suite

```http
POST /api/marketplace/{slug}/install
Authorization: Bearer <token>
```

Tracks that the user has installed the suite.

**Query Parameters**:
- `version` (string): Specific version to install (default: latest)

**Response**:
```json
{
  "suite_id": 1,
  "version": "2.1.0",
  "installed_at": "2024-06-20T10:00:00Z",
  "suite_definition": {...}
}
```

---

### Uninstall Suite

```http
DELETE /api/marketplace/{slug}/install
Authorization: Bearer <token>
```

---

### List Installed Suites

```http
GET /api/marketplace/installed
Authorization: Bearer <token>
```

Returns suites installed by the current user.

---

### Get Marketplace Statistics

```http
GET /api/marketplace/stats
```

**Response**:
```json
{
  "total_suites": 150,
  "total_downloads": 45000,
  "total_installs": 12000,
  "total_reviews": 890,
  "categories": [
    {"name": "api", "count": 45},
    {"name": "integration", "count": 38},
    {"name": "ui", "count": 25}
  ],
  "popular_tags": [
    {"tag": "rest", "count": 60},
    {"tag": "database", "count": 42}
  ]
}
```

---

### List Categories

```http
GET /api/marketplace/categories
```

**Response**:
```json
{
  "items": [
    {"slug": "api", "name": "API Testing", "count": 45},
    {"slug": "integration", "name": "Integration", "count": 38},
    {"slug": "ui", "name": "UI/E2E", "count": 25}
  ]
}
```

---

### GitHub Import

```http
POST /api/marketplace/import/github
Authorization: Bearer <token>
```

Import a test suite from a GitHub repository.

**Request Body**:
```json
{
  "repository_url": "https://github.com/owner/repo",
  "branch": "main",
  "suite_path": "tests/suite.yaml",
  "name": "Imported Suite",
  "category": "api"
}
```

**Response**:
```json
{
  "id": 43,
  "slug": "imported-suite",
  "imported_from": "https://github.com/owner/repo",
  "files_imported": 5
}
```

---

## Admin Endpoints

### Feature Suite

```http
POST /api/marketplace/{slug}/feature
Authorization: Bearer <admin-token>
```

Mark a suite as featured (appears prominently in listings).

---

### Unfeature Suite

```http
DELETE /api/marketplace/{slug}/feature
Authorization: Bearer <admin-token>
```

---

### Verify Suite

```http
POST /api/marketplace/{slug}/verify
Authorization: Bearer <admin-token>
```

Mark a suite as verified (quality reviewed by ATP team).

---

### Moderate Review

```http
DELETE /api/marketplace/reviews/{id}/flag
Authorization: Bearer <admin-token>
```

Remove a flagged or inappropriate review.

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid version format. Use semantic versioning (e.g., 1.0.0)"
}
```

### 403 Forbidden

```json
{
  "detail": "Permission 'marketplace:write' required"
}
```

### 404 Not Found

```json
{
  "detail": "Suite 'invalid-slug' not found"
}
```

### 409 Conflict

```json
{
  "detail": "Suite with slug 'my-suite' already exists"
}
```

---

## See Also

- [RBAC Guide](../guides/rbac-guide.md) - Permission management
- [Dashboard API](dashboard-api.md) - Core API reference
- [WebSocket Guide](../guides/websocket-guide.md) - Real-time updates
