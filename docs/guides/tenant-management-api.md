# Tenant Management API

The Tenant Management API provides administrative endpoints for managing tenants in a multi-tenant ATP deployment. All endpoints require admin privileges.

## Overview

Tenants represent isolated organizations within ATP, each with their own:
- Users and agents
- Test suites and results
- Resource quotas
- Configuration settings

## Authentication

All tenant management endpoints require admin authentication. Include a valid JWT token with admin privileges in the Authorization header:

```
Authorization: Bearer <admin_token>
```

Non-admin users will receive a `403 Forbidden` response.

## Endpoints

### List Tenants

```http
GET /api/tenants
```

Returns a paginated list of tenants.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `active_only` | boolean | `true` | Only return active tenants |
| `plan` | string | - | Filter by plan (`free`, `pro`, `enterprise`) |
| `limit` | integer | `50` | Maximum results (1-100) |
| `offset` | integer | `0` | Results to skip |

**Response:**

```json
{
  "total": 15,
  "items": [
    {
      "id": "acme-corp",
      "name": "Acme Corporation",
      "plan": "enterprise",
      "is_active": true,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "limit": 50,
  "offset": 0
}
```

### Create Tenant

```http
POST /api/tenants
```

Creates a new tenant with optional quotas and settings.

**Request Body:**

```json
{
  "id": "acme-corp",
  "name": "Acme Corporation",
  "plan": "enterprise",
  "description": "Enterprise customer",
  "contact_email": "admin@acme.com",
  "quotas": {
    "max_tests_per_day": 1000,
    "max_parallel_runs": 20,
    "max_storage_gb": 100.0,
    "max_agents": 50,
    "llm_budget_monthly": 500.00,
    "max_users": 100,
    "max_suites": 200
  },
  "settings": {
    "default_timeout_seconds": 600,
    "allow_external_agents": true,
    "require_mfa": true,
    "sso_enabled": true,
    "sso_provider": "okta",
    "retention_days": 180
  }
}
```

**Tenant ID Requirements:**
- Must be 1-50 characters
- Lowercase letters, numbers, and hyphens only
- Cannot start or end with a hyphen
- Cannot be a reserved ID (`default`, `public`, `admin`, `system`, `root`, `master`)

**Response:** `201 Created` with full tenant object

### Get Tenant

```http
GET /api/tenants/{tenant_id}
```

Returns full details for a specific tenant.

**Response:**

```json
{
  "id": "acme-corp",
  "name": "Acme Corporation",
  "plan": "enterprise",
  "description": "Enterprise customer",
  "quotas": { ... },
  "settings": { ... },
  "schema_name": "tenant_acme_corp",
  "is_active": true,
  "contact_email": "admin@acme.com",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-20T14:00:00Z"
}
```

### Update Tenant

```http
PUT /api/tenants/{tenant_id}
```

Updates tenant properties. All fields are optional.

**Request Body:**

```json
{
  "name": "Acme Corporation Inc.",
  "plan": "enterprise",
  "description": "Updated description",
  "contact_email": "newadmin@acme.com",
  "is_active": true
}
```

**Response:** `200 OK` with updated tenant object

### Delete Tenant

```http
DELETE /api/tenants/{tenant_id}
```

Deletes a tenant. By default performs a soft delete (marks inactive).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hard_delete` | boolean | `false` | Permanently delete tenant and all data |

**Response:** `204 No Content`

**Warning:** Hard delete is irreversible and will drop the tenant's database schema.

## Quota Management

### Get Quotas

```http
GET /api/tenants/{tenant_id}/quotas
```

Returns current quota configuration.

**Response:**

```json
{
  "max_tests_per_day": 100,
  "max_parallel_runs": 5,
  "max_storage_gb": 10.0,
  "max_agents": 10,
  "llm_budget_monthly": 100.00,
  "max_users": 10,
  "max_suites": 50
}
```

### Update Quotas

```http
PUT /api/tenants/{tenant_id}/quotas
```

Updates all quota values.

**Request Body:**

```json
{
  "max_tests_per_day": 1000,
  "max_parallel_runs": 20,
  "max_storage_gb": 100.0,
  "max_agents": 50,
  "llm_budget_monthly": 500.00,
  "max_users": 100,
  "max_suites": 200
}
```

### Get Quota Status

```http
GET /api/tenants/{tenant_id}/quota-status
```

Returns detailed usage vs. limits for all quotas.

**Response:**

```json
{
  "tenant_id": "acme-corp",
  "checks": [
    {
      "quota_name": "max_users",
      "current_value": 45,
      "limit_value": 100,
      "percentage_used": 45.0,
      "is_exceeded": false
    },
    {
      "quota_name": "max_tests_per_day",
      "current_value": 850,
      "limit_value": 1000,
      "percentage_used": 85.0,
      "is_exceeded": false
    }
  ],
  "any_exceeded": false
}
```

## Settings Management

### Get Settings

```http
GET /api/tenants/{tenant_id}/settings
```

Returns current settings configuration.

**Response:**

```json
{
  "default_timeout_seconds": 300,
  "allow_external_agents": true,
  "require_mfa": false,
  "sso_enabled": false,
  "sso_provider": null,
  "sso_config": {},
  "custom_branding": {},
  "notification_channels": ["email"],
  "retention_days": 90
}
```

### Update Settings

```http
PUT /api/tenants/{tenant_id}/settings
```

Updates all settings values.

**Request Body:**

```json
{
  "default_timeout_seconds": 600,
  "allow_external_agents": true,
  "require_mfa": true,
  "sso_enabled": true,
  "sso_provider": "okta",
  "sso_config": {
    "domain": "acme.okta.com",
    "client_id": "xxx"
  },
  "notification_channels": ["email", "slack"],
  "retention_days": 180
}
```

## Usage Statistics

### Get Usage

```http
GET /api/tenants/{tenant_id}/usage
```

Returns current resource usage and any exceeded quotas.

**Response:**

```json
{
  "tenant": { ... },
  "usage": {
    "tenant_id": "acme-corp",
    "user_count": 45,
    "agent_count": 12,
    "suite_count": 35,
    "execution_count": 1250,
    "storage_gb": 25.5,
    "tests_today": 85,
    "llm_cost_this_month": 125.50
  },
  "quotas_exceeded": []
}
```

## Lifecycle Management

### Deactivate Tenant

```http
POST /api/tenants/{tenant_id}/deactivate
```

Soft-disables a tenant. The tenant's data is preserved but access is blocked.

**Response:** `200 OK` with updated tenant (is_active: false)

### Activate Tenant

```http
POST /api/tenants/{tenant_id}/activate
```

Re-enables a previously deactivated tenant.

**Response:** `200 OK` with updated tenant (is_active: true)

## Error Responses

| Status Code | Description |
|-------------|-------------|
| `400 Bad Request` | Invalid tenant ID or validation error |
| `401 Unauthorized` | Missing or invalid authentication |
| `403 Forbidden` | User is not an admin |
| `404 Not Found` | Tenant does not exist |
| `409 Conflict` | Tenant already exists (on create) |
| `500 Internal Server Error` | Schema creation/deletion failed |

## Data Model

### TenantQuotas

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tests_per_day` | integer | 100 | Maximum tests per day |
| `max_parallel_runs` | integer | 5 | Maximum concurrent test runs |
| `max_storage_gb` | float | 10.0 | Maximum storage in GB |
| `max_agents` | integer | 10 | Maximum number of agents |
| `llm_budget_monthly` | float | 100.00 | Monthly LLM budget (USD) |
| `max_users` | integer | 10 | Maximum number of users |
| `max_suites` | integer | 50 | Maximum number of test suites |

### TenantSettings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_timeout_seconds` | integer | 300 | Default test timeout |
| `allow_external_agents` | boolean | true | Allow external agent connections |
| `require_mfa` | boolean | false | Require MFA for all users |
| `sso_enabled` | boolean | false | Enable SSO integration |
| `sso_provider` | string | null | SSO provider name |
| `sso_config` | object | {} | SSO configuration |
| `custom_branding` | object | {} | Custom branding settings |
| `notification_channels` | array | ["email"] | Enabled notification channels |
| `retention_days` | integer | 90 | Data retention period |

## Examples

### Create Enterprise Tenant

```bash
curl -X POST http://localhost:8000/api/tenants \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "enterprise-customer",
    "name": "Enterprise Customer Inc.",
    "plan": "enterprise",
    "quotas": {
      "max_tests_per_day": 10000,
      "max_parallel_runs": 100,
      "max_storage_gb": 1000.0,
      "llm_budget_monthly": 5000.00
    },
    "settings": {
      "require_mfa": true,
      "sso_enabled": true,
      "sso_provider": "okta"
    }
  }'
```

### Update Quota Limits

```bash
curl -X PUT http://localhost:8000/api/tenants/acme-corp/quotas \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "max_tests_per_day": 2000,
    "max_parallel_runs": 50
  }'
```

### Check Quota Status

```bash
curl http://localhost:8000/api/tenants/acme-corp/quota-status \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```
