# Role-Based Access Control (RBAC) Guide

This guide explains how to use the Role-Based Access Control system in the ATP Dashboard to manage user permissions and access to resources.

## Overview

The ATP Dashboard implements a fine-grained RBAC system that controls access to API endpoints based on user roles and permissions. Each user can be assigned one or more roles, and each role grants a set of permissions that determine what actions the user can perform.

## Key Concepts

### Permissions

Permissions are the atomic units of access control. Each permission follows the format `resource:action`:

- **Resource**: The type of resource being accessed (e.g., `suites`, `agents`, `analytics`)
- **Action**: The operation being performed (`read`, `write`, `execute`, `delete`, `export`)

### Roles

Roles are named collections of permissions. Users are assigned roles, and inherit all permissions granted by those roles. ATP provides four default system roles:

| Role | Description |
|------|-------------|
| `admin` | Full administrative access to all resources |
| `developer` | Create and execute test suites, manage agents |
| `analyst` | View results and analytics, export data |
| `viewer` | Read-only access to view test results |

### System vs Custom Roles

- **System roles** (admin, developer, analyst, viewer) are predefined and cannot be deleted or have their permissions modified
- **Custom roles** can be created, updated, and deleted by administrators

## Available Permissions

### Suite Permissions
| Permission | Description |
|------------|-------------|
| `suites:read` | View test suites and definitions |
| `suites:write` | Create and update test suites |
| `suites:execute` | Run test executions |
| `suites:delete` | Delete test suites |

### Agent Permissions
| Permission | Description |
|------------|-------------|
| `agents:read` | View agents |
| `agents:write` | Create and update agents |
| `agents:execute` | Execute agent operations |
| `agents:delete` | Delete agents |

### Results Permissions
| Permission | Description |
|------------|-------------|
| `results:read` | View test results, comparisons, leaderboards |
| `results:write` | Modify test results |
| `results:delete` | Delete test results |

### Analytics Permissions
| Permission | Description |
|------------|-------------|
| `analytics:read` | View trends, anomalies, correlations |
| `analytics:write` | Create and update scheduled reports |
| `analytics:delete` | Delete scheduled reports |
| `analytics:export` | Export data to CSV/Excel |

### Budget Permissions
| Permission | Description |
|------------|-------------|
| `budgets:read` | View budgets and usage |
| `budgets:write` | Create and update budgets |
| `budgets:delete` | Delete budgets |

### Marketplace Permissions
| Permission | Description |
|------------|-------------|
| `marketplace:read` | View marketplace suites |
| `marketplace:write` | Publish, update, and review suites |
| `marketplace:delete` | Delete own suites and reviews |
| `marketplace:admin` | Feature, verify, and moderate suites |

### Public Leaderboard Permissions
| Permission | Description |
|------------|-------------|
| `leaderboard:read` | View public leaderboard |
| `leaderboard:write` | Publish results, manage agent profiles |
| `leaderboard:admin` | Manage categories, add verification badges |

### Administrative Permissions
| Permission | Description |
|------------|-------------|
| `users:read` | View user information |
| `users:write` | Manage user accounts |
| `users:delete` | Delete user accounts |
| `roles:read` | View roles and permissions |
| `roles:write` | Create and update roles |
| `roles:delete` | Delete custom roles |
| `tenants:read` | View tenant information |
| `tenants:write` | Manage tenants |
| `tenants:delete` | Delete tenants |

## Default Role Permissions

### Admin Role
All permissions (full access to all resources).

### Developer Role
```
suites:read, suites:write, suites:execute
agents:read, agents:write, agents:execute
results:read, results:write
baselines:read, baselines:write
settings:read
budgets:read, budgets:write
analytics:read, analytics:write, analytics:delete, analytics:export
marketplace:read, marketplace:write
leaderboard:read, leaderboard:write
roles:read
```

### Analyst Role
```
suites:read
agents:read
results:read
baselines:read
budgets:read
analytics:read, analytics:write, analytics:export
marketplace:read
leaderboard:read
roles:read
```

### Viewer Role
```
suites:read
agents:read
results:read
baselines:read
budgets:read
analytics:read
marketplace:read
leaderboard:read
roles:read
```

## Using RBAC in the API

### Authentication

First, obtain an authentication token:

```bash
# Login to get access token
TOKEN=$(curl -s -X POST "http://localhost:8000/api/auth/token" \
  -d "username=admin&password=secret" | jq -r '.access_token')

# Use the token in subsequent requests
curl -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/agents"
```

### Checking Your Permissions

Get your current permissions:

```bash
curl -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/roles/me/permissions"
```

Response:
```json
{
  "user_id": 1,
  "username": "admin",
  "is_admin": true,
  "roles": [
    {
      "id": 1,
      "name": "admin",
      "permissions": ["suites:read", "suites:write", ...]
    }
  ],
  "permissions": ["suites:read", "suites:write", "agents:read", ...]
}
```

### Managing Roles (Admin Only)

#### List All Roles

```bash
curl -H "Authorization: Bearer $TOKEN" "http://localhost:8000/api/roles"
```

#### Create a Custom Role

```bash
curl -X POST "http://localhost:8000/api/roles" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-runner",
    "description": "Can only run tests and view results",
    "permissions": ["suites:read", "suites:execute", "results:read"]
  }'
```

#### Update a Custom Role

```bash
curl -X PATCH "http://localhost:8000/api/roles/5" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "permissions": ["suites:read", "suites:execute", "results:read", "agents:read"]
  }'
```

#### Delete a Custom Role

```bash
curl -X DELETE "http://localhost:8000/api/roles/5" \
  -H "Authorization: Bearer $TOKEN"
```

### Managing User Role Assignments (Admin Only)

#### View User's Roles

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/roles/users/5/roles"
```

#### Assign Role to User

```bash
curl -X POST "http://localhost:8000/api/roles/users/assign" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 5,
    "role_id": 2
  }'
```

#### Remove Role from User

```bash
curl -X DELETE "http://localhost:8000/api/roles/users/5/roles/2" \
  -H "Authorization: Bearer $TOKEN"
```

## Error Responses

### 401 Unauthorized

The request lacks valid authentication credentials:

```json
{
  "detail": "Not authenticated"
}
```

### 403 Forbidden

The authenticated user lacks the required permission:

```json
{
  "detail": "Permission 'agents:write' required"
}
```

## Best Practices

### 1. Principle of Least Privilege

Assign users the minimum set of permissions needed to perform their tasks:

- Use the `viewer` role for users who only need to see results
- Use the `analyst` role for users who need export capabilities
- Reserve `developer` and `admin` roles for trusted users

### 2. Use Custom Roles for Specific Needs

If default roles don't fit your needs, create custom roles:

```bash
# Create a role for CI/CD pipelines that can only run tests
curl -X POST "http://localhost:8000/api/roles" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ci-runner",
    "description": "Automated CI/CD test execution",
    "permissions": ["suites:read", "suites:execute", "results:read"]
  }'
```

### 3. Audit Role Assignments

Regularly review user role assignments:

```bash
# Get all users with a specific role
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/roles/2" # Returns role details with user count
```

### 4. Avoid Over-Privileged Users

- Don't assign `admin` role to regular users
- Don't assign `write` permissions when only `read` is needed
- Review and revoke unnecessary permissions periodically

## Integrating RBAC in Custom Code

### Python Example

```python
from atp.dashboard.rbac import Permission, require_permission
from fastapi import Depends
from typing import Annotated

# Protect an endpoint with permission check
@router.get("/my-endpoint")
async def my_endpoint(
    _: Annotated[None, Depends(require_permission(Permission.SUITES_READ))],
) -> dict:
    """This endpoint requires SUITES_READ permission."""
    return {"message": "Access granted"}
```

### Checking Permissions Programmatically

```python
from atp.dashboard.rbac import has_permission, get_user_permissions, Permission

# Check if user has a specific permission
if has_permission(user, roles, Permission.AGENTS_WRITE):
    # Perform write operation
    pass

# Get all effective permissions for a user
permissions = get_user_permissions(user, roles)
print(f"User has {len(permissions)} permissions")
```

## Troubleshooting

### "Permission required" Error

1. Check your current permissions: `GET /api/roles/me/permissions`
2. Verify the endpoint's required permission in the API documentation
3. Contact an admin to request the needed role/permission

### Role Assignment Not Working

1. Verify you have admin privileges
2. Check that the role exists and is active
3. Ensure the user ID is correct

### Custom Role Not Visible

1. Custom roles may need `is_active: true` to appear in listings
2. Use `include_inactive=true` query parameter to see inactive roles

## See Also

- [Dashboard API Reference](../reference/dashboard-api.md) - Complete API documentation
- [Authentication Guide](authentication.md) - User authentication setup
- [Tenant Management](tenant-management-api.md) - Multi-tenant configuration
