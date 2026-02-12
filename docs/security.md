# ATP Security Model

> Security architecture and guidelines for the Agent Test Platform

## 1. Overview

ATP (Agent Test Platform) provides security controls to safely execute and evaluate AI agents. This document describes the security model, threat mitigations, and best practices.

### 1.1 Security Principles

| Principle | Description |
|-----------|-------------|
| **Defense in Depth** | Multiple layers of security controls |
| **Least Privilege** | Minimal permissions for agent execution |
| **Input Validation** | All inputs validated before processing |
| **Secret Protection** | Automatic redaction of sensitive data |
| **Isolation** | Sandboxed execution environments |

### 1.2 Trust Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trust Boundary                            │
├─────────────────────────────────────────────────────────────────┤
│  Trusted:                                                        │
│  - ATP Platform code                                             │
│  - Test definitions (YAML) from trusted sources                  │
│  - Configuration files                                           │
├─────────────────────────────────────────────────────────────────┤
│  Untrusted:                                                      │
│  - Agent responses and artifacts                                 │
│  - External HTTP endpoints                                       │
│  - Docker images (unless from trusted registry)                  │
│  - User-provided input data                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Input Validation

### 2.1 Protocol Validation

All ATP Protocol messages are validated using Pydantic models with strict constraints:

#### Task ID Validation
```python
# Must match: alphanumeric, underscore, hyphen only
# Max length: 128 characters
TASK_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
```

#### Path Validation
- No path traversal (`..`)
- No absolute paths in artifacts
- No null bytes
- Maximum length: 4096 characters

#### Description Validation
- Maximum length: 100,000 characters
- Stripped of leading/trailing whitespace

### 2.2 URL Validation (SSRF Prevention)

The HTTP adapter validates URLs to prevent Server-Side Request Forgery:

**Blocked:**
- Internal/private IP ranges (10.x, 172.16.x, 192.168.x, 127.x)
- Cloud metadata endpoints (169.254.169.254)
- `localhost` and `localhost.localdomain`
- `file://` and other non-HTTP schemes

**Configuration:**
```yaml
agents:
  my-agent:
    type: http
    endpoint: "https://api.example.com/agent"
    allow_internal: false  # Default: blocked
```

### 2.3 Docker Image Validation

Docker image names are validated against a strict pattern:

```
^(?:registry/)?name(:tag)?$
```

**Examples:**
- ✅ `nginx:latest`
- ✅ `registry.io/myagent:v1.0`
- ❌ `../../../etc/passwd`
- ❌ `; rm -rf /`

---

## 3. Sandbox Isolation

### 3.1 Workspace Isolation

Each test runs in an isolated sandbox directory:

```
/tmp/atp-sandboxes/
└── sandbox-{uuid}/
    ├── workspace/     # Agent working directory
    ├── logs/          # Execution logs
    └── artifacts/     # Output artifacts
```

**Security features:**
- Full UUID (32 chars) prevents collision attacks
- Restrictive permissions (0o700 - owner only)
- Path validation prevents traversal escapes
- Automatic cleanup after execution

### 3.2 Container Isolation

Docker containers run with security hardening:

```yaml
# Default security settings
security:
  no_new_privileges: true
  read_only_root: false  # Optional
  cap_drop: ["ALL"]
  pids_limit: 256
  memory_swap: {memory_limit}  # Prevent swap abuse
```

**Network isolation:**
```yaml
agents:
  secure-agent:
    type: container
    network: "none"  # Completely isolated (default)
```

### 3.3 Process Isolation (CLI Adapter)

CLI adapters run with:
- Filtered environment variables
- Minimal PATH
- Secure temporary files (0o600 permissions)
- Timeout enforcement

---

## 4. Secret Protection

### 4.1 Environment Variable Filtering

Sensitive environment variables are automatically filtered from subprocess environments:

**Blocked patterns:**
- `*_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`
- `AWS_*`, `AZURE_*`, `GCP_*`, `GOOGLE_*`
- `ANTHROPIC_*`, `OPENAI_*`
- `DATABASE_*`, `DB_*`

**Safe allowlist:**
- `PATH`, `HOME`, `USER`, `SHELL`, `TERM`
- `LANG`, `LC_*`, `TZ`
- `PYTHONPATH`, `VIRTUAL_ENV`
- `ATP_*` (platform-specific)

**Configuration:**
```yaml
agents:
  my-cli-agent:
    type: cli
    inherit_environment: false  # Default: don't inherit
    allowed_env_vars:          # Additional safe vars
      - MY_SAFE_VAR
```

### 4.2 Log Sanitization

Secrets are automatically redacted from:
- Error messages
- Log output
- Adapter responses

**Detected patterns:**
- API keys (`api_key=...`)
- Bearer tokens (`bearer ...`)
- Passwords (`password=...`)
- AWS access keys (`AKIA...`)
- Private keys (`-----BEGIN...PRIVATE KEY-----`)

**Redaction:**
```
Before: "api_key=sk-1234567890abcdef"
After:  "api_key=[REDACTED]"
```

### 4.3 Secure Logging

Use `SecureFormatter` for automatic redaction:

```python
from atp.core.security import setup_secure_logging

setup_secure_logging()  # Configure root logger
```

---

## 5. Resource Limits

### 5.1 Container Resources

| Resource | Default | Maximum |
|----------|---------|---------|
| Memory | 2GB | 16GB |
| CPU | 1 core | 16 cores |
| PIDs | 256 | 256 |
| Timeout | 300s | 3600s |

### 5.2 Validation

Resource limits are validated before execution:

```python
from atp.core.security import parse_memory_limit, validate_cpu_limit

# Memory: accepts "512m", "2g", "1Gi"
bytes_limit = parse_memory_limit("2g")

# CPU: accepts "1", "0.5", "2"
cpu_limit = validate_cpu_limit("1.5")
```

---

## 6. Network Security

### 6.1 Network Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `none` | No network access | Secure default |
| `bridge` | Docker bridge network | Local testing |
| `host` | Host network (dangerous) | Development only |

### 6.2 SSRF Mitigation

All outbound HTTP requests are validated:

```python
from atp.core.security import validate_url

# Raises SecurityValidationError for:
# - Internal IPs (10.x, 172.16.x, 192.168.x)
# - Localhost
# - Cloud metadata endpoints
validate_url("https://api.example.com/endpoint")
```

---

## 7. Threat Model

### 7.1 Threats and Mitigations

| Threat | Severity | Mitigation |
|--------|----------|------------|
| Path traversal in artifacts | HIGH | Path validation, workspace containment |
| SSRF via HTTP adapter | HIGH | URL validation, IP blocking |
| Command injection (CLI) | HIGH | Shell escaping, env filtering |
| Docker escape | HIGH | Capability drops, no-new-privileges |
| Secret leakage in logs | MEDIUM | Automatic redaction |
| Resource exhaustion | MEDIUM | Memory/CPU limits, PID limits |
| Environment pollution | MEDIUM | Filtered env inheritance |

### 7.2 Assumptions

- Test definitions (YAML) come from trusted sources
- ATP Platform runs with appropriate OS-level permissions
- Docker daemon is properly secured
- File system permissions are correctly configured

### 7.3 Out of Scope

- Multi-tenant isolation (ATP is single-user)
- Encrypted storage
- Network intrusion detection

> **Note**: Authentication/authorization is implemented in the dashboard module with full JWT auth and RBAC support.

---

## 8. Security Configuration

### 8.1 Recommended Production Settings

```yaml
# atp.config.yaml
defaults:
  timeout_seconds: 300

agents:
  production-agent:
    type: container
    image: "registry.internal/agent:v1"
    resources:
      memory: "2g"
      cpu: "1"
    network: "none"
    no_new_privileges: true
    cap_drop: ["ALL"]
    read_only_root: true
    allowed_volume_paths:
      - "/data/inputs"
```

### 8.2 Development Settings

```yaml
agents:
  dev-agent:
    type: http
    endpoint: "http://localhost:8000/agent"
    allow_internal: true  # Only for development!
```

---

## 9. Security Checklist

### Before Production Deployment

- [ ] All agents use `network: "none"` or explicit allowlist
- [ ] Container images are from trusted registry
- [ ] `allow_internal: false` for HTTP adapters
- [ ] `inherit_environment: false` for CLI adapters
- [ ] Resource limits are configured appropriately
- [ ] Secure logging is enabled
- [ ] Volume mounts are restricted to necessary paths

### Regular Security Review

- [ ] Review adapter configurations
- [ ] Audit environment variable allowlists
- [ ] Check for exposed secrets in logs
- [ ] Verify sandbox cleanup is working
- [ ] Test resource limit enforcement

---

## 10. API Reference

### Core Security Functions

```python
from atp.core.security import (
    # Path validation
    validate_path_within_workspace,
    sanitize_filename,
    open_file_safely,  # TOCTOU-safe file opening

    # URL validation
    validate_url,
    validate_url_with_dns,  # With DNS resolution check

    # Docker validation
    validate_docker_image,
    validate_docker_network,
    validate_volume_mount,

    # Resource validation
    parse_memory_limit,
    validate_cpu_limit,
    validate_timeout,

    # Input size validation
    validate_request_size,
    validate_object_depth,
    validate_artifacts_count,
    validate_env_vars_count,

    # Command validation
    validate_command,
    escape_shell_arg,

    # Secret handling
    redact_secrets,
    filter_environment_variables,
    is_sensitive_env_var,
    redact_dict_secrets,
    sanitize_env_value,

    # Error/Log sanitization
    sanitize_log_message,
    sanitize_error_message,

    # Logging
    SecureFormatter,
    setup_secure_logging,
    log_security_event,
    SecurityEventType,
)
```

### Exception Handling

```python
from atp.core.security import SecurityValidationError, SecurityEventType

try:
    validate_url("http://localhost/internal")
except SecurityValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Field: {e.field}")
    print(f"Event type: {e.event_type}")
```

### Security Event Types

| Event Type | Description |
|------------|-------------|
| `VALIDATION_FAILURE` | General input validation failure |
| `PATH_TRAVERSAL_ATTEMPT` | Path traversal attack detected |
| `SSRF_ATTEMPT` | Server-side request forgery attempt |
| `SECRET_REDACTED` | Secret was detected and redacted |
| `RESOURCE_LIMIT_EXCEEDED` | Resource limit exceeded |
| `SUSPICIOUS_INPUT` | Suspicious input pattern detected |
| `SANDBOX_VIOLATION` | Sandbox security violation |

---

## 11. Input Size Limits

The following size limits are enforced:

| Limit | Value | Description |
|-------|-------|-------------|
| `MAX_REQUEST_SIZE_BYTES` | 10 MB | Maximum request body size |
| `MAX_RESPONSE_SIZE_BYTES` | 100 MB | Maximum response body size |
| `MAX_ARTIFACTS_COUNT` | 1,000 | Maximum artifacts per response |
| `MAX_ENV_VARS_COUNT` | 100 | Maximum environment variables |
| `MAX_OBJECT_DEPTH` | 50 | Maximum nested object depth |
| `MAX_ARRAY_SIZE` | 10,000 | Maximum array elements |
| `MAX_METADATA_KEYS` | 50 | Maximum metadata keys |

---

## 12. Command Validation

CLI adapter commands are validated against an allowlist:

**Allowed binaries:**
- Python: `python`, `python3`
- Node.js: `node`, `npm`, `npx`
- Other: `bash`, `sh`, `java`, `go`, `cargo`, `docker`, `kubectl`, `git`

**Blocked patterns:**
- Shell metacharacters: `;`, `&`, `|`, `` ` ``, `$`, `(`, `)`, `{`, `}`, `<`, `>`
- Null bytes
- Newlines

---

## 13. Security Audit Logging

Security events are automatically logged for audit purposes:

```python
from atp.core.security import setup_secure_logging

# Configure secure logging on startup
setup_secure_logging(include_audit_logger=True)
```

Audit logs include:
- Timestamp
- Event type
- Message (redacted)
- Field name
- Source IP (if available)

---

## 14. Reporting Security Issues

If you discover a security vulnerability in ATP:

1. **Do not** open a public issue
2. Contact the maintainers directly
3. Provide detailed reproduction steps
4. Allow reasonable time for a fix

---

*Document Version: 2.0*
*Last Updated: 2026-01-23*
