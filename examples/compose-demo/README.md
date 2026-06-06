# ATP Compose Demo — On-Prem (variant 2)

Two core containers via docker-compose — the **ATP platform** tests an **agent
over HTTP** — plus an optional **dashboard** to browse the run. No Docker socket,
no privileges: the platform just POSTs to the agent across the compose network.
Runs fully offline.

This is the on-prem half of a two-archetype demo. The **cloud** half (variant 1)
reuses the *same* `suite.yaml` against a managed **Bedrock Agent** — only the
adapter changes (see [Cloud variant](#cloud-variant-1--bedrock) below).

```
┌─────────────┐   POST /execute (ATPRequest JSON)   ┌──────────────┐
│ atp (test)  │ ──────────────────────────────────► │ agent (HTTP) │
│  platform   │ ◄────────────────────────────────── │  under test  │
└─────┬───────┘        ATPResponse JSON              └──────────────┘
      │ writes run history (shared volume)
      ▼
┌─────────────┐
│ dashboard   │  http://localhost:8080/ui/
└─────────────┘
```

## Layout

| File | Role |
|------|------|
| `docker-compose.yml` | `agent` + `atp` (one-shot runner) + `dashboard` |
| `agent/agent.py` | Minimal FastAPI agent speaking the ATP HTTP contract |
| `agent/Dockerfile` | Builds the agent image |
| `suite.yaml` | Adapter-neutral test suite (deterministic, offline) |
| `results/` | JSON report is written here |

## Run it

### Docker
```bash
cd examples/compose-demo
docker compose up --build -d agent dashboard
docker compose run --rm atp          # one-shot test run → results/results.json
open http://localhost:8080/ui/        # browse the run (auth disabled for the demo)
docker compose down -v
```

### Podman (rootless — preferred on-prem)
```bash
cd examples/compose-demo
podman compose up --build -d agent dashboard
podman compose run --rm atp
```

Expected: `demo-001` passes — the agent returns an `output.txt` artifact and
emits no error events (`artifact_exists` + `behavior:no_errors`).

## The ATP HTTP contract (what the agent implements)

The platform's `http` adapter POSTs `ATPRequest.model_dump(mode="json")` and
validates the reply as an `ATPResponse`:

- **Request** → `POST /execute` with `{ task_id, task{description, ...},
  constraints{...}, context{...}, metadata{...} }`
- **Response** ← `{ task_id, status, artifacts[], metrics{...}, error? }`
  - `status` is `completed | failed | timeout | partial | cancelled`
  - inline file artifacts (`type:file` + `content`) are materialized by the
    platform before evaluation, so `artifact_exists` works without a shared FS.

Swap `agent/` for your own image — any HTTP service honoring that contract works.

> **SSRF guard:** the `http` adapter blocks private/internal IPs by default, and a
> compose service name (`agent`) resolves to a private address. The runner therefore
> passes `allow_internal=true` in `--adapter-config`. Drop it when targeting a public
> endpoint.

## Cloud variant (1) — Bedrock

Same `suite.yaml`, swap the adapter. The tested agent is a managed **Bedrock
Agent**; quality is judged by **Claude** via `llm_eval`.

```bash
atp test suite.yaml \
  --adapter=bedrock \
  --adapter-config agent_id=<AGENT_ID>,agent_alias_id=<ALIAS_ID>,region=us-east-1
```

- **Auth:** if the platform runs **in AWS** (EC2/ECS), the bedrock adapter uses
  the **IAM instance/task role** automatically (boto3 default credential chain) —
  no static keys. On a non-AWS host, pass `access_key_id` / `secret_access_key`.
- **LLM judge (`llm_eval`):** uncomment `demo-002` in `suite.yaml` and provide
  `ANTHROPIC_API_KEY`. The judge currently uses the Anthropic API directly;
  using **Bedrock-hosted Claude** as the judge (all-in-AWS via IAM) is a small
  planned extension.

## Notes

- The agent here is deterministic and LLM-free so the on-prem demo is reliably
  green and air-gapped. Point the adapter at a real agent for a live demo.
- `atp` and `dashboard` share the `atp-data` volume so the run shows up at
  `/ui/`. In production use a shared `ATP_DATABASE_URL` (Postgres) instead.
- `ATP_DISABLE_AUTH=true` and the demo secret are for local demos only.
