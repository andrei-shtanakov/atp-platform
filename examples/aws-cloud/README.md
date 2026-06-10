# ATP Cloud Variant — all-in-AWS via IAM

The cloud archetype of the demo (variant 1). The **same** suite (including the
`method/` agent-eval-case sweep) that runs on-prem now runs in AWS — you swap the
adapter. Both the **tested agent** (a managed **Bedrock Agent**) and the
**grader** (**Bedrock-hosted Claude**) authenticate with the instance's **IAM
role** — no API keys, no static AWS credentials.

```
EC2/ECS (IAM role)
  ├─ adapter=bedrock ───────────────► Bedrock Agent (under test)   ← IAM
  └─ ATP_JUDGE_PROVIDER=bedrock ────► Bedrock-Claude (the judge)   ← IAM
```

## Prerequisites

- An AWS account with **Bedrock model access enabled** for the Claude model you
  will use as the judge (Bedrock console → Model access), in your region.
- A deployed **Bedrock Agent** to test — note its **agent id** and **alias id**.
- An IAM **role** (instance/task role) with `iam-policy.json` attached.

## 1. IAM role

Attach [`iam-policy.json`](iam-policy.json) to the role the compute will assume.
It grants exactly two things:

- `bedrock:InvokeAgent` — the bedrock adapter invokes the agent under test;
- `bedrock:InvokeModel*` — the Bedrock-Claude judge invokes the model.

The policy ships **scoped to placeholder ARNs** (no `Resource: "*"`). Replace
`REGION`, `ACCOUNT_ID`, `AGENT_ID`, `ALIAS_ID`, and the model id with your values
— the agent alias (`arn:aws:bedrock:REGION:ACCOUNT_ID:agent-alias/AGENT_ID/ALIAS_ID`)
and the model / inference-profile
(`arn:aws:bedrock:REGION::foundation-model/<MODEL_ID>`).

## 2. Compute

### EC2 (simplest — mirrors the VPS Docker setup)

Launch an instance (Amazon Linux 2023) with:
- the IAM role from step 1,
- [`user-data.sh`](user-data.sh) (installs Docker, builds the image, starts the
  dashboard **bound to localhost** with auth enabled).

> **Do not** open port 8080 in the security group. Reach the dashboard via an SSH
> tunnel: `ssh -L 8080:localhost:8080 ec2-user@<instance>`, then
> `http://localhost:8080/ui/`.

> ⚠️ **IMDSv2 hop limit.** A container reaches the instance role via the metadata
> service. With the default hop limit of **1**, containers cannot read it. Set the
> instance metadata **`HttpPutResponseHopLimit` to 2** (launch option
> `--metadata-options HttpTokens=required,HttpPutResponseHopLimit=2`), otherwise
> boto3/the Bedrock client gets no credentials.

### ECS / Fargate (managed alternative)

Run the image as a task with a **task role** carrying `iam-policy.json`. Fargate
exposes the task role to the container automatically (no hop-limit caveat).

## 3. Run the methodology sweep (all-in-AWS)

On the instance (image built, IAM role active). The judge is steered to Bedrock
by env (requires the judge-provider-env change); the tested agent by the adapter:

```bash
docker run --rm \
  -e ATP_JUDGE_PROVIDER=bedrock \
  -e ATP_JUDGE_REGION=us-east-1 \
  -e ATP_JUDGE_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0 \
  -v atp-data:/root/.atp \
  -v /opt/atp-platform/method:/work/method:ro \
  atp-platform:latest \
  uv run --no-sync atp test /work/method/cases/req-extraction \
    --adapter=bedrock \
    --adapter-config agent_id=<AGENT_ID>,agent_alias_id=<ALIAS_ID>,region=us-east-1 \
    --runs=5
```

No `ANTHROPIC_API_KEY`, no AWS keys — both the agent and the judge use the IAM
role. The same `critical_check` hard-gates the trap; `--runs` surfaces the curve
of collapse. The run is stored in the shared `atp-data` volume, browsable in the
dashboard via the SSH tunnel above (`http://localhost:8080/ui/`).

## The pitch in one line

The **identical** cases run on-prem (`--adapter=http`, local model) and in the
cloud (`--adapter=bedrock`, Bedrock-Claude judge) — the adapter and a couple of
env vars are all that change.

## Notes

- **Database:** defaults to SQLite on the instance (ephemeral). For durability
  use **RDS Postgres** via `ATP_DATABASE_URL=postgresql+asyncpg://…` (build the
  image with the `postgres` extra).
- **Cost:** every agent and judge call is a paid Bedrock invocation — keep
  `--runs` modest for a demo.
- The Bedrock paths need `boto3` (the agent adapter and the judge's
  `AsyncAnthropicBedrock`). The image installs it via `--extra bedrock`, alongside
  `--extra llm` (anthropic + openai) and `--all-packages` (plugins) — see the
  `Dockerfile`. `boto3` is otherwise an optional extra, not a base dependency.
