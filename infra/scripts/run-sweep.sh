#!/usr/bin/env bash
# Run the methodology sweep on the instance over SSH. Pulls all values from
# Terraform outputs — no hand-copying of agent ids / model / region.
#
#   ./run-sweep.sh <path-to-ssh-key.pem> [RUNS]
#
# Both the agent (under test) and the judge authenticate via the instance IAM
# role — no ANTHROPIC_API_KEY, no AWS keys. method/cases ship inside the image
# (Dockerfile COPY), so we reference the in-container path, no host mount.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="${TF_DIR:-$HERE/../terraform}"

KEY="${1:?usage: run-sweep.sh <ssh-key.pem> [RUNS]}"
RUNS="${2:-3}"   # modest by default — Sonnet 4.5 calls are billed. NOTE: req-extraction
                 # is a 4-case sweep, so total ≈ 4*RUNS agent + 4*RUNS judge calls
                 # (RUNS=3 -> ~24). Bump cautiously.

REGION="$(terraform -chdir="$TF_DIR" output -raw region)"
DNS="$(terraform -chdir="$TF_DIR" output -raw instance_public_dns)"
AGENT_ID="$(terraform -chdir="$TF_DIR" output -raw agent_id)"
ALIAS_ID="$(terraform -chdir="$TF_DIR" output -raw agent_alias_id)"
JUDGE_MODEL="$(terraform -chdir="$TF_DIR" output -raw judge_model)"

echo ">> ssh ec2-user@$DNS  | runs=$RUNS  region=$REGION"

ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ec2-user@$DNS" \
  REGION="$REGION" AGENT_ID="$AGENT_ID" ALIAS_ID="$ALIAS_ID" \
  JUDGE_MODEL="$JUDGE_MODEL" RUNS="$RUNS" 'bash -s' <<'REMOTE'
set -euo pipefail
docker run --rm \
  -e ATP_JUDGE_PROVIDER=bedrock \
  -e ATP_JUDGE_REGION="$REGION" \
  -e ATP_JUDGE_MODEL="$JUDGE_MODEL" \
  -v atp-data:/root/.atp \
  atp-platform:latest \
  uv run --no-sync atp test method/cases/req-extraction \
    --adapter=bedrock \
    --adapter-config "agent_id=$AGENT_ID,agent_alias_id=$ALIAS_ID,region=$REGION" \
    --runs="$RUNS"
REMOTE

echo ">> Done. Results are in the atp-data volume — browse via the dashboard:"
echo "   ssh -i $KEY -L 8080:localhost:8080 ec2-user@$DNS   then open http://localhost:8080/ui/"
