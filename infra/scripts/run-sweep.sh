#!/usr/bin/env bash
# Run the methodology sweep on the instance. Pulls all values from Terraform
# outputs — no hand-copying of agent ids / model / region. Two access modes:
#
#   ./run-sweep.sh <path-to-ssh-key.pem> [RUNS]   # SSH (default mode, enable_ssh=true)
#   ./run-sweep.sh --ssm [RUNS]                    # SSM-only (no key, no open port)
#
# Both the agent (under test) and the judge authenticate via the instance IAM
# role — no ANTHROPIC_API_KEY, no AWS keys. method/cases ship inside the image
# (Dockerfile COPY), so we reference the in-container path, no host mount.
#
# NOTE: req-extraction is a 4-case sweep, so total ≈ 4*RUNS agent + 4*RUNS judge
# calls (RUNS=3 -> ~24). All Sonnet 4.5 calls are billed — bump cautiously.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="${TF_DIR:-$HERE/../terraform}"

# --- mode parsing ------------------------------------------------------------
if [ "${1:-}" = "--ssm" ]; then
  MODE="ssm"
  RUNS="${2:-3}"
else
  MODE="ssh"
  KEY="${1:?usage: run-sweep.sh <ssh-key.pem> [RUNS]  |  run-sweep.sh --ssm [RUNS]}"
  RUNS="${2:-3}"
fi

tf() { terraform -chdir="$TF_DIR" output -raw "$1"; }
REGION="$(tf region)"
AGENT_ID="$(tf agent_id)"
ALIAS_ID="$(tf agent_alias_id)"
JUDGE_MODEL="$(tf judge_model)"

# The exact command to run on the box. No embedded double quotes, so it also
# survives JSON embedding for SSM send-command below.
DOCKER_CMD="docker run --rm \
-e ATP_JUDGE_PROVIDER=bedrock -e ATP_JUDGE_REGION=$REGION -e ATP_JUDGE_MODEL=$JUDGE_MODEL \
-v atp-data:/root/.atp atp-platform:latest \
uv run --no-sync atp test method/cases/req-extraction \
--adapter=bedrock --adapter-config agent_id=$AGENT_ID,agent_alias_id=$ALIAS_ID,region=$REGION \
--runs=$RUNS"

if [ "$MODE" = "ssh" ]; then
  DNS="$(tf instance_public_dns)"
  echo ">> ssh ec2-user@$DNS  | runs=$RUNS  region=$REGION"
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ec2-user@$DNS" "$DOCKER_CMD"
  echo ">> Done. Results are in the atp-data volume — browse via the dashboard:"
  echo "   ssh -i $KEY -L 8080:localhost:8080 ec2-user@$DNS   then open http://localhost:8080/ui/"
  exit 0
fi

# --- SSM mode: send-command + poll (no SSH, no open port) --------------------
INSTANCE_ID="$(tf instance_id)"
echo ">> ssm send-command -> $INSTANCE_ID  | runs=$RUNS  region=$REGION"

PARAMS="$(mktemp)"
trap 'rm -f "$PARAMS"' EXIT
printf '{"commands":["%s"]}' "$DOCKER_CMD" >"$PARAMS"

CMD_ID="$(aws ssm send-command --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --comment "ATP req-extraction sweep RUNS=$RUNS" \
  --timeout-seconds 3600 \
  --parameters "file://$PARAMS" \
  --query 'Command.CommandId' --output text)"
echo ">> command id: $CMD_ID  (polling; the sweep takes a few minutes)"

# Custom poll loop — the built-in `command-executed` waiter caps at ~100s, too
# short for a multi-minute billed sweep. Cap ~30 min.
STATUS="Pending"
for _ in $(seq 1 360); do
  STATUS="$(aws ssm get-command-invocation --region "$REGION" \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'Status' --output text 2>/dev/null || echo Pending)"
  case "$STATUS" in
    Success | Failed | Cancelled | TimedOut) break ;;
  esac
  sleep 5
done

echo ">> status: $STATUS"
echo ">> ---- stdout (SSM truncates inline output; full results in the dashboard) ----"
aws ssm get-command-invocation --region "$REGION" \
  --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
  --query 'StandardOutputContent' --output text 2>/dev/null | tail -n 30 || true
if [ "$STATUS" != "Success" ]; then
  echo ">> ---- stderr ----" >&2
  aws ssm get-command-invocation --region "$REGION" \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'StandardErrorContent' --output text 2>/dev/null | tail -n 30 >&2 || true
fi

echo ">> Results are in the atp-data volume — browse via the dashboard over an SSM port-forward:"
echo "   aws ssm start-session --target $INSTANCE_ID \\"
echo "     --document-name AWS-StartPortForwardingSession \\"
echo "     --parameters '{\"portNumber\":[\"8080\"],\"localPortNumber\":[\"8080\"]}'"

[ "$STATUS" = "Success" ]
