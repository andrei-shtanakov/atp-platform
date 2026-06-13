#!/usr/bin/env bash
# Preflight — catch the expensive failure modes BEFORE a paid run. Runs from the
# operator machine with YOUR AWS creds (SSO/profile), not the instance role
# (which is scoped to invoke-only and can't list/describe).
#
#   ./preflight.sh           # free checks only (no Bedrock charges)
#   ./preflight.sh --invoke  # also do ONE tiny paid InvokeModel to prove access
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="${TF_DIR:-$HERE/../terraform}"

REGION="$(terraform -chdir="$TF_DIR" output -raw region)"
PROFILE_ID="$(terraform -chdir="$TF_DIR" output -raw judge_model)"
AGENT_ID="$(terraform -chdir="$TF_DIR" output -raw agent_id)"
ALIAS_ID="$(terraform -chdir="$TF_DIR" output -raw agent_alias_id)"

ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAILED=1; }
FAILED=0

echo "Region=$REGION  profile=$PROFILE_ID"
echo "Agent=$AGENT_ID  alias=$ALIAS_ID"
echo

echo "[1/3] Inference profile reachable in region (model access + region OK)?"
if aws bedrock get-inference-profile --region "$REGION" \
     --inference-profile-identifier "$PROFILE_ID" >/dev/null 2>&1; then
  ok "inference profile $PROFILE_ID resolves"
else
  fail "profile not found — wrong region, or Sonnet 4.5 model access not enabled in the console"
fi

echo "[2/3] Agent alias PREPARED?"
STATUS="$(aws bedrock-agent get-agent-alias --region "$REGION" \
            --agent-id "$AGENT_ID" --agent-alias-id "$ALIAS_ID" \
            --query 'agentAlias.agentAliasStatus' --output text 2>/dev/null || true)"
if [ "$STATUS" = "PREPARED" ]; then
  ok "alias status=PREPARED"
else
  fail "alias status=${STATUS:-<none>} (expected PREPARED)"
fi

echo "[3/3] Optional paid InvokeModel (judge path end-to-end)"
if [ "${1:-}" = "--invoke" ]; then
  BODY='{"anthropic_version":"bedrock-2023-05-31","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'
  # AWS CLI v2: pass raw JSON and let the CLI base64-encode it. Avoids manual
  # `base64`, whose 76-col line wrapping (BSD/macOS) corrupts the payload.
  if aws bedrock-runtime invoke-model --region "$REGION" \
       --model-id "$PROFILE_ID" \
       --content-type application/json --accept application/json \
       --cli-binary-format raw-in-base64-out \
       --body "$BODY" /dev/stdout >/dev/null 2>&1; then
    ok "InvokeModel succeeded (judge will work)"
  else
    fail "InvokeModel denied — check IAM destination-region ARNs and model access"
  fi
else
  echo "  (skipped — pass --invoke to run a ~1-token paid check)"
fi

echo
if [ "$FAILED" = "1" ]; then echo "Preflight FAILED — fix above before running the sweep."; exit 1; fi
echo "Preflight passed."
