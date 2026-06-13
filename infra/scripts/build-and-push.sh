#!/usr/bin/env bash
# Build the ATP image from the repo's Dockerfile and push it to the ECR repo
# created by Terraform. Run from the operator machine with AWS creds (SSO/profile)
# that can push to ECR. The image already contains method/cases (Dockerfile does
# `COPY . .`), so no host mount is needed on the instance.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="${TF_DIR:-$HERE/../terraform}"
# Project root = two levels up from infra/scripts/ (…/atp-platform).
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}"

# REGION/ECR_URL may be injected via env (e.g. by the terraform_data provisioner
# during `apply`, when outputs aren't finalized yet). Otherwise read TF outputs.
REGION="${REGION:-$(terraform -chdir="$TF_DIR" output -raw region)}"
ECR_URL="${ECR_URL:-$(terraform -chdir="$TF_DIR" output -raw ecr_repo_url)}"
TAG="${1:-latest}"
REGISTRY="${ECR_URL%%/*}"

# Container engine: docker or podman (either is fine — both are CLI-compatible
# for login/build/push). Override with CONTAINER_ENGINE=docker|podman.
ENGINE="${CONTAINER_ENGINE:-$(command -v docker || command -v podman || true)}"
[ -n "$ENGINE" ] || {
  echo "No container engine found — need docker or podman in PATH (or set CONTAINER_ENGINE)." >&2
  exit 1
}
# Validate it's a real executable — catches a bad CONTAINER_ENGINE override (or a
# stale path) here with a clear message instead of a generic failure mid-build.
command -v "$ENGINE" >/dev/null 2>&1 || {
  echo "Container engine '$ENGINE' not found in PATH (check CONTAINER_ENGINE)." >&2
  exit 1
}
ENGINE="$(basename "$ENGINE")"

echo ">> Engine:      $ENGINE"
echo ">> Repo root:   $REPO_ROOT"
echo ">> ECR target:  $ECR_URL:$TAG"

test -f "$REPO_ROOT/Dockerfile" || { echo "Dockerfile not found at $REPO_ROOT" >&2; exit 1; }

aws ecr get-login-password --region "$REGION" \
  | "$ENGINE" login --username AWS --password-stdin "$REGISTRY"

# linux/amd64 to match the t3 (x86_64) instance even if you build on Apple silicon.
"$ENGINE" build --platform linux/amd64 -t "$ECR_URL:$TAG" "$REPO_ROOT"
"$ENGINE" push "$ECR_URL:$TAG"

echo ">> Pushed $ECR_URL:$TAG"
echo ">> If the instance booted before this push, restart its container:"
echo "   ssh ec2-user@<dns> 'docker pull $ECR_URL:$TAG && docker restart atp-dashboard'"
