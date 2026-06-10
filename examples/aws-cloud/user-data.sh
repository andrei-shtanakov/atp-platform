#!/bin/bash
# EC2 user-data (Amazon Linux 2023): bootstrap the ATP platform image and start
# the dashboard. The instance must have an IAM role with iam-policy.json attached
# so the Bedrock adapter and the Bedrock-Claude judge work with no static keys.
#
# After boot, run a sweep from the instance — see README.md.
set -euxo pipefail

dnf install -y docker git
systemctl enable --now docker

cd /opt
git clone https://github.com/andrei-shtanakov/atp-platform.git
cd atp-platform

# Build the platform image (installs all workspace members incl. the atp-method
# plugin and the LLM-judge clients — see Dockerfile).
docker build -t atp-platform:latest .

# Optional: run the dashboard so runs can be browsed at http://<instance>:8080/ui/
docker run -d --restart unless-stopped --name atp-dashboard \
  -p 8080:8080 \
  -e ATP_SECRET_KEY="$(openssl rand -hex 16)" \
  -e ATP_DISABLE_AUTH=true \
  -v atp-data:/root/.atp \
  atp-platform:latest \
  uv run --no-sync atp dashboard --host 0.0.0.0 --port 8080
