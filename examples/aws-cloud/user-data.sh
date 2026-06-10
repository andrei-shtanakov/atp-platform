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

# Optional: run the dashboard, bound to localhost only. Reach it via an SSH
# tunnel (`ssh -L 8080:localhost:8080 ec2-user@<instance>`), NOT a public port —
# do not expose 8080 in the security group, and do not disable auth on a public
# bind. Auth is left enabled; the first user to register becomes admin.
docker run -d --restart unless-stopped --name atp-dashboard \
  -p 127.0.0.1:8080:8080 \
  -e ATP_SECRET_KEY="$(openssl rand -hex 32)" \
  -v atp-data:/root/.atp \
  atp-platform:latest \
  uv run --no-sync atp dashboard --host 0.0.0.0 --port 8080
