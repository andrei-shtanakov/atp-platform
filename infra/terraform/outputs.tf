# outputs.tf — single source of truth for the run/preflight scripts.

output "region" {
  value = var.region
}

output "ecr_repo_url" {
  description = "Push the image here; the instance pulls from it."
  value       = aws_ecr_repository.atp.repository_url
}

output "agent_id" {
  description = "Pass to --adapter-config agent_id=..."
  value       = aws_bedrockagent_agent.stub.agent_id
}

output "agent_alias_id" {
  description = "Pass to --adapter-config agent_alias_id=..."
  value       = aws_bedrockagent_agent_alias.live.agent_alias_id
}

output "judge_model" {
  description = "ATP_JUDGE_MODEL — the EU inference profile id."
  value       = var.inference_profile_id
}

output "instance_id" {
  value = aws_instance.atp.id
}

output "instance_public_dns" {
  description = "SSH target: ssh -L 8080:localhost:8080 ec2-user@<this>"
  value       = aws_instance.atp.public_dns
}

# Convenience block printed after apply.
output "next_steps" {
  value = <<-EOT

    (you are in infra/terraform/ after `terraform apply` — scripts live in ../scripts/)
    The image was already built & pushed during apply (build_push_on_apply=true).
    1. Preflight (free checks):  ../scripts/preflight.sh   # add --invoke for a ~1-token paid check
    2. SSH tunnel to dashboard:  ssh -L 8080:localhost:8080 ec2-user@${aws_instance.atp.public_dns}
    3. Run the sweep on the box: ../scripts/run-sweep.sh <ssh-key.pem> [RUNS]
    4. Tear down:                terraform destroy

    Optional — after changing project code, re-push and refresh the box:
      ../scripts/build-and-push.sh        # rebuild + push (or: apply -var source_version=<sha>)
      then re-pull/restart on the box, or: terraform taint aws_instance.atp && terraform apply
  EOT
}
