# image.tf — build & push the ATP image to ECR DURING `terraform apply`, before
# the EC2 instance is created. This removes the race where the box boots and
# tries to pull an image that hasn't been pushed yet.
#
# terraform_data is built-in (TF >= 1.4) — no extra provider. The local-exec
# reuses scripts/build-and-push.sh, passing REGION/ECR_URL via env so the script
# does NOT depend on `terraform output` (not finalized mid-apply).
#
# Toggle off with build_push_on_apply=false for CI / out-of-band push flows.

resource "terraform_data" "image_push" {
  count = var.build_push_on_apply ? 1 : 0

  # Re-run when the tag, the repo, or the declared source version changes.
  # Terraform can't see arbitrary code edits, so bump var.source_version (e.g. to
  # a git SHA) to force a rebuild after changing project code.
  triggers_replace = [
    var.image_tag,
    var.source_version,
    aws_ecr_repository.atp.repository_url,
  ]

  provisioner "local-exec" {
    command = "${path.module}/../scripts/build-and-push.sh ${var.image_tag}"
    environment = {
      REGION  = var.region
      ECR_URL = aws_ecr_repository.atp.repository_url
    }
  }
}
