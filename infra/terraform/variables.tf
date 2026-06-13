# variables.tf — all knobs. Defaults target the agreed demo:
# eu-central-1 + Claude Sonnet 4.5 via the EU cross-region inference profile.

variable "region" {
  description = "Source AWS region (where the instance and requests originate)."
  type        = string
  default     = "eu-central-1"
}

variable "owner_tag" {
  description = "Free-form owner tag applied to all resources."
  type        = string
  default     = "andrei"
}

# --- Model / judge -----------------------------------------------------------

variable "model_id" {
  description = "Bare foundation-model id (used to build destination-region ARNs)."
  type        = string
  default     = "anthropic.claude-sonnet-4-5-20250929-v1:0"
}

variable "inference_profile_id" {
  description = <<-EOT
    EU geographic cross-Region inference profile id. Sonnet 4.5 in EU is NOT
    callable on-demand — it must go through this profile. Used as both the
    judge model (ATP_JUDGE_MODEL) and the Bedrock Agent's foundationModel.
  EOT
  type        = string
  default     = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
}

variable "model_destination_regions" {
  description = <<-EOT
    All regions the EU inference profile may route to. The IAM policy must allow
    InvokeModel on the foundation-model ARN in EVERY one of these, or you get
    AccessDenied mid-run. Source list per AWS EU geographic profile.
  EOT
  type        = list(string)
  default = [
    "eu-central-1",
    "eu-west-1",
    "eu-west-3",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
  ]
}

# --- Compute -----------------------------------------------------------------

variable "instance_type" {
  description = "EC2 type. t3.small is enough since we PULL the image (no on-box build)."
  type        = string
  default     = "t3.small"
}

variable "root_volume_gb" {
  description = "Root EBS size. >=20 GB to hold the image + Docker layers."
  type        = number
  default     = 30
}

variable "enable_ssh" {
  description = <<-EOT
    When true (default), open port 22 to ssh_ingress_cidr and attach key_pair_name
    — the classic SSH-tunnel flow. Set false for **SSM-only**: zero inbound ports,
    no key pair; shell/port-forward via `aws ssm start-session` (the instance role
    already carries AmazonSSMManagedInstanceCore). key_pair_name/ssh_ingress_cidr
    are then unused.
  EOT
  type        = bool
  default     = true
}

variable "key_pair_name" {
  description = "Existing EC2 key pair name for SSH. Required when enable_ssh=true; leave null for SSM-only."
  type        = string
  default     = null
}

variable "ssh_ingress_cidr" {
  description = <<-EOT
    CIDR allowed to SSH (port 22). Required when enable_ssh=true — set to YOUR.IP/32,
    never 0.0.0.0/0. Unused (and no port is opened) when enable_ssh=false. The
    dashboard is always reached via tunnel (SSH or SSM port-forward) — never 8080.
  EOT
  type        = string
  default     = null
}

variable "image_tag" {
  description = "ECR image tag the instance pulls and runs."
  type        = string
  default     = "latest"
}

variable "source_version" {
  description = <<-EOT
    Bump this (e.g. to `git rev-parse --short HEAD`) to force `terraform apply` to
    rebuild & re-push the image after PROJECT CODE changed. Terraform cannot detect
    arbitrary source edits — without a change to this (or image_tag / the ECR URL)
    the build is NOT re-triggered, and the old image stays in ECR.
    NOTE: a rebuild re-pushes the tag but does NOT update an already-running
    instance — re-pull + restart the container on the box (build-and-push.sh prints
    the line) or recreate the instance (`terraform taint aws_instance.atp`).
  EOT
  type        = string
  default     = "v1"
}

variable "build_push_on_apply" {
  description = <<-EOT
    When true (default), `terraform apply` builds and pushes the image to ECR
    BEFORE creating the EC2 instance (via a local-exec on terraform_data), so the
    box never boots to an empty registry. Requires docker + AWS creds on the
    machine running terraform. Set false for CI / split flows that push the image
    out of band — then ensure the image exists before the instance boots.
  EOT
  type        = bool
  default     = true
}

# --- Budget ------------------------------------------------------------------

variable "monthly_budget_usd" {
  description = "Monthly budget alert threshold. NOTE: alerts only, not a hard cap."
  type        = number
  default     = 5
}

variable "budget_email" {
  description = "Email for budget alert notifications."
  type        = string
}
