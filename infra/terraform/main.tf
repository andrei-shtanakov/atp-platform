# main.tf — shared data sources and locals (ARN construction).

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Latest Amazon Linux 2023 x86_64 AMI via SSM public parameter.
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  account_id = data.aws_caller_identity.current.account_id

  # Inference-profile ARN in the source region (account-scoped resource).
  inference_profile_arn = "arn:aws:bedrock:${var.region}:${local.account_id}:inference-profile/${var.inference_profile_id}"

  # foundation-model ARN in every destination region the profile can route to.
  # These are account-agnostic (foundation-model has no account segment).
  foundation_model_arns = [
    for r in var.model_destination_regions :
    "arn:aws:bedrock:${r}::foundation-model/${var.model_id}"
  ]

  # Everything the judge (and the agent) needs to InvokeModel through the profile.
  invoke_model_resources = concat(
    [local.inference_profile_arn],
    local.foundation_model_arns,
  )

  name_prefix = "atp-bedrock-demo"
}
