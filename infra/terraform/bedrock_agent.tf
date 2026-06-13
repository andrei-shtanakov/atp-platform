# bedrock_agent.tf — a trivial "agent under test" stub + its execution role + alias.
#
# This is the methodology stub: a single-prompt Bedrock Agent, no action groups,
# no knowledge base. It exists only so ATP has a real agent-alias to invoke via
# --adapter=bedrock. Swap foundationModel/instruction later for a real agent.

# --- Agent execution role ----------------------------------------------------
# Bedrock REQUIRES this role's NAME to start with "AmazonBedrockExecutionRoleForAgents_".
data "aws_iam_policy_document" "agent_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }
    # Scope the trust to this account's agents only.
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:bedrock:${var.region}:${local.account_id}:agent/*"]
    }
  }
}

resource "aws_iam_role" "agent_exec" {
  # name_prefix (not a fixed name) so parallel environments in one account don't
  # collide. Terraform appends a unique suffix; the result still starts with the
  # Bedrock-required "AmazonBedrockExecutionRoleForAgents_" prefix (36 chars, well
  # under the 64-char role-name limit). Nothing references the name literally.
  name_prefix        = "AmazonBedrockExecutionRoleForAgents_"
  assume_role_policy = data.aws_iam_policy_document.agent_trust.json
}

data "aws_iam_policy_document" "agent_invoke_model" {
  # The agent orchestrates through the EU inference profile, so its execution
  # role needs InvokeModel on the profile ARN *and* every destination FM, plus
  # GetInferenceProfile to resolve the profile at prepare/invoke time. Missing
  # GetInferenceProfile is a common cause of a failed agent PREPARE.
  statement {
    sid    = "AgentInvokeModelViaEUProfile"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:GetInferenceProfile",
    ]
    resources = local.invoke_model_resources
  }
}

resource "aws_iam_role_policy" "agent_invoke_model" {
  name   = "invoke-model"
  role   = aws_iam_role.agent_exec.id
  policy = data.aws_iam_policy_document.agent_invoke_model.json
}

# --- The agent + alias -------------------------------------------------------
resource "aws_bedrockagent_agent" "stub" {
  agent_name              = "${local.name_prefix}-stub"
  agent_resource_role_arn = aws_iam_role.agent_exec.arn

  # Sonnet 4.5 in EU must run through the inference profile, not the bare model id.
  # foundationModel accepts the inference-profile *id* (eu./us. prefix) or its ARN;
  # the id is fine. What it needs is the IAM above (InvokeModel on profile + every
  # destination FM + GetInferenceProfile) — that, not the value format, is the gate.
  foundation_model = var.inference_profile_id

  # Instruction must be >= 40 chars. Trivial methodology stub.
  instruction = "You are a requirements-extraction assistant under evaluation. Read the user's text and extract the explicit functional requirements faithfully, without inventing any not stated."

  idle_session_ttl_in_seconds = 600
  prepare_agent               = true
}

resource "aws_bedrockagent_agent_alias" "live" {
  agent_alias_name = "live"
  agent_id         = aws_bedrockagent_agent.stub.agent_id
  description      = "Alias ATP invokes via --adapter=bedrock"
}
