# iam.tf — EC2 instance role/profile. Grants exactly:
#   - InvokeAgent on the stub agent's alias (agent under test)
#   - InvokeModel* on the EU inference profile + destination FMs (the judge)
#   - ECR pull (to fetch the prebuilt image)
# No Resource:"*" on Bedrock. No static credentials anywhere.

data "aws_iam_policy_document" "instance_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${local.name_prefix}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.instance_trust.json
}

data "aws_iam_policy_document" "instance_bedrock" {
  # Agent under test.
  statement {
    sid       = "InvokeBedrockAgentUnderTest"
    effect    = "Allow"
    actions   = ["bedrock:InvokeAgent"]
    resources = [aws_bedrockagent_agent_alias.live.agent_alias_arn]
  }

  # Judge — InvokeModel through the EU inference profile (+ all destination FMs).
  # GetInferenceProfile lets the client resolve the profile before invoking.
  statement {
    sid    = "InvokeBedrockModelForJudge"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:GetInferenceProfile",
    ]
    resources = local.invoke_model_resources
  }
}

resource "aws_iam_role_policy" "instance_bedrock" {
  name   = "bedrock"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.instance_bedrock.json
}

# ECR pull. GetAuthorizationToken is account-wide (no resource scoping possible);
# the layer/image reads are scoped to our repo.
data "aws_iam_policy_document" "instance_ecr" {
  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid    = "EcrPull"
    effect = "Allow"
    actions = [
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = [aws_ecr_repository.atp.arn]
  }
}

resource "aws_iam_role_policy" "instance_ecr" {
  name   = "ecr-pull"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.instance_ecr.json
}

# Optional but handy: SSM Session Manager so you can shell in without a keypair
# and tunnel the dashboard over SSM instead of raw SSH. Comment out if unused.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${local.name_prefix}-ec2-profile"
  role = aws_iam_role.instance.name
}
