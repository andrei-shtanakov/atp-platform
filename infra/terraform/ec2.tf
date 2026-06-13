# ec2.tf — default-VPC instance with the IAM profile, IMDSv2 hop-limit=2,
# and a security group that opens ONLY SSH. The dashboard (8080) is never exposed;
# reach it via SSH tunnel or SSM port-forward.

data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "instance" {
  name_prefix = "${local.name_prefix}-"
  description = "ATP demo: optional SSH in, all out. NEVER 8080."
  vpc_id      = data.aws_vpc.default.id

  # SSH only when enable_ssh=true AND a CIDR is set. SSM-only mode opens ZERO
  # inbound ports. Guarding on the non-null CIDR avoids a cidr_blocks=[null]
  # error masking the clearer aws_instance precondition below.
  dynamic "ingress" {
    for_each = var.enable_ssh && var.ssh_ingress_cidr != null ? [1] : []
    content {
      description = "SSH from operator IP only"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [var.ssh_ingress_cidr]
    }
  }

  egress {
    description = "All outbound (Bedrock, ECR, package mirrors)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_instance" "atp" {
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  iam_instance_profile   = aws_iam_instance_profile.instance.name
  vpc_security_group_ids = [aws_security_group.instance.id]

  # CRITICAL: hop limit 2 so the Docker container can read the instance role
  # from IMDSv2. With the default hop limit of 1, boto3 inside the container
  # gets no credentials and every Bedrock call fails.
  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 2
  }

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/../user-data.sh.tftpl", {
    aws_region   = var.region
    ecr_repo_url = aws_ecr_repository.atp.repository_url
    image_tag    = var.image_tag
  })

  tags = { Name = "${local.name_prefix}-runner" }

  lifecycle {
    # SSH mode needs both a key pair and an ingress CIDR; SSM-only needs neither.
    precondition {
      condition     = !var.enable_ssh || (var.key_pair_name != null && var.ssh_ingress_cidr != null)
      error_message = "enable_ssh=true requires key_pair_name and ssh_ingress_cidr. Set both, or use enable_ssh=false for SSM-only."
    }
  }

  # Image must exist in ECR before the instance boots and pulls it. When
  # build_push_on_apply=true, terraform_data.image_push builds+pushes first and
  # this dependency enforces the ordering; when false, the repo alone is the dep
  # (you must push out of band before the box pulls — user-data retries cover a
  # short gap).
  depends_on = [aws_ecr_repository.atp, terraform_data.image_push]
}
