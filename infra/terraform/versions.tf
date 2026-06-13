# versions.tf — provider + Terraform version pins
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60" # aws_bedrockagent_* resources require >= 5.40
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project   = "atp-bedrock-demo"
      ManagedBy = "terraform"
      Owner     = var.owner_tag
    }
  }
}
