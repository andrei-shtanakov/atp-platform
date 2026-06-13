# ecr.tf — private registry for the prebuilt ATP image (pull on the instance,
# instead of building on a small box).

resource "aws_ecr_repository" "atp" {
  name                 = "${local.name_prefix}/atp-platform"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # demo: allow `terraform destroy` to remove images too

  image_scanning_configuration {
    scan_on_push = false
  }
}

# Keep only the few most recent images so the repo doesn't accrue storage cost.
resource "aws_ecr_lifecycle_policy" "atp" {
  repository = aws_ecr_repository.atp.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 3 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}
