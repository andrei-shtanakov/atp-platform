# budget.tf — a $5 monthly cost alert.
# IMPORTANT: AWS Budgets ALERT, they do NOT hard-stop spend. To actually cap a
# runaway, keep --runs small and `terraform destroy` after the demo. A true hard
# stop needs a Budgets action + automation (out of scope for a $5 demo).

resource "aws_budgets_budget" "demo" {
  name         = "${local.name_prefix}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
