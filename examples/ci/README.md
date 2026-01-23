# CI/CD Examples

This directory contains example configurations for integrating ATP with various CI/CD systems.

## Quick Start

Choose the appropriate file for your CI system:

| CI System | File | Copy To |
|-----------|------|---------|
| GitHub Actions | `github-actions-basic.yml` | `.github/workflows/atp.yml` |
| GitLab CI | `gitlab-ci-basic.yml` | `.gitlab-ci.yml` |
| Jenkins | `Jenkinsfile` | `Jenkinsfile` |
| Azure DevOps | `azure-pipelines.yml` | `azure-pipelines.yml` |
| CircleCI | `circleci-config.yml` | `.circleci/config.yml` |

## Common Steps

All examples follow these steps:

1. **Set up Python** - Install Python 3.12+
2. **Install dependencies** - Use `uv sync` to install project dependencies
3. **Run ATP tests** - Execute `atp test` with JUnit output
4. **Publish results** - Upload JUnit XML for test visualization

## Configuration

### Required Secrets

If using LLM-as-judge evaluations, configure these secrets:

| Secret | Description |
|--------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `OPENAI_API_KEY` | OpenAI API key (alternative) |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ATP_SUITE` | `tests/suite.yaml` | Test suite file path |
| `ATP_RUNS` | `1` | Runs per test |
| `ATP_TAGS` | (none) | Filter tests by tags |
| `ATP_PARALLEL` | `4` | Parallel test execution |

## Advanced Features

For more advanced setups including:
- Baseline comparison
- Multiple test stages (smoke, full, nightly)
- HTML report generation
- Manual triggers

See the full templates:
- GitHub Actions: `../.github/workflows/atp-test.yml`
- GitLab CI: `../ci-templates/gitlab-ci.yml`

## Documentation

For complete CI/CD documentation, see [docs/ci-cd.md](../../docs/ci-cd.md).
