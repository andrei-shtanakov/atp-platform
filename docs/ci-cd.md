# CI/CD Integration Guide

This guide covers integrating ATP (Agent Test Platform) with continuous integration and deployment systems.

## Exit Codes

ATP uses standard exit codes to indicate test results, making it easy to integrate with CI/CD systems:

| Exit Code | Constant | Description |
|-----------|----------|-------------|
| `0` | `EXIT_SUCCESS` | All tests passed successfully |
| `1` | `EXIT_FAILURE` | One or more tests failed |
| `2` | `EXIT_ERROR` | Error occurred (invalid config, missing file, etc.) |

### Exit Code Details

#### Exit Code 0 - Success
All tests in the suite passed. This includes:
- All assertions evaluated to true
- All evaluators returned passing scores
- No errors occurred during execution

#### Exit Code 1 - Test Failure
One or more tests did not pass. This occurs when:
- Test assertions fail (artifact not found, content mismatch, etc.)
- Evaluator scores fall below thresholds
- Agent returns an error status
- No tests match the specified criteria

#### Exit Code 2 - Error
An error prevented test execution. Common causes:
- Test suite file not found or invalid YAML
- Configuration file parse errors
- Adapter initialization failures
- Invalid command-line arguments
- Network connectivity issues

### Using Exit Codes in CI

```bash
# Basic usage - fail pipeline if any tests fail
atp test suite.yaml
if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi

# More detailed handling
atp test suite.yaml
EXIT_CODE=$?

case $EXIT_CODE in
    0)
        echo "All tests passed"
        ;;
    1)
        echo "Test failures detected"
        exit 1
        ;;
    2)
        echo "Configuration or execution error"
        exit 2
        ;;
esac
```

## Output Formats

ATP supports multiple output formats for CI/CD integration:

### JSON Output

Machine-readable format for automation and custom processing:

```bash
atp test suite.yaml --output=json --output-file=results.json
```

JSON structure:
```json
{
  "version": "1.0",
  "generated_at": "2024-01-15T10:30:00",
  "summary": {
    "suite_name": "my-tests",
    "agent_name": "my-agent",
    "total_tests": 10,
    "passed_tests": 8,
    "failed_tests": 2,
    "success_rate": 0.8,
    "success": false
  },
  "tests": [...]
}
```

### JUnit XML Output

Compatible with most CI systems (GitHub Actions, GitLab CI, Jenkins, Azure DevOps):

```bash
atp test suite.yaml --output=junit --output-file=junit.xml
```

This format enables:
- Test result visualization in CI dashboards
- Automatic annotations on pull requests
- Historical test tracking
- Flaky test detection

### HTML Output

Human-readable reports for sharing:

```bash
atp test suite.yaml --output=html --output-file=report.html
```

## GitHub Actions Integration

### Quick Start

Add this workflow to `.github/workflows/atp-test.yml`:

```yaml
name: ATP Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install ATP
        run: |
          pip install uv
          uv sync

      - name: Run tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          atp test tests/suite.yaml --output=junit --output-file=results/junit.xml

      - name: Publish Test Results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: results/junit.xml
```

### Full Template

See `.github/workflows/atp-test.yml` for a complete template with:
- Manual trigger with inputs
- Baseline comparison on PRs
- Test result annotations
- Artifact upload

## GitLab CI Integration

### Quick Start

Add this to your `.gitlab-ci.yml`:

```yaml
test:atp:
  image: python:3.12
  script:
    - pip install uv && uv sync
    - atp test tests/suite.yaml --output=junit --output-file=junit.xml
  artifacts:
    reports:
      junit: junit.xml
```

### Full Template

See `ci-templates/gitlab-ci.yml` for a complete template with:
- Suite validation stage
- Smoke tests on every push
- Full tests on merge requests
- Baseline comparison
- Nightly comprehensive tests
- GitLab Pages HTML reports

## Jenkins Integration

### Pipeline Example

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.12'
        }
    }

    environment {
        ANTHROPIC_API_KEY = credentials('anthropic-api-key')
    }

    stages {
        stage('Setup') {
            steps {
                sh 'pip install uv && uv sync'
            }
        }

        stage('Test') {
            steps {
                sh '''
                    atp test tests/suite.yaml \
                        --output=junit \
                        --output-file=results/junit.xml
                '''
            }
        }
    }

    post {
        always {
            junit 'results/junit.xml'
            archiveArtifacts artifacts: 'results/**', fingerprint: true
        }
    }
}
```

## Azure DevOps Integration

### Pipeline Example

```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.12'

  - script: |
      pip install uv
      uv sync
    displayName: 'Install dependencies'

  - script: |
      atp test tests/suite.yaml --output=junit --output-file=$(Build.ArtifactStagingDirectory)/junit.xml
    displayName: 'Run ATP tests'
    env:
      ANTHROPIC_API_KEY: $(ANTHROPIC_API_KEY)

  - task: PublishTestResults@2
    inputs:
      testResultsFormat: 'JUnit'
      testResultsFiles: '$(Build.ArtifactStagingDirectory)/junit.xml'
      failTaskOnFailedTests: true
```

## CircleCI Integration

### Config Example

```yaml
version: 2.1

jobs:
  test:
    docker:
      - image: cimg/python:3.12
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: |
            pip install uv
            uv sync
      - run:
          name: Run ATP tests
          command: |
            mkdir -p test-results
            atp test tests/suite.yaml --output=junit --output-file=test-results/junit.xml
      - store_test_results:
          path: test-results

workflows:
  main:
    jobs:
      - test
```

## Best Practices

### 1. Use Multiple Runs for Statistical Significance

For LLM-based agents, single runs can be misleading due to non-determinism:

```bash
# Production CI - run each test 3-5 times
atp test suite.yaml --runs=3

# Nightly/weekly - more comprehensive
atp test suite.yaml --runs=10
```

### 2. Separate Smoke and Full Tests

```bash
# On every commit - quick smoke tests
atp test suite.yaml --tags=smoke --runs=1

# On PRs - full test suite
atp test suite.yaml --runs=3

# Nightly - comprehensive with statistics
atp test suite.yaml --runs=5
```

### 3. Use Baselines for Regression Detection

```bash
# Save baseline (run from main branch)
atp baseline save suite.yaml -o baselines/baseline.json --runs=5

# Compare in CI
atp baseline compare suite.yaml -b baselines/baseline.json --fail-on-regression
```

### 4. Parallel Execution for Speed

```bash
# Run tests in parallel (4 concurrent)
atp test suite.yaml --parallel=4

# Adjust based on CI runner resources
atp test suite.yaml --parallel=$(nproc)
```

### 5. Fail Fast for Quick Feedback

During development:
```bash
atp test suite.yaml --fail-fast
```

In CI, let all tests complete:
```bash
atp test suite.yaml  # No --fail-fast
```

### 6. Protect API Keys

Always use CI secrets for API keys:

```yaml
# GitHub Actions
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

# GitLab CI - use CI/CD variables (masked)
```

### 7. Cache Dependencies

```yaml
# GitHub Actions
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}

# GitLab CI - use cache directive
```

## Troubleshooting

### Exit Code 2 in CI

Check:
1. Test suite file exists and path is correct
2. Configuration file is valid YAML
3. Required environment variables are set
4. Network access to agent endpoints

### JUnit XML Not Found

Ensure output directory exists:
```bash
mkdir -p results
atp test suite.yaml --output=junit --output-file=results/junit.xml
```

### Tests Timeout in CI

Increase timeout limits:
```yaml
# GitHub Actions
timeout-minutes: 60

# GitLab CI
test:
  timeout: 1 hour
```

### LLM Evaluator Failures

Ensure API keys are available:
```bash
# Verify key is set
echo "ANTHROPIC_API_KEY is ${ANTHROPIC_API_KEY:+set}"

# Use fallback evaluators
atp test suite.yaml --tags=!llm_eval
```
