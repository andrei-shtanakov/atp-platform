# Video Tutorials Guide

Planning and content guide for ATP video tutorials.

## Overview

This document outlines the planned video tutorial series for ATP (Agent Test Platform). These tutorials will provide hands-on guidance for getting started, advanced usage, and best practices.

**Status**: Planning Phase - Videos to be created

---

## Tutorial Series Structure

### Beginner Series (Getting Started)

#### Tutorial 1: Introduction to ATP (5 minutes)
**Target Audience**: Developers new to agent testing

**Topics**:
- What is ATP and why use it?
- Key concepts: tests, agents, assertions
- Demo: Loading a simple test suite
- Where to get help

**Demo Code**:
```python
from atp.loader import TestLoader

loader = TestLoader()
suite = loader.load_file("examples/test_suites/basic_suite.yaml")
print(f"Loaded {len(suite.tests)} tests")
```

---

#### Tutorial 2: Your First Test Suite (10 minutes)
**Target Audience**: Developers creating their first ATP test

**Topics**:
- Creating a test suite YAML file
- Defining a simple task
- Adding basic assertions
- Running the test
- Understanding results

**Demo**: Create `my_first_test.yaml`:
```yaml
test_suite: "my_first_suite"
version: "1.0"

agents:
  - name: "test-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "File creation test"
    task:
      description: "Create a file named output.txt with content 'Hello, ATP!'"
    assertions:
      - type: "artifact_exists"
        config:
          path: "output.txt"
```

---

#### Tutorial 3: Configuring Agents (8 minutes)
**Target Audience**: Developers integrating their agents with ATP

**Topics**:
- Agent adapter types (HTTP, Docker, CLI)
- Configuration basics
- Environment variables
- Testing the connection

**Demo**: Configure different adapter types:
```yaml
# HTTP adapter
agents:
  - name: "http-agent"
    type: "http"
    config:
      endpoint: "${API_ENDPOINT}"
      api_key: "${API_KEY}"

# Docker adapter
  - name: "docker-agent"
    type: "docker"
    config:
      image: "my-agent:latest"

# CLI adapter
  - name: "cli-agent"
    type: "cli"
    config:
      command: "python"
      args: ["run_agent.py"]
```

---

### Intermediate Series (Core Features)

#### Tutorial 4: Understanding Assertions (12 minutes)
**Target Audience**: Developers writing comprehensive tests

**Topics**:
- Assertion types overview
- Structural checks (artifact_exists, format)
- Content checks (contains, min_length)
- Behavior checks (tool usage, errors)
- LLM-based evaluation
- Combining assertions

**Demo**: Build layered assertions:
```yaml
assertions:
  # Layer 1: Structure
  - type: "artifact_exists"
    config:
      path: "report.md"

  # Layer 2: Content
  - type: "contains"
    config:
      artifact: "report.md"
      text: "Summary"

  # Layer 3: Quality
  - type: "llm_eval"
    config:
      artifact: "report.md"
      criteria: "completeness"
      threshold: 0.8
```

---

#### Tutorial 5: Working with Constraints (10 minutes)
**Target Audience**: Developers setting execution limits

**Topics**:
- Why constraints matter
- Types of constraints
- Setting appropriate limits
- Tool restrictions
- Budget management

**Demo**: Configure realistic constraints:
```yaml
constraints:
  max_steps: 30
  max_tokens: 50000
  timeout_seconds: 300
  allowed_tools:
    - web_search
    - file_write
  budget_usd: 1.0
```

---

#### Tutorial 6: Scoring and Evaluation (15 minutes)
**Target Audience**: Developers optimizing evaluation

**Topics**:
- Understanding scoring weights
- Quality vs efficiency tradeoffs
- Statistical reliability (multiple runs)
- Interpreting results
- Tuning thresholds

**Demo**: Configure scoring for different scenarios:
```yaml
# Quality-focused
scoring:
  quality_weight: 0.6
  completeness_weight: 0.25
  efficiency_weight: 0.1
  cost_weight: 0.05

# Cost-optimized
scoring:
  quality_weight: 0.3
  completeness_weight: 0.2
  efficiency_weight: 0.2
  cost_weight: 0.3
```

---

### Advanced Series (Power Features)

#### Tutorial 7: LLM-as-Judge Evaluation (15 minutes)
**Target Audience**: Developers using semantic evaluation

**Topics**:
- When to use LLM evaluation
- Standard criteria (completeness, accuracy, clarity)
- Custom evaluation prompts
- Calibration and thresholds
- Cost considerations

**Demo**: Create custom LLM evaluation:
```yaml
assertions:
  - type: "llm_eval"
    config:
      artifact: "report.md"
      criteria: "custom"
      prompt: |
        Evaluate if the competitor analysis includes:
        1. Market share data (weight: 40%)
        2. Key features comparison (weight: 30%)
        3. Pricing information (weight: 20%)
        4. SWOT analysis (weight: 10%)

        Provide score 0-1 based on weighted criteria.
      threshold: 0.75
```

---

#### Tutorial 8: Testing Multiple Agents (12 minutes)
**Target Audience**: Developers comparing agent implementations

**Topics**:
- Configuring multiple agents
- Running comparative tests
- Analyzing differences
- A/B testing strategies
- Regression detection

**Demo**: Compare agent versions:
```yaml
agents:
  - name: "agent-v1"
    type: "http"
    config:
      endpoint: "${V1_ENDPOINT}"

  - name: "agent-v2"
    type: "http"
    config:
      endpoint: "${V2_ENDPOINT}"

tests:
  - id: "comparison-test"
    name: "Version comparison"
    # Same test runs on both agents
```

---

#### Tutorial 9: Custom Evaluators (18 minutes)
**Target Audience**: Advanced developers with domain-specific needs

**Topics**:
- When to create custom evaluators
- Evaluator interface
- Implementing evaluation logic
- Registering custom evaluators
- Testing evaluators

**Demo**: Create domain-specific evaluator:
```python
from atp.evaluators.base import Evaluator, EvalResult

class FinancialAccuracyEvaluator(Evaluator):
    name = "financial_accuracy"

    async def evaluate(self, task, response, trace, assertion):
        artifact = self._get_artifact(response, assertion.config["artifact"])

        # Custom validation
        accuracy = await self._validate_financial_data(artifact)

        return EvalResult(
            evaluator=self.name,
            checks=[{
                "name": "financial_accuracy",
                "passed": accuracy >= assertion.config.get("threshold", 0.9),
                "score": accuracy
            }]
        )
```

---

#### Tutorial 10: CI/CD Integration (20 minutes)
**Target Audience**: DevOps engineers and developers

**Topics**:
- Test suite organization
- GitHub Actions setup
- GitLab CI configuration
- Jenkins integration
- Reporting and notifications
- Handling secrets

**Demo**: GitHub Actions workflow:
```yaml
name: Agent Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.12"

      - name: Install ATP
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          uv sync

      - name: Run Smoke Tests
        run: atp run tests/smoke_tests.yaml

      - name: Run Full Suite
        if: github.event_name == 'pull_request'
        run: atp run tests/regression_suite.yaml
```

---

### Expert Series (Advanced Topics)

#### Tutorial 11: Performance Optimization (15 minutes)
**Target Audience**: Developers optimizing test execution

**Topics**:
- Identifying bottlenecks
- Parallel test execution
- Caching strategies
- Optimizing LLM costs
- Test suite organization

---

#### Tutorial 12: Statistical Analysis (18 minutes)
**Target Audience**: Developers analyzing test reliability

**Topics**:
- Understanding variance
- Confidence intervals
- Stability assessment
- Determining optimal run count
- Handling outliers

---

#### Tutorial 13: Security Testing (20 minutes)
**Target Audience**: Security-focused developers

**Topics**:
- Testing prompt injection resistance
- Validating data sanitization
- Testing access controls
- Secret handling validation
- Security assertion patterns

---

## Tutorial Format Standards

### Video Structure

Each tutorial should follow this structure:

1. **Introduction** (30 seconds)
   - What you'll learn
   - Prerequisites
   - Resources needed

2. **Concept Overview** (2-3 minutes)
   - Key concepts
   - Why it matters
   - Use cases

3. **Hands-on Demo** (5-10 minutes)
   - Step-by-step walkthrough
   - Real code examples
   - Common pitfalls

4. **Results and Analysis** (2-3 minutes)
   - Understanding output
   - Interpreting results
   - Troubleshooting

5. **Summary** (30 seconds)
   - Key takeaways
   - Next steps
   - Additional resources

### Technical Requirements

**Video Quality**:
- Resolution: 1920x1080 minimum
- Frame rate: 30fps
- Audio: Clear narration, no background noise
- Screen recording: High DPI, readable text

**Code Display**:
- Font: Monospace, 14pt minimum
- Theme: High contrast (light or dark)
- Syntax highlighting: Enabled
- Terminal: Readable font size

**Editing**:
- Trim dead air and mistakes
- Add captions/subtitles
- Include chapter markers
- Zoom on important details

---

## Support Materials

Each tutorial should include:

### Code Repository
```
tutorials/
├── 01-introduction/
│   └── demo.py
├── 02-first-test/
│   ├── my_first_test.yaml
│   └── agent_server.py
├── 03-configure-agents/
│   ├── http_agent.yaml
│   ├── docker_agent.yaml
│   └── cli_agent.yaml
...
```

### Written Companion Guide
- Transcript or detailed notes
- Code snippets
- Links to documentation
- Practice exercises

### Practice Exercises
- Beginner: Follow-along exercises
- Intermediate: Modified scenarios
- Advanced: Challenge problems

---

## Tutorial Publishing Plan

### Phase 1: Foundation (Tutorials 1-3)
**Timeline**: Month 1
- Focus on getting started
- Core concepts
- First test suite

### Phase 2: Core Skills (Tutorials 4-6)
**Timeline**: Month 2
- Assertions
- Constraints
- Scoring

### Phase 3: Advanced Features (Tutorials 7-10)
**Timeline**: Month 3
- LLM evaluation
- Multiple agents
- Custom evaluators
- CI/CD

### Phase 4: Expert Topics (Tutorials 11-13)
**Timeline**: Month 4
- Performance
- Statistics
- Security

---

## Distribution Channels

### Primary Channels
- **YouTube**: Main hosting platform
- **Documentation Site**: Embedded videos
- **GitHub**: Code repositories

### Promotional Channels
- **Twitter/X**: Tutorial announcements
- **LinkedIn**: Professional audience
- **Reddit**: r/MachineLearning, r/LangChain
- **Discord/Slack**: Community channels

---

## Tutorial Playlist Organization

### YouTube Playlists

**ATP Getting Started**:
- Tutorial 1: Introduction to ATP
- Tutorial 2: Your First Test Suite
- Tutorial 3: Configuring Agents

**ATP Core Features**:
- Tutorial 4: Understanding Assertions
- Tutorial 5: Working with Constraints
- Tutorial 6: Scoring and Evaluation

**ATP Advanced Topics**:
- Tutorial 7: LLM-as-Judge Evaluation
- Tutorial 8: Testing Multiple Agents
- Tutorial 9: Custom Evaluators
- Tutorial 10: CI/CD Integration

**ATP Expert Series**:
- Tutorial 11: Performance Optimization
- Tutorial 12: Statistical Analysis
- Tutorial 13: Security Testing

---

## Metrics and Success Criteria

Track tutorial effectiveness:

**Engagement Metrics**:
- View count and watch time
- Likes and comments
- Subscriber growth

**Educational Metrics**:
- GitHub repository stars/forks
- Documentation page views
- Support question trends

**Success Criteria**:
- 80% average watch completion
- Positive feedback ratio >90%
- Decrease in repetitive support questions

---

## Maintenance Plan

### Update Schedule
- **Monthly**: Check for outdated information
- **Quarterly**: Update code examples
- **Major releases**: Record updated tutorials

### Version Compatibility
- Note ATP version in video description
- Create "Version Updates" supplementary videos
- Maintain written update log

---

## Community Contributions

Encourage community tutorials:

**Guidelines for Contributors**:
- Follow format standards
- Submit for review
- Include code repository
- Add to community playlist

**Featured Community Topics**:
- Framework-specific integrations
- Industry-specific use cases
- Advanced customization
- Tool integrations

---

## Resources for Tutorial Creators

### Tools
- **Screen Recording**: OBS Studio, Camtasia
- **Video Editing**: DaVinci Resolve, Adobe Premiere
- **Thumbnail Creation**: Canva, Figma
- **Code Highlighting**: Carbon, highlight.js

### Templates
- Video description template
- Thumbnail template
- Code repository structure
- Practice exercise template

---

## Feedback and Iteration

### Collecting Feedback
- YouTube comments
- GitHub issues
- Community surveys
- Direct user feedback

### Improvement Process
1. Review feedback monthly
2. Identify common issues
3. Plan updates or new tutorials
4. Implement improvements
5. Announce updates

---

## Quick Reference Card

After watching tutorials, users should have:

### Beginner
- Can load and inspect test suites
- Can create basic test YAML
- Can configure simple agents
- Can run tests and view results

### Intermediate
- Can write comprehensive assertions
- Can set appropriate constraints
- Can configure scoring weights
- Can interpret statistical results

### Advanced
- Can create custom evaluators
- Can compare multiple agents
- Can integrate with CI/CD
- Can optimize performance

### Expert
- Can analyze statistical reliability
- Can implement security tests
- Can extend ATP for custom needs
- Can contribute to ATP project

---

## Next Steps

1. **Script Development**: Write detailed scripts for each tutorial
2. **Demo Preparation**: Create demo environments and code
3. **Recording Setup**: Set up recording environment
4. **Pilot Production**: Create first 3 tutorials
5. **Community Feedback**: Gather initial feedback
6. **Full Production**: Complete entire series

---

## Contact for Tutorial Development

- **Documentation Questions**: See [docs/](.)
- **Code Examples**: See [examples/](../../examples/)
- **Contribution Guide**: See [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **Issues/Suggestions**: [GitHub Issues](https://github.com/yourusername/atp-platform/issues)

---

## See Also

- [Quick Start Guide](quickstart.md) - Written getting started guide
- [Usage Guide](usage.md) - Common workflows
- [Best Practices](best-practices.md) - Testing best practices
- [API Reference](../reference/api-reference.md) - Complete API documentation
