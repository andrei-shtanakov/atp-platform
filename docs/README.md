# ATP Documentation

Complete documentation for the Agent Test Platform.

## Overview

ATP (Agent Test Platform) is a framework-agnostic platform for testing and evaluating AI agents. This documentation provides comprehensive guides, references, and examples to help you get the most out of ATP.

**Current Version**: v1.0.0 (GA - All features complete)

---

## Quick Links

- **[Installation Guide](guides/installation.md)** - Get ATP installed and running
- **[Quick Start](guides/quickstart.md)** - Create your first test in 5 minutes
- **[FAQ](reference/faq.md)** - Common questions and answers
- **[API Reference](reference/api-reference.md)** - Complete Python API documentation

---

## Documentation Structure

### 🚀 Getting Started

Start here if you're new to ATP:

1. **[Installation Guide](guides/installation.md)**
   - System requirements
   - Installation steps
   - Verification
   - Troubleshooting installation issues

2. **[Quick Start Guide](guides/quickstart.md)**
   - Your first test suite in 5 minutes
   - Basic concepts
   - First test execution
   - Understanding results

3. **[Usage Guide](guides/usage.md)**
   - Common workflows
   - Working with test suites
   - Configuring agents
   - Running tests
   - Analyzing results

### 📚 Core Concepts

Understand ATP's architecture and design:

1. **[Vision & Goals](01-vision.md)**
   - Project motivation
   - Key principles
   - Target audience
   - Success criteria

2. **[Requirements](02-requirements.md)**
   - Functional requirements
   - Non-functional requirements
   - User stories
   - Use cases

3. **[Architecture](03-architecture.md)**
   - System overview
   - Component design
   - Data flow
   - Technology stack
   - Design patterns

4. **[ATP Protocol](04-protocol.md)**
   - Protocol specification
   - Request/response format
   - Event streaming
   - Error handling
   - Examples

5. **[Evaluation System](05-evaluators.md)**
   - Evaluation philosophy
   - Evaluator types
   - Scoring system
   - Statistical analysis
   - Custom evaluators

6. **[Integration Guide](06-integration.md)**
   - Agent integration patterns
   - Adapter development
   - Framework-specific guides
   - Custom implementations

7. **[Roadmap](07-roadmap.md)**
   - Development phases
   - Current status
   - Upcoming features
   - Long-term vision

### 📖 Reference Documentation

Complete reference for ATP features:

1. **[API Reference](reference/api-reference.md)**
   - `atp.loader` module
   - Data models
   - Exceptions
   - Complete Python API
   - Usage examples

2. **[Dashboard API Reference](reference/dashboard-api.md)**
   - REST API endpoints
   - Agent comparison (side-by-side, general)
   - Leaderboard matrix
   - Timeline events and comparison
   - Authentication
   - Python client examples

3. **[Marketplace API Reference](reference/marketplace-api.md)**
   - Test suite publishing and discovery
   - Search and filtering
   - Ratings and reviews
   - Version management
   - GitHub import

4. **[Public Leaderboard API Reference](reference/public-leaderboard-api.md)**
   - Agent performance rankings
   - Benchmark categories
   - Agent profiles
   - Result publishing
   - Historical trends

5. **[Dashboard v2 Migration Guide](reference/dashboard-migration.md)**
   - v1 to v2 architecture changes
   - Route migration patterns
   - Service layer patterns
   - Dependency injection

6. **[Configuration Reference](reference/configuration.md)**
   - YAML structure
   - Configuration options
   - Environment variables
   - Default values
   - Complete examples

7. **[Test Format Reference](reference/test-format.md)**
   - Test suite structure
   - Task definition
   - Constraints
   - Assertions
   - Scoring weights
   - Validation rules

8. **[Adapter Configuration](reference/adapters.md)**
   - HTTP adapter
   - Docker adapter
   - CLI adapter
   - LangGraph adapter
   - CrewAI adapter
   - Custom adapters

9. **[Troubleshooting Guide](reference/troubleshooting.md)**
   - Common errors
   - Debugging techniques
   - Performance issues
   - Known limitations
   - Solutions and workarounds

10. **[FAQ](reference/faq.md)**
    - General questions
    - Installation and setup
    - Test suite creation
    - Configuration
    - Execution and results
    - Advanced topics

### 🎓 Guides and Best Practices

Learn how to use ATP effectively:

1. **[Best Practices Guide](guides/best-practices.md)**
   - Test suite design
   - Task definition
   - Constraint configuration
   - Assertion strategies
   - Scoring optimization
   - Environment management
   - CI/CD integration
   - Performance optimization

2. **[RBAC Guide](guides/rbac-guide.md)**
   - Permissions and roles
   - Default role configuration
   - Custom role creation
   - API integration
   - Best practices

3. **[WebSocket Guide](guides/websocket-guide.md)**
   - Real-time updates
   - Topics and subscriptions
   - Message types
   - JavaScript client library
   - React hooks integration

4. **[Migration Guide](guides/migration.md)**
   - Migrating from custom solutions
   - Migration strategies
   - Step-by-step process
   - Framework-specific migrations
   - Common challenges
   - Success criteria

5. **[Video Tutorials](guides/video-tutorials.md)** (Planned)
   - Tutorial series structure
   - Beginner series
   - Intermediate series
   - Advanced series
   - Expert series
   - Practice exercises

### 🏗️ Architecture Decision Records

Design decisions and rationale:

1. **[ADR-001: Framework-Agnostic Design](adr/001-framework-agnostic.md)**
   - Decision: Why framework-agnostic?
   - Alternatives considered
   - Consequences
   - Implementation approach

---

## Documentation by Role

### For New Users

Start with these documents:

1. [Installation Guide](guides/installation.md)
2. [Quick Start Guide](guides/quickstart.md)
3. [FAQ](reference/faq.md)
4. [Test Format Reference](reference/test-format.md)

### For Developers

Technical documentation:

1. [Architecture](03-architecture.md)
2. [API Reference](reference/api-reference.md)
3. [Dashboard API Reference](reference/dashboard-api.md)
4. [WebSocket Guide](guides/websocket-guide.md)
5. [Integration Guide](06-integration.md)
6. [Best Practices](guides/best-practices.md)

### For DevOps Engineers

CI/CD and deployment:

1. [Best Practices - CI/CD Integration](guides/best-practices.md#cicd-integration)
2. [Configuration Reference](reference/configuration.md)
3. [Troubleshooting Guide](reference/troubleshooting.md)

### For Team Leads

Strategy and planning:

1. [Vision & Goals](01-vision.md)
2. [Requirements](02-requirements.md)
3. [Migration Guide](guides/migration.md)
4. [Roadmap](07-roadmap.md)

---

## Documentation by Task

### I want to...

**...get started quickly**
→ [Quick Start Guide](guides/quickstart.md)

**...understand what ATP can do**
→ [Vision & Goals](01-vision.md)
→ [Requirements](02-requirements.md)

**...create my first test suite**
→ [Quick Start Guide](guides/quickstart.md)
→ [Test Format Reference](reference/test-format.md)

**...configure my agent**
→ [Adapter Configuration](reference/adapters.md)
→ [Integration Guide](06-integration.md)

**...understand test results**
→ [Usage Guide](guides/usage.md)
→ [Evaluation System](05-evaluators.md)

**...improve my tests**
→ [Best Practices Guide](guides/best-practices.md)

**...migrate from another solution**
→ [Migration Guide](guides/migration.md)

**...integrate with CI/CD**
→ [Best Practices - CI/CD](guides/best-practices.md#cicd-integration)

**...create custom evaluators**
→ [Evaluation System](05-evaluators.md)
→ [API Reference](reference/api-reference.md)

**...troubleshoot issues**
→ [Troubleshooting Guide](reference/troubleshooting.md)
→ [FAQ](reference/faq.md)

**...understand the architecture**
→ [Architecture](03-architecture.md)
→ [ATP Protocol](04-protocol.md)

**...use real-time dashboard updates**
→ [WebSocket Guide](guides/websocket-guide.md)

**...share test suites with others**
→ [Marketplace API Reference](reference/marketplace-api.md)

**...compare agent performance publicly**
→ [Public Leaderboard API Reference](reference/public-leaderboard-api.md)

**...manage user permissions**
→ [RBAC Guide](guides/rbac-guide.md)

---

## Examples

### Complete Examples

See [examples/](../examples/) directory for:

- **Basic Test Suites**: Simple smoke tests and core functionality
- **Advanced Test Suites**: Complex multi-agent scenarios
- **Integration Examples**: Framework-specific integrations
- **Custom Evaluators**: Domain-specific evaluation logic

### Quick Examples

**Minimal Test Suite**:
```yaml
test_suite: "quick_example"
version: "1.0"

agents:
  - name: "my-agent"
    type: "http"
    config:
      endpoint: "http://localhost:8000"

tests:
  - id: "test-001"
    name: "Basic file creation"
    task:
      description: "Create a file named output.txt"
    assertions:
      - type: "artifact_exists"
        config:
          path: "output.txt"
```

**Loading Test Suite**:
```python
from atp.loader import TestLoader

loader = TestLoader()
suite = loader.load_file("test_suite.yaml")

print(f"Suite: {suite.test_suite}")
print(f"Tests: {len(suite.tests)}")
```

---

## Development Status

### Current (v1.0.0 - GA)

✅ **Milestone 1 (MVP) - Complete**:
- ATP Protocol models (Request/Response/Event)
- Core adapters (HTTP, Docker, CLI)
- Test loader with YAML/JSON parsing
- Test runner with parallel execution
- Basic evaluators (Artifact, Behavior)
- Reporters (Console, JSON)
- CLI implementation

✅ **Milestone 2 (Beta) - Complete**:
- Framework adapters (LangGraph, CrewAI, AutoGen)
- Multiple runs with statistical analysis
- LLM-as-Judge evaluator
- Baseline & regression detection
- HTML reporter
- CI/CD integration (GitHub Actions, GitLab CI, JUnit XML)
- Code execution evaluator
- Mock tools for deterministic testing

✅ **Milestone 3 (GA) - Complete**:
- Web dashboard (FastAPI)
- Security hardening
- Performance optimization
- Complete documentation

See [Roadmap](07-roadmap.md) and [spec/tasks.md](../spec/tasks.md) for details.

---

## Contributing to Documentation

We welcome documentation improvements!

**How to Contribute**:

1. **Report Issues**: Found a typo or error? [Open an issue](https://github.com/yourusername/atp-platform-ru/issues)

2. **Suggest Improvements**: Have ideas for better documentation? Start a [discussion](https://github.com/yourusername/atp-platform-ru/discussions)

3. **Submit Changes**:
   - Fork the repository
   - Make your changes
   - Submit a pull request

**Documentation Standards**:
- Clear, concise writing
- Code examples that work
- Proper formatting (Markdown)
- Links to related content

---

## Documentation Maintenance

### Versioning

Documentation is versioned alongside code releases:
- **v0.1.x**: MVP documentation
- **v0.2.x**: Runner and adapter documentation
- **v1.0.x**: Complete feature documentation

### Updates

Documentation is updated:
- **With each release**: Feature documentation
- **Monthly**: General improvements
- **As needed**: Bug fixes, clarifications

### Feedback

Help us improve documentation:
- [Report documentation issues](https://github.com/yourusername/atp-platform-ru/issues)
- [Suggest improvements](https://github.com/yourusername/atp-platform-ru/discussions)
- [Contribute examples](https://github.com/yourusername/atp-platform-ru/pulls)

---

## Additional Resources

### Community

- **GitHub**: [atp-platform-ru](https://github.com/yourusername/atp-platform-ru)
- **Issues**: [Bug reports and feature requests](https://github.com/yourusername/atp-platform-ru/issues)
- **Discussions**: [Q&A and community](https://github.com/yourusername/atp-platform-ru/discussions)

### Related Projects

- **LangChain**: Framework for LLM applications
- **LangGraph**: Build stateful agents with LangChain
- **CrewAI**: Multi-agent collaboration framework
- **AutoGen**: Microsoft's multi-agent framework

### External Resources

- **Testing AI Systems**: Best practices for testing AI/ML systems
- **LLM Evaluation**: Methods for evaluating large language models
- **Agent Architectures**: Design patterns for AI agents

---

## Quick Reference

### Common Commands

```bash
# Install ATP
uv sync

# Verify installation
uv run pytest tests/ -v

# Run tests
uv run atp test --agent=my-agent suite.yaml
uv run atp test --agent=my-agent --runs=5 suite.yaml
uv run atp test --agent=my-agent --tags=smoke suite.yaml
uv run atp test --agent=my-agent --parallel=4 suite.yaml

# Output formats
uv run atp test --agent=my-agent --output=json --output-file=results.json suite.yaml
uv run atp test --agent=my-agent --output=html --output-file=report.html suite.yaml
uv run atp test --agent=my-agent --output=junit --output-file=results.xml suite.yaml

# Validate test definitions
uv run atp validate suite.yaml

# Baseline management
uv run atp baseline save --name=v1.0 results.json
uv run atp baseline compare --baseline=v1.0 results.json

# List agents
uv run atp list-agents
```

### Key Concepts

- **Test Suite**: Collection of tests with configuration
- **Test Definition**: Single test case with task and assertions
- **Agent**: System under test (HTTP, Docker, CLI, etc.)
- **Adapter**: Bridge between ATP and agent implementation
- **Assertion**: Validation rule for test results
- **Evaluator**: Component that assesses agent performance
- **Artifact**: Output file produced by agent

### File Locations

```
atp-platform-ru/
├── docs/              # This documentation
├── examples/          # Example code and test suites
│   ├── test_suites/   # Sample YAML test suites
│   └── ci/            # CI/CD templates
├── atp/               # ATP source code
│   ├── cli/           # CLI commands
│   ├── core/          # Core utilities, security
│   ├── protocol/      # ATP protocol models
│   ├── loader/        # Test suite loader
│   ├── runner/        # Test orchestration
│   ├── adapters/      # Agent adapters
│   ├── evaluators/    # Result evaluation
│   ├── scoring/       # Score aggregation
│   ├── statistics/    # Statistical analysis
│   ├── baseline/      # Baseline management
│   ├── reporters/     # Output formatting
│   ├── streaming/     # Event streaming
│   ├── mock_tools/    # Mock tool server
│   ├── performance/   # Performance utilities
│   └── dashboard/     # Web interface
├── tests/             # ATP's own tests (80%+ coverage)
│   ├── unit/          # Unit tests
│   └── e2e/           # End-to-end tests
└── spec/              # Requirements and tasks
```

---

## Getting Help

### Quick Help

- **Installation issues**: [Installation Guide](guides/installation.md)
- **Getting started**: [Quick Start Guide](guides/quickstart.md)
- **Common questions**: [FAQ](reference/faq.md)
- **Errors and bugs**: [Troubleshooting Guide](reference/troubleshooting.md)

### Support Channels

1. **Documentation**: Start here - most questions are answered
2. **FAQ**: Check [FAQ](reference/faq.md) for common questions
3. **GitHub Issues**: [Report bugs](https://github.com/yourusername/atp-platform-ru/issues)
4. **GitHub Discussions**: [Ask questions](https://github.com/yourusername/atp-platform-ru/discussions)

---

## License

ATP is released under the MIT License. See [LICENSE](../LICENSE) for details.

---

## Acknowledgments

ATP is built on ideas and inspiration from:
- LangChain and LangGraph communities
- Software testing best practices
- AI evaluation research
- Open source contributors

---

**Ready to get started?** → [Installation Guide](guides/installation.md)

**Have questions?** → [FAQ](reference/faq.md)

**Need help?** → [Troubleshooting Guide](reference/troubleshooting.md)
