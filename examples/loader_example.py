"""Example demonstrating the TestLoader usage."""

from atp.loader import TestLoader

# Load a test suite from YAML file
loader = TestLoader()
suite = loader.load_file("tests/fixtures/test_suites/valid_suite.yaml")

# Print suite information
print(f"Test Suite: {suite.test_suite}")
print(f"Version: {suite.version}")
print(f"Number of tests: {len(suite.tests)}")
print(f"\nTests:")
for test in suite.tests:
    print(f"  - {test.id}: {test.name}")
    print(f"    Tags: {test.tags}")
    print(f"    Assertions: {len(test.assertions)}")

# Example with variable substitution
print("\n\nExample with variables:")
loader_with_env = TestLoader(
    env={"API_ENDPOINT": "https://api.example.com", "TEST_VAR": "production"}
)
suite_with_vars = loader_with_env.load_file("tests/fixtures/test_suites/with_vars.yaml")
print(f"Agent endpoint: {suite_with_vars.agents[0].config['endpoint']}")
print(f"Task description: {suite_with_vars.tests[0].task.description}")
