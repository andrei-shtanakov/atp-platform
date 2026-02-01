"""Unit tests for chaos engineering integration with loader models."""

import pytest

from atp.chaos import (
    ChaosConfig,
    ChaosProfile,
    ErrorType,
    LatencyConfig,
    ToolFailureConfig,
)
from atp.loader.models import (
    ChaosSettings,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)


class TestChaosSettings:
    """Tests for ChaosSettings model."""

    def test_empty_settings(self) -> None:
        """Test empty chaos settings."""
        settings = ChaosSettings()
        assert settings.profile is None
        assert settings.custom is None
        assert settings.get_config() is None

    def test_profile_only(self) -> None:
        """Test settings with profile only."""
        settings = ChaosSettings(profile=ChaosProfile.LIGHT)
        config = settings.get_config()

        assert config is not None
        assert config.enabled is True

    def test_profile_string(self) -> None:
        """Test settings with profile as string."""
        settings = ChaosSettings(profile="moderate")
        config = settings.get_config()

        assert config is not None
        assert config.enabled is True

    def test_custom_config(self) -> None:
        """Test settings with custom config."""
        custom = ChaosConfig(
            enabled=True,
            tool_failures=[ToolFailureConfig(tool="*", probability=0.5)],
        )
        settings = ChaosSettings(custom=custom)
        config = settings.get_config()

        assert config is not None
        assert config == custom

    def test_custom_overrides_profile(self) -> None:
        """Test custom config overrides profile."""
        custom = ChaosConfig(enabled=False)
        settings = ChaosSettings(
            profile=ChaosProfile.HEAVY,
            custom=custom,
        )
        config = settings.get_config()

        assert config is not None
        assert config.enabled is False


class TestTestDefaultsChaos:
    """Tests for chaos in TestDefaults."""

    def test_defaults_no_chaos(self) -> None:
        """Test defaults without chaos settings."""
        defaults = TestDefaults()
        assert defaults.chaos is None

    def test_defaults_with_chaos_profile(self) -> None:
        """Test defaults with chaos profile."""
        defaults = TestDefaults(
            chaos=ChaosSettings(profile=ChaosProfile.LIGHT),
        )
        assert defaults.chaos is not None
        config = defaults.chaos.get_config()
        assert config is not None
        assert config.enabled is True


class TestTestSuiteChaos:
    """Tests for chaos in TestSuite."""

    @pytest.fixture
    def sample_test(self) -> TestDefinition:
        """Create a sample test definition."""
        return TestDefinition(
            id="test-001",
            name="Sample Test",
            task=TaskDefinition(description="Test task"),
        )

    @pytest.fixture
    def sample_suite(self, sample_test: TestDefinition) -> TestSuite:
        """Create a sample test suite."""
        return TestSuite(
            test_suite="Sample Suite",
            tests=[sample_test],
        )

    def test_suite_no_chaos(self, sample_suite: TestSuite) -> None:
        """Test suite without chaos settings."""
        assert sample_suite.chaos is None
        assert sample_suite.get_chaos_config() is None

    def test_suite_with_chaos(self, sample_test: TestDefinition) -> None:
        """Test suite with chaos settings."""
        suite = TestSuite(
            test_suite="Chaos Suite",
            tests=[sample_test],
            chaos=ChaosSettings(profile=ChaosProfile.MODERATE),
        )

        config = suite.get_chaos_config()
        assert config is not None
        assert config.enabled is True

    def test_suite_defaults_chaos(self, sample_test: TestDefinition) -> None:
        """Test suite with defaults chaos settings."""
        suite = TestSuite(
            test_suite="Chaos Suite",
            tests=[sample_test],
            defaults=TestDefaults(
                chaos=ChaosSettings(profile=ChaosProfile.LIGHT),
            ),
        )

        config = suite.get_chaos_config()
        assert config is not None

    def test_suite_chaos_override_defaults(self, sample_test: TestDefinition) -> None:
        """Test suite chaos overrides defaults chaos."""
        suite = TestSuite(
            test_suite="Chaos Suite",
            tests=[sample_test],
            defaults=TestDefaults(
                chaos=ChaosSettings(profile=ChaosProfile.LIGHT),
            ),
            chaos=ChaosSettings(profile=ChaosProfile.HEAVY),
        )

        config = suite.get_chaos_config()
        assert config is not None
        # Heavy profile has higher latency
        assert config.latency is not None
        assert config.latency.max_ms >= 2000

    def test_per_test_chaos_override(self, sample_test: TestDefinition) -> None:
        """Test per-test chaos overrides suite-level."""
        test_with_chaos = TestDefinition(
            id="test-002",
            name="Chaos Test",
            task=TaskDefinition(description="Test with chaos"),
            chaos=ChaosSettings(
                custom=ChaosConfig(
                    enabled=True,
                    latency=LatencyConfig(min_ms=1000, max_ms=2000),
                )
            ),
        )

        suite = TestSuite(
            test_suite="Chaos Suite",
            tests=[sample_test, test_with_chaos],
            chaos=ChaosSettings(profile=ChaosProfile.LIGHT),
        )

        # Suite-level config for regular test
        config1 = suite.get_chaos_config(sample_test)
        assert config1 is not None
        assert config1.latency is not None
        assert config1.latency.max_ms <= 200  # Light profile

        # Per-test config overrides suite
        config2 = suite.get_chaos_config(test_with_chaos)
        assert config2 is not None
        assert config2.latency is not None
        assert config2.latency.max_ms == 2000


class TestChaosConfigYamlCompatibility:
    """Tests for YAML-compatible chaos configuration."""

    def test_chaos_config_from_dict(self) -> None:
        """Test creating ChaosConfig from dict (like YAML parse)."""
        data = {
            "enabled": True,
            "tool_failures": [
                {
                    "tool": "web_search",
                    "probability": 0.3,
                    "error_type": "timeout",
                },
            ],
            "latency": {
                "min_ms": 100,
                "max_ms": 500,
                "affected_tools": ["*"],
            },
        }

        config = ChaosConfig(**data)
        assert config.enabled is True
        assert len(config.tool_failures) == 1
        assert config.tool_failures[0].error_type == ErrorType.TIMEOUT
        assert config.latency is not None

    def test_chaos_settings_from_dict(self) -> None:
        """Test creating ChaosSettings from dict."""
        data = {
            "profile": "light",
        }

        settings = ChaosSettings(**data)
        config = settings.get_config()
        assert config is not None
        assert config.enabled is True

    def test_full_suite_with_chaos_dict(self) -> None:
        """Test creating full suite with chaos from dict."""
        data = {
            "test_suite": "Chaos Test Suite",
            "version": "1.0",
            "chaos": {
                "profile": "moderate",
            },
            "tests": [
                {
                    "id": "test-001",
                    "name": "Sample Test",
                    "task": {"description": "Test task"},
                },
            ],
        }

        suite = TestSuite(**data)
        config = suite.get_chaos_config()
        assert config is not None
        assert config.enabled is True

    def test_test_with_custom_chaos_dict(self) -> None:
        """Test creating test with custom chaos from dict."""
        data = {
            "id": "test-001",
            "name": "Chaos Test",
            "task": {"description": "Test task"},
            "chaos": {
                "custom": {
                    "enabled": True,
                    "tool_failures": [
                        {"tool": "*", "probability": 0.5},
                    ],
                },
            },
        }

        test = TestDefinition(**data)
        assert test.chaos is not None
        config = test.chaos.get_config()
        assert config is not None
        assert len(config.tool_failures) == 1
