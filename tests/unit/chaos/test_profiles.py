"""Unit tests for chaos engineering profiles."""

import pytest

from atp.chaos import (
    ChaosConfig,
    ChaosProfile,
    get_profile,
    get_profile_description,
    list_profiles,
)


class TestGetProfile:
    """Tests for get_profile function."""

    def test_get_profile_none(self) -> None:
        """Test getting 'none' profile."""
        config = get_profile(ChaosProfile.NONE)
        assert config.enabled is False

    def test_get_profile_light(self) -> None:
        """Test getting 'light' profile."""
        config = get_profile(ChaosProfile.LIGHT)
        assert config.enabled is True
        assert len(config.tool_failures) == 1
        assert config.tool_failures[0].probability == 0.05
        assert config.latency is not None
        assert config.latency.max_ms == 200

    def test_get_profile_moderate(self) -> None:
        """Test getting 'moderate' profile."""
        config = get_profile(ChaosProfile.MODERATE)
        assert config.enabled is True
        assert config.tool_failures[0].probability == 0.15
        assert config.latency is not None
        assert config.partial_response is not None
        assert config.partial_response.enabled is True

    def test_get_profile_heavy(self) -> None:
        """Test getting 'heavy' profile."""
        config = get_profile(ChaosProfile.HEAVY)
        assert config.enabled is True
        assert config.tool_failures[0].probability == 0.30
        assert config.latency is not None
        assert config.latency.max_ms == 2000
        assert config.partial_response is not None
        assert config.rate_limit is not None
        assert config.rate_limit.enabled is True

    def test_get_profile_high_latency(self) -> None:
        """Test getting 'high_latency' profile."""
        config = get_profile(ChaosProfile.HIGH_LATENCY)
        assert config.enabled is True
        assert config.latency is not None
        assert config.latency.min_ms == 500
        assert config.latency.max_ms == 3000
        assert len(config.tool_failures) == 0

    def test_get_profile_flaky_tools(self) -> None:
        """Test getting 'flaky_tools' profile."""
        config = get_profile(ChaosProfile.FLAKY_TOOLS)
        assert config.enabled is True
        assert len(config.tool_failures) >= 2
        # Should have higher failure rates
        assert any(f.probability >= 0.25 for f in config.tool_failures)

    def test_get_profile_rate_limited(self) -> None:
        """Test getting 'rate_limited' profile."""
        config = get_profile(ChaosProfile.RATE_LIMITED)
        assert config.enabled is True
        assert config.rate_limit is not None
        assert config.rate_limit.enabled is True
        assert config.rate_limit.requests_per_minute == 10

    def test_get_profile_resource_constrained(self) -> None:
        """Test getting 'resource_constrained' profile."""
        config = get_profile(ChaosProfile.RESOURCE_CONSTRAINED)
        assert config.enabled is True
        assert config.token_limits is not None
        assert config.partial_response is not None

    def test_get_profile_by_string(self) -> None:
        """Test getting profile by string name."""
        config = get_profile("light")
        assert config.enabled is True
        assert config.latency is not None

    def test_get_profile_invalid_string(self) -> None:
        """Test getting profile with invalid string name."""
        with pytest.raises(ValueError) as exc_info:
            get_profile("invalid_profile")
        assert "Unknown chaos profile" in str(exc_info.value)


class TestListProfiles:
    """Tests for list_profiles function."""

    def test_list_profiles_returns_all(self) -> None:
        """Test list_profiles returns all profile names."""
        profiles = list_profiles()
        assert "none" in profiles
        assert "light" in profiles
        assert "moderate" in profiles
        assert "heavy" in profiles
        assert "high_latency" in profiles
        assert "flaky_tools" in profiles
        assert "rate_limited" in profiles
        assert "resource_constrained" in profiles

    def test_list_profiles_count(self) -> None:
        """Test list_profiles returns correct count."""
        profiles = list_profiles()
        assert len(profiles) == len(ChaosProfile)


class TestGetProfileDescription:
    """Tests for get_profile_description function."""

    def test_description_none(self) -> None:
        """Test description for 'none' profile."""
        desc = get_profile_description(ChaosProfile.NONE)
        assert "No chaos" in desc or "normal" in desc

    def test_description_light(self) -> None:
        """Test description for 'light' profile."""
        desc = get_profile_description("light")
        assert "Light" in desc or "occasional" in desc

    def test_description_heavy(self) -> None:
        """Test description for 'heavy' profile."""
        desc = get_profile_description(ChaosProfile.HEAVY)
        assert "Heavy" in desc or "high" in desc.lower()

    def test_all_profiles_have_descriptions(self) -> None:
        """Test all profiles have descriptions."""
        for profile in ChaosProfile:
            desc = get_profile_description(profile)
            assert desc is not None
            assert len(desc) > 0

    def test_description_invalid_profile(self) -> None:
        """Test description for invalid profile."""
        with pytest.raises(ValueError):
            get_profile_description("invalid_profile")


class TestProfileConfigurations:
    """Tests for profile configuration validity."""

    @pytest.mark.parametrize("profile", list(ChaosProfile))
    def test_profile_is_valid_chaos_config(self, profile: ChaosProfile) -> None:
        """Test all profiles return valid ChaosConfig."""
        config = get_profile(profile)
        assert isinstance(config, ChaosConfig)

    @pytest.mark.parametrize("profile", list(ChaosProfile))
    def test_profile_probabilities_in_range(self, profile: ChaosProfile) -> None:
        """Test all profile probabilities are in valid range."""
        config = get_profile(profile)
        for failure in config.tool_failures:
            assert 0.0 <= failure.probability <= 1.0

    def test_profile_latency_ranges_valid(self) -> None:
        """Test all profiles with latency have valid ranges."""
        for profile in ChaosProfile:
            config = get_profile(profile)
            if config.latency:
                assert config.latency.min_ms <= config.latency.max_ms
                assert config.latency.min_ms >= 0
