"""Unit tests for tag filtering."""

from atp.loader.filters import TagFilter


class TestTagFilter:
    """Test tag filtering logic."""

    def test_empty_filter_matches_all(self) -> None:
        """Empty filter should match all tests."""
        filter_obj = TagFilter([])
        assert filter_obj.matches([])
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["smoke", "slow"])

    def test_include_single_tag(self) -> None:
        """Include filter with single tag."""
        filter_obj = TagFilter(["smoke"])
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["smoke", "fast"])
        assert not filter_obj.matches(["fast"])
        assert not filter_obj.matches([])

    def test_include_multiple_tags_or_logic(self) -> None:
        """Include filter with multiple tags uses OR logic."""
        filter_obj = TagFilter(["smoke", "core"])
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["core"])
        assert filter_obj.matches(["smoke", "core"])
        assert filter_obj.matches(["smoke", "fast"])
        assert not filter_obj.matches(["fast"])
        assert not filter_obj.matches([])

    def test_exclude_single_tag(self) -> None:
        """Exclude filter with single tag."""
        filter_obj = TagFilter(["!slow"])
        assert filter_obj.matches([])
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["fast"])
        assert not filter_obj.matches(["slow"])
        assert not filter_obj.matches(["smoke", "slow"])

    def test_exclude_multiple_tags(self) -> None:
        """Exclude filter with multiple tags."""
        filter_obj = TagFilter(["!slow", "!flaky"])
        assert filter_obj.matches([])
        assert filter_obj.matches(["smoke"])
        assert not filter_obj.matches(["slow"])
        assert not filter_obj.matches(["flaky"])
        assert not filter_obj.matches(["smoke", "slow"])
        assert not filter_obj.matches(["smoke", "flaky"])

    def test_combination_include_and_exclude(self) -> None:
        """Combination of include and exclude filters (AND logic)."""
        filter_obj = TagFilter(["smoke", "!slow"])
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["smoke", "fast"])
        assert not filter_obj.matches(["smoke", "slow"])
        assert not filter_obj.matches(["fast"])
        assert not filter_obj.matches([])

    def test_complex_combination(self) -> None:
        """Complex combination of multiple includes and excludes."""
        filter_obj = TagFilter(["smoke", "core", "!slow", "!flaky"])
        # Must have smoke OR core, and NOT have slow or flaky
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["core"])
        assert filter_obj.matches(["smoke", "core"])
        assert filter_obj.matches(["smoke", "fast"])
        assert not filter_obj.matches(["smoke", "slow"])
        assert not filter_obj.matches(["core", "flaky"])
        assert not filter_obj.matches(["fast"])
        assert not filter_obj.matches([])

    def test_from_string_empty(self) -> None:
        """Create filter from empty string."""
        filter_obj = TagFilter.from_string("")
        assert filter_obj.matches([])
        assert filter_obj.matches(["smoke"])

    def test_from_string_single_include(self) -> None:
        """Create filter from string with single include tag."""
        filter_obj = TagFilter.from_string("smoke")
        assert filter_obj.matches(["smoke"])
        assert not filter_obj.matches(["fast"])

    def test_from_string_multiple_includes(self) -> None:
        """Create filter from string with multiple include tags."""
        filter_obj = TagFilter.from_string("smoke,core")
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["core"])
        assert not filter_obj.matches(["fast"])

    def test_from_string_single_exclude(self) -> None:
        """Create filter from string with single exclude tag."""
        filter_obj = TagFilter.from_string("!slow")
        assert filter_obj.matches(["smoke"])
        assert not filter_obj.matches(["slow"])

    def test_from_string_combination(self) -> None:
        """Create filter from string with combination of include and exclude."""
        filter_obj = TagFilter.from_string("smoke,core,!slow")
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["core"])
        assert not filter_obj.matches(["smoke", "slow"])
        assert not filter_obj.matches(["fast"])

    def test_from_string_with_whitespace(self) -> None:
        """Create filter from string with whitespace."""
        filter_obj = TagFilter.from_string(" smoke , core , !slow ")
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["core"])
        assert not filter_obj.matches(["smoke", "slow"])

    def test_from_string_with_empty_elements(self) -> None:
        """Create filter from string with empty elements."""
        filter_obj = TagFilter.from_string("smoke,,core")
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["core"])

    def test_tag_filter_properties(self) -> None:
        """Test filter properties are set correctly."""
        filter_obj = TagFilter(["smoke", "core", "!slow", "!flaky"])
        assert filter_obj.include_tags == {"smoke", "core"}
        assert filter_obj.exclude_tags == {"slow", "flaky"}

    def test_only_exclude_tags(self) -> None:
        """Test filter with only exclude tags matches tests without them."""
        filter_obj = TagFilter(["!slow", "!flaky"])
        # Should match tests that don't have excluded tags
        assert filter_obj.matches([])
        assert filter_obj.matches(["smoke"])
        assert filter_obj.matches(["smoke", "fast"])
        # Should not match tests with excluded tags
        assert not filter_obj.matches(["slow"])
        assert not filter_obj.matches(["flaky"])
        assert not filter_obj.matches(["smoke", "slow"])
