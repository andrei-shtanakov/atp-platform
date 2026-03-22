"""Tag filtering utilities for test selection."""

from collections.abc import Sequence


class TagFilter:
    """Filter for selecting tests by tags.

    Supports include and exclude patterns with logical combination:
    - Include: --tags=smoke,core (OR logic: has smoke OR core)
    - Exclude: --tags=!slow (NOT logic: does not have slow)
    - Combination: --tags=smoke,!slow (has smoke AND not slow)
    """

    def __init__(self, tag_expressions: Sequence[str]) -> None:
        """Initialize tag filter.

        Args:
            tag_expressions: List of tag expressions (e.g., ["smoke", "!slow"])
        """
        self.include_tags: set[str] = set()
        self.exclude_tags: set[str] = set()

        for expr in tag_expressions:
            expr = expr.strip()
            if not expr:
                continue

            if expr.startswith("!"):
                # Exclude tag
                self.exclude_tags.add(expr[1:])
            else:
                # Include tag
                self.include_tags.add(expr)

    def matches(self, test_tags: Sequence[str]) -> bool:
        """Check if test tags match the filter.

        Logic:
        1. If exclude tags specified, test must not have any of them
        2. If include tags specified, test must have at least one of them
        3. If only exclude tags, test passes if it doesn't have excluded tags
        4. If only include tags, test passes if it has any included tag
        5. If both, test must match both conditions (AND logic)

        Args:
            test_tags: Tags from a test definition

        Returns:
            True if test matches the filter
        """
        test_tag_set = set(test_tags)

        # Check exclude filter (must not have any excluded tags)
        if self.exclude_tags:
            if test_tag_set & self.exclude_tags:
                return False

        # Check include filter (must have at least one included tag)
        if self.include_tags:
            if not (test_tag_set & self.include_tags):
                return False

        # If no filters specified, match everything
        if not self.include_tags and not self.exclude_tags:
            return True

        # Passed all filters
        return True

    @classmethod
    def from_string(cls, tag_string: str) -> "TagFilter":
        """Create filter from comma-separated string.

        Args:
            tag_string: Comma-separated tags (e.g., "smoke,core,!slow")

        Returns:
            TagFilter instance

        Example:
            >>> filter = TagFilter.from_string("smoke,core,!slow")
            >>> filter.matches(["smoke", "fast"])
            True
            >>> filter.matches(["smoke", "slow"])
            False
        """
        if not tag_string:
            return cls([])

        tags = [tag.strip() for tag in tag_string.split(",")]
        return cls(tags)
