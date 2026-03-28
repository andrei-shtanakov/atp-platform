"""Unit tests for atp.catalog.comparison."""

from atp.catalog.comparison import format_comparison_table


def test_format_comparison_table_basic() -> None:
    """format_comparison_table should include test names, scores, and top agents."""
    tests = [
        {"name": "create-file", "score": 95.0, "avg": 88.5, "best": 99.0},
        {"name": "read-and-transform", "score": 80.0, "avg": 75.0, "best": 92.0},
    ]
    top3 = [
        {"name": "gpt-4o", "score": 99.0},
        {"name": "claude-3-5", "score": 97.5},
        {"name": "gemini-1.5", "score": 94.0},
    ]

    result = format_comparison_table(tests, top3)

    # Rows contain test names
    assert "create-file" in result
    assert "read-and-transform" in result

    # Scores appear formatted
    assert "95.00" in result
    assert "80.00" in result

    # Top agents section
    assert "Top 3 Agents:" in result
    assert "gpt-4o" in result
    assert "claude-3-5" in result
    assert "gemini-1.5" in result

    # Ranking markers
    assert "1." in result
    assert "2." in result
    assert "3." in result


def test_format_comparison_table_empty() -> None:
    """format_comparison_table should return a no-data message for empty input."""
    result = format_comparison_table([], [])
    assert result == "No tests to display."


def test_format_comparison_table_no_top3() -> None:
    """format_comparison_table with empty top3 should omit the top agents section."""
    tests = [{"name": "my-test", "score": 70.0, "avg": 65.0, "best": 80.0}]
    result = format_comparison_table(tests, [])

    assert "my-test" in result
    assert "Top 3 Agents:" not in result


def test_format_comparison_table_none_values() -> None:
    """format_comparison_table should handle None scores gracefully."""
    tests = [{"name": "incomplete", "score": None, "avg": None, "best": None}]
    result = format_comparison_table(tests, [])

    assert "incomplete" in result
    assert "N/A" in result
