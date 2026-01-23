"""Reporter for baseline comparison results."""

import json
import sys
from pathlib import Path

from .comparison import ComparisonResult, TestComparison
from .models import ChangeType


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"


# Symbols for status
SYMBOLS = {
    ChangeType.REGRESSION: "↓",
    ChangeType.IMPROVEMENT: "↑",
    ChangeType.NO_CHANGE: "=",
    ChangeType.NEW_TEST: "+",
    ChangeType.REMOVED_TEST: "-",
}


def _supports_color() -> bool:
    """Check if terminal supports color output."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


def _colorize(text: str, color: str, use_colors: bool = True) -> str:
    """Apply color to text if colors are enabled.

    Args:
        text: Text to colorize.
        color: ANSI color code.
        use_colors: Whether to apply colors.

    Returns:
        Colorized text or plain text.
    """
    if not use_colors:
        return text
    return f"{color}{text}{Colors.RESET}"


def format_comparison_console(
    result: ComparisonResult,
    use_colors: bool = True,
    verbose: bool = False,
) -> str:
    """Format comparison result for console output.

    Args:
        result: Comparison result to format.
        use_colors: Whether to use ANSI colors.
        verbose: Whether to show detailed statistics.

    Returns:
        Formatted string for console output.
    """
    lines: list[str] = []

    # Header
    header = "ATP Baseline Comparison"
    lines.append(_colorize(header, Colors.BOLD, use_colors))
    lines.append("=" * len(header))
    lines.append("")

    # Suite info
    lines.append(f"Suite: {result.suite_name}")
    lines.append(f"Agent: {result.agent_name}")
    if result.baseline_created_at:
        baseline_date = result.baseline_created_at.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"Baseline: {baseline_date}")
    lines.append("")

    # Summary
    summary_header = "Summary"
    lines.append(_colorize(summary_header, Colors.BOLD, use_colors))
    lines.append("-" * len(summary_header))

    # Regressions count with color
    reg_count = f"  Regressions: {result.regressions}"
    if result.regressions > 0:
        lines.append(_colorize(reg_count, Colors.RED, use_colors))
    else:
        lines.append(reg_count)

    # Improvements count with color
    imp_count = f"  Improvements: {result.improvements}"
    if result.improvements > 0:
        lines.append(_colorize(imp_count, Colors.GREEN, use_colors))
    else:
        lines.append(imp_count)

    lines.append(f"  No changes: {result.no_changes}")

    if result.new_tests > 0:
        lines.append(f"  New tests: {result.new_tests}")
    if result.removed_tests > 0:
        lines.append(f"  Removed tests: {result.removed_tests}")

    lines.append("")

    # Test details
    if result.comparisons:
        details_header = "Test Details"
        lines.append(_colorize(details_header, Colors.BOLD, use_colors))
        lines.append("-" * len(details_header))

        for comp in result.comparisons:
            line = _format_test_comparison(comp, use_colors, verbose)
            lines.append(line)

            # Show additional details in verbose mode
            if verbose and comp.change_type not in (
                ChangeType.NEW_TEST,
                ChangeType.REMOVED_TEST,
            ):
                details = _format_test_details(comp, use_colors)
                for detail in details:
                    lines.append(f"      {detail}")

        lines.append("")

    # Final status
    if result.has_regressions:
        status = "REGRESSION DETECTED"
        lines.append(_colorize(status, Colors.RED + Colors.BOLD, use_colors))
    elif result.has_improvements:
        status = "IMPROVEMENTS DETECTED"
        lines.append(_colorize(status, Colors.GREEN + Colors.BOLD, use_colors))
    else:
        status = "NO SIGNIFICANT CHANGES"
        lines.append(_colorize(status, Colors.DIM, use_colors))

    return "\n".join(lines)


def _format_test_comparison(
    comp: TestComparison,
    use_colors: bool,
    verbose: bool,
) -> str:
    """Format a single test comparison line.

    Args:
        comp: Test comparison to format.
        use_colors: Whether to use ANSI colors.
        verbose: Whether to show detailed info.

    Returns:
        Formatted line.
    """
    symbol = SYMBOLS.get(comp.change_type, "?")

    # Choose color based on change type
    if comp.change_type == ChangeType.REGRESSION:
        color = Colors.RED
    elif comp.change_type == ChangeType.IMPROVEMENT:
        color = Colors.GREEN
    elif comp.change_type == ChangeType.NEW_TEST:
        color = Colors.CYAN
    elif comp.change_type == ChangeType.REMOVED_TEST:
        color = Colors.YELLOW
    else:
        color = Colors.DIM

    # Build the line
    parts = [f"  {symbol}"]

    # Test name
    name_part = f"{comp.test_name}"
    parts.append(_colorize(name_part, color, use_colors))

    # Score comparison
    if comp.current_mean is not None and comp.baseline_mean is not None:
        score_part = f"{comp.current_mean:.1f}"
        baseline_part = f"(was {comp.baseline_mean:.1f})"
        parts.append(score_part)
        parts.append(_colorize(baseline_part, Colors.DIM, use_colors))

        # Delta
        if comp.delta is not None:
            sign = "+" if comp.delta >= 0 else ""
            delta_str = f"[{sign}{comp.delta:.1f}"
            if comp.delta_percent is not None:
                delta_str += f", {sign}{comp.delta_percent:.1f}%"
            delta_str += "]"
            parts.append(_colorize(delta_str, color, use_colors))

    elif comp.current_mean is not None:
        # New test
        parts.append(f"{comp.current_mean:.1f}")
        parts.append(_colorize("(new)", Colors.CYAN, use_colors))

    elif comp.baseline_mean is not None:
        # Removed test
        baseline_str = f"(was {comp.baseline_mean:.1f})"
        parts.append(_colorize(baseline_str, Colors.YELLOW, use_colors))

    # Significance marker
    if comp.is_significant and comp.p_value is not None:
        sig_str = f"p={comp.p_value:.4f}"
        parts.append(_colorize(f"*{sig_str}", Colors.BOLD, use_colors))

    return " ".join(parts)


def _format_test_details(comp: TestComparison, use_colors: bool) -> list[str]:
    """Format detailed statistics for a test comparison.

    Args:
        comp: Test comparison.
        use_colors: Whether to use colors.

    Returns:
        List of detail lines.
    """
    details = []

    if comp.current_std is not None and comp.baseline_std is not None:
        std_line = f"std: {comp.current_std:.2f} (was {comp.baseline_std:.2f})"
        details.append(_colorize(std_line, Colors.DIM, use_colors))

    if comp.t_statistic is not None:
        stat_line = f"t={comp.t_statistic:.3f}, p={comp.p_value:.4f}"
        details.append(_colorize(stat_line, Colors.DIM, use_colors))

    return details


def format_comparison_json(result: ComparisonResult, indent: int = 2) -> str:
    """Format comparison result as JSON.

    Args:
        result: Comparison result to format.
        indent: JSON indentation level.

    Returns:
        JSON string.
    """
    return json.dumps(result.to_dict(), indent=indent, ensure_ascii=False)


def print_comparison(
    result: ComparisonResult,
    output_format: str = "console",
    output_file: Path | None = None,
    use_colors: bool = True,
    verbose: bool = False,
) -> None:
    """Print comparison result.

    Args:
        result: Comparison result to print.
        output_format: Output format ('console' or 'json').
        output_file: Optional file path to write output.
        use_colors: Whether to use ANSI colors (console only).
        verbose: Whether to show detailed statistics (console only).
    """
    if output_format == "json":
        output = format_comparison_json(result)
    else:
        use_colors = use_colors and _supports_color()
        output = format_comparison_console(result, use_colors, verbose)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")
    else:
        print(output)
