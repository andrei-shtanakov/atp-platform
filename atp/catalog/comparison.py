"""Plain-text comparison table renderer for catalog test results."""


def format_comparison_table(tests: list[dict], top3: list[dict]) -> str:
    """Format a comparison table of tests and top-3 agent results.

    Args:
        tests: List of dicts with keys ``name``, ``score``, ``avg``, ``best``.
        top3: List of dicts with keys ``name`` and ``score`` for top agents.

    Returns:
        A plain-text formatted table string.
    """
    if not tests:
        return "No tests to display."

    # Build header
    lines: list[str] = []
    col_name = "Test"
    col_score = "Score"
    col_avg = "Avg"
    col_best = "Best"

    # Determine column widths
    name_width = max(len(col_name), *(len(str(t.get("name", ""))) for t in tests))
    score_width = max(len(col_score), 6)
    avg_width = max(len(col_avg), 6)
    best_width = max(len(col_best), 6)

    def fmt_float(value: object) -> str:
        if value is None:
            return "  N/A"
        try:
            return f"{float(value):6.2f}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "  N/A"

    sep = (
        f"+-{'-' * name_width}-+-{'-' * score_width}-+"
        f"-{'-' * avg_width}-+-{'-' * best_width}-+"
    )

    header = (
        f"| {col_name:<{name_width}} | {col_score:>{score_width}} |"
        f" {col_avg:>{avg_width}} | {col_best:>{best_width}} |"
    )

    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for test in tests:
        name_val = str(test.get("name", ""))
        score_val = fmt_float(test.get("score"))
        avg_val = fmt_float(test.get("avg"))
        best_val = fmt_float(test.get("best"))
        lines.append(
            f"| {name_val:<{name_width}} | {score_val:>{score_width}} |"
            f" {avg_val:>{avg_width}} | {best_val:>{best_width}} |"
        )

    lines.append(sep)

    if top3:
        lines.append("")
        lines.append("Top 3 Agents:")
        for rank, agent in enumerate(top3[:3], start=1):
            agent_name = agent.get("name", "")
            agent_score = agent.get("score")
            score_str = fmt_float(agent_score).strip()
            lines.append(f"  {rank}. {agent_name} — {score_str}")

    return "\n".join(lines)
