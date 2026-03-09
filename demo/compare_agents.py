"""Compare ATP test results across multiple agents.

Usage:
    python demo/compare_agents.py demo/results/openai demo/results/anthropic
    python demo/compare_agents.py demo/results/  # all subdirectories
    python demo/compare_agents.py demo/results/ --json comparison.json
"""

import json
import sys
from pathlib import Path
from typing import Any


def load_agent_results(agent_dir: Path) -> dict[str, Any]:
    """Load all JSON result files from an agent directory."""
    results: dict[str, Any] = {}
    for f in sorted(agent_dir.glob("*.json")):
        with open(f) as fh:
            results[f.stem] = json.load(fh)
    return results


def extract_test_scores(
    agent_results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract per-test scores from agent results."""
    tests: dict[str, dict[str, Any]] = {}
    for suite_name, suite_data in agent_results.items():
        for test in suite_data.get("tests", []):
            test_id = test["test_id"]
            evals = test.get("evaluations", [])
            eval_detail = {}
            for e in evals:
                status = "PASS" if e["passed"] else "FAIL"
                eval_detail[e["evaluator"]] = status

            breakdown = test.get("score_breakdown", {})
            components = breakdown.get("components", {})

            tests[test_id] = {
                "suite": suite_name,
                "name": test["test_name"],
                "score": test.get("score"),
                "passed": test["success"],
                "duration": test.get("duration_seconds", 0),
                "evaluations": eval_detail,
                "quality": components.get("quality", {}).get("normalized"),
                "completeness": components.get("completeness", {}).get("normalized"),
                "efficiency": components.get("efficiency", {}).get("normalized"),
                "cost": components.get("cost", {}).get("normalized"),
            }
    return tests


def compute_summary(
    tests: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    """Compute aggregate metrics."""
    scores = [t["score"] for t in tests.values() if t["score"] is not None]
    passed = sum(1 for t in tests.values() if t["passed"])
    total_duration = sum(t["duration"] for t in tests.values())

    dims = ["quality", "completeness", "efficiency", "cost"]
    dim_avgs = {}
    for dim in dims:
        vals = [t[dim] for t in tests.values() if t[dim] is not None]
        dim_avgs[dim] = round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "total_tests": len(tests),
        "passed": passed,
        "pass_rate": round(passed / len(tests), 3) if tests else 0,
        "avg_score": (round(sum(scores) / len(scores), 1) if scores else 0),
        "min_score": round(min(scores), 1) if scores else 0,
        "max_score": round(max(scores), 1) if scores else 0,
        "total_duration": round(total_duration, 1),
        **dim_avgs,
    }


def print_comparison(
    agents: dict[str, dict[str, dict[str, Any]]],
    summaries: dict[str, dict[str, float | int]],
) -> None:
    """Print comparison table to stdout."""
    agent_names = list(agents.keys())

    # Header
    print("\n## Summary\n")
    header = "| Metric |"
    separator = "|--------|"
    for name in agent_names:
        header += f" {name} |"
        separator += "--------|"
    print(header)
    print(separator)

    metrics = [
        ("Tests passed", "passed", "total_tests"),
        ("Pass rate", "pass_rate", None),
        ("Avg score", "avg_score", None),
        ("Min score", "min_score", None),
        ("Max score", "max_score", None),
        ("Duration (s)", "total_duration", None),
        ("Quality", "quality", None),
        ("Completeness", "completeness", None),
        ("Efficiency", "efficiency", None),
        ("Cost", "cost", None),
    ]

    for label, key, total_key in metrics:
        row = f"| {label} |"
        for name in agent_names:
            s = summaries[name]
            if total_key:
                row += f" {s[key]}/{s[total_key]} |"
            elif isinstance(s[key], float) and s[key] <= 1.0:
                row += f" {s[key]:.3f} |"
            else:
                row += f" {s[key]} |"
        print(row)

    # Per-test comparison
    all_test_ids = sorted({tid for a in agents.values() for tid in a})

    print("\n## Per-test scores\n")
    header = "| Test |"
    separator = "|------|"
    for name in agent_names:
        header += f" {name} |"
        separator += "--------|"
    print(header)
    print(separator)

    for tid in all_test_ids:
        # Get test name from first agent that has it
        test_name = ""
        for a in agents.values():
            if tid in a:
                test_name = a[tid]["name"][:30]
                break

        row = f"| {tid} {test_name} |"
        for name in agent_names:
            if tid in agents[name]:
                t = agents[name][tid]
                score = t["score"]
                mark = "+" if t["passed"] else "x"
                score_str = f"{score:.0f}" if score is not None else "N/A"
                row += f" {mark} {score_str} |"
            else:
                row += " — |"
        print(row)

    # Per-test evaluator details
    print("\n## Evaluator details\n")
    for tid in all_test_ids:
        print(f"**{tid}**")
        for name in agent_names:
            if tid in agents[name]:
                evals = agents[name][tid]["evaluations"]
                parts = [f"{k}:{v}" for k, v in evals.items()]
                print(f"  {name}: {', '.join(parts)}")
        print()


def main() -> None:
    args = sys.argv[1:]
    json_output = None

    if "--json" in args:
        idx = args.index("--json")
        json_output = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]

    if not args:
        print(__doc__)
        sys.exit(1)

    # Resolve agent directories
    agent_dirs: list[Path] = []
    for arg in args:
        p = Path(arg)
        if not p.exists():
            print(f"Error: {p} does not exist")
            sys.exit(1)
        if (p / "smoke.json").exists() or (p / "functional.json").exists():
            # Direct agent results dir
            agent_dirs.append(p)
        else:
            # Parent dir — scan subdirectories
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and any(sub.glob("*.json")):
                    agent_dirs.append(sub)

    if not agent_dirs:
        print("No result directories found")
        sys.exit(1)

    agents: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, float | int]] = {}

    for d in agent_dirs:
        name = d.name
        raw = load_agent_results(d)
        tests = extract_test_scores(raw)
        agents[name] = tests
        summaries[name] = compute_summary(tests)

    print_comparison(agents, summaries)

    if json_output:
        comparison = {
            "agents": list(agents.keys()),
            "summaries": summaries,
            "per_test": {},
        }
        all_ids = sorted({tid for a in agents.values() for tid in a})
        for tid in all_ids:
            comparison["per_test"][tid] = {}
            for name in agents:
                if tid in agents[name]:
                    comparison["per_test"][tid][name] = agents[name][tid]
        with open(json_output, "w") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved to {json_output}")


if __name__ == "__main__":
    main()
