"""Git commit reconstruction evaluator.

Evaluates agent ability to reconstruct real git commits by comparing
the agent's output diff against the ground truth diff from a commit.
"""

import logging
import subprocess
from pathlib import Path

from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

logger = logging.getLogger(__name__)


def _get_commit_diff(repo_path: str, commit_sha: str) -> str | None:
    """Get the diff for a specific commit.

    Args:
        repo_path: Path to git repository.
        commit_sha: Commit SHA to get diff for.

    Returns:
        Diff string or None if failed.
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{commit_sha}~1", commit_sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Failed to get commit diff: %s", e)
    return None


def _extract_changed_files(diff: str) -> set[str]:
    """Extract file paths changed in a diff."""
    files: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line.split("/", 1)[1] if "/" in line else ""
            if path and path != "/dev/null":
                files.add(path)
    return files


def _compute_line_similarity(ground_truth: str, candidate: str) -> float:
    """Compute line-level similarity between two diffs.

    Returns a score between 0.0 and 1.0.
    """
    gt_lines = set(ground_truth.strip().splitlines())
    cd_lines = set(candidate.strip().splitlines())

    if not gt_lines:
        return 1.0 if not cd_lines else 0.0

    intersection = gt_lines & cd_lines
    union = gt_lines | cd_lines

    if not union:
        return 1.0

    return len(intersection) / len(union)


class GitCommitEvaluator(Evaluator):
    """Evaluates agent output against ground truth git commits.

    Compares across 4 dimensions:
    - Completeness: are all changed files present?
    - Accuracy: line-level diff similarity
    - Scope: did the agent touch only the right files?
    - Quality: absence of extraneous changes
    """

    @property
    def name(self) -> str:
        return "git_commit"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate agent diff against ground truth commit."""
        checks: list[EvalCheck] = []

        # Get ground truth diff
        ground_truth_diff = self._get_ground_truth(task, assertion)
        if ground_truth_diff is None:
            checks.append(
                self._create_check(
                    "ground_truth",
                    passed=False,
                    message="Could not obtain ground truth diff",
                )
            )
            return self._create_result(checks)

        # Get agent's diff from response artifacts
        agent_diff = self._get_agent_diff(response)
        if agent_diff is None:
            checks.append(
                self._create_check(
                    "agent_output",
                    passed=False,
                    message="No diff found in agent response artifacts",
                )
            )
            return self._create_result(checks)

        # Dimension 1: Completeness (file coverage)
        gt_files = _extract_changed_files(ground_truth_diff)
        agent_files = _extract_changed_files(agent_diff)

        if gt_files:
            covered = len(gt_files & agent_files) / len(gt_files)
        else:
            covered = 1.0

        checks.append(
            EvalCheck(
                name="completeness",
                passed=covered >= 0.8,
                score=covered,
                message=f"Agent covered {covered:.0%} of changed files",
                details={
                    "expected_files": sorted(gt_files),
                    "agent_files": sorted(agent_files),
                },
            )
        )

        # Dimension 2: Accuracy (line similarity)
        similarity = _compute_line_similarity(ground_truth_diff, agent_diff)
        checks.append(
            EvalCheck(
                name="accuracy",
                passed=similarity >= 0.5,
                score=similarity,
                message=f"Line-level similarity: {similarity:.0%}",
            )
        )

        # Dimension 3: Scope (no extraneous files)
        extra_files: set[str] = set()
        if agent_files:
            extra_files = agent_files - gt_files
            scope_score = 1.0 - (len(extra_files) / len(agent_files))
        else:
            scope_score = 1.0 if not gt_files else 0.0

        checks.append(
            EvalCheck(
                name="scope",
                passed=scope_score >= 0.8,
                score=max(scope_score, 0.0),
                message=f"Scope precision: {scope_score:.0%}",
                details={"extra_files": sorted(extra_files) if agent_files else []},
            )
        )

        # Dimension 4: Overall quality (weighted average)
        quality = (covered * 0.4) + (similarity * 0.4) + (scope_score * 0.2)
        checks.append(
            EvalCheck(
                name="overall_quality",
                passed=quality >= 0.6,
                score=quality,
                message=f"Overall quality score: {quality:.2f}",
            )
        )

        return self._create_result(checks)

    def _get_ground_truth(
        self,
        task: TestDefinition,
        assertion: Assertion,
    ) -> str | None:
        """Extract ground truth diff from assertion config."""
        config = assertion.config

        # Check for inline expected diff
        expected = config.get("expected")
        if expected and isinstance(expected, str):
            return expected

        # Check for commit SHA + repo path
        commit_sha = config.get("commit_sha")
        repo_path = config.get("repo_path")

        if commit_sha and repo_path:
            return _get_commit_diff(repo_path, commit_sha)

        return None

    def _get_agent_diff(self, response: ATPResponse) -> str | None:
        """Extract diff from agent response artifacts."""
        for artifact in response.artifacts:
            # Check content attr on artifact types that have it
            content = getattr(artifact, "content", None)
            if content and isinstance(content, str):
                if content.startswith("diff ") or content.startswith("--- "):
                    return content

            # Check path-based artifacts for diff files
            path = getattr(artifact, "path", None)
            if path and isinstance(path, str) and path.endswith(".diff"):
                file_path = Path(path)
                if file_path.exists():
                    return file_path.read_text()

        return None
