#!/usr/bin/env bash
# Fail PRs labelled `feat` or `breaking` if they don't add lines under
# `## [Unreleased]` in CHANGELOG.md. Designed to run inside GitHub Actions
# but tested locally via tests/ci/test_check_changelog.py.
#
# Required env:
#   PR_LABELS  — JSON array of label names (e.g. '["feat","bug"]').
#   BASE_SHA   — base commit of the PR (merge base or target branch tip).
#   HEAD_SHA   — head commit of the PR.

set -euo pipefail

err() {
  echo "ERROR: $*" >&2
}

label_present() {
  local needle="$1"
  # Lightweight JSON-array containment check. PR_LABELS comes from
  # GitHub Actions' toJson(...), which never quotes label names with
  # embedded quotes, so substring search is safe enough here.
  printf '%s' "${PR_LABELS:-[]}" | grep -q "\"${needle}\""
}

if ! label_present feat && ! label_present breaking; then
  exit 0
fi

# Step A: CHANGELOG.md must be in the diff at all.
if ! git diff --name-only "$BASE_SHA" "$HEAD_SHA" -- CHANGELOG.md \
     | grep -qx 'CHANGELOG.md'; then
  err "PR labelled \`breaking\`/\`feat\` requires a CHANGELOG.md entry under \`## [Unreleased]\`. Add a bullet to the top section of CHANGELOG.md."
  exit 1
fi

# Step B: parse the post-merge file structure.
post_file=$(git show "${HEAD_SHA}:CHANGELOG.md")
total_lines=$(printf '%s\n' "$post_file" | wc -l | tr -d ' ')

# Find line number of `## [Unreleased]`
u_start=$(printf '%s\n' "$post_file" | grep -n '^## \[Unreleased\]' | head -n1 | cut -d: -f1 || true)
if [[ -z "${u_start:-}" ]]; then
  err "CHANGELOG.md is missing a \`## [Unreleased]\` heading. Add it before submitting."
  exit 1
fi

# Find line number of the next `## [` heading after Unreleased (exclusive end).
u_end=$(printf '%s\n' "$post_file" | awk -v start="$u_start" '
  NR > start && /^## \[/ { print NR; exit }
')
if [[ -z "${u_end:-}" ]]; then
  u_end=$((total_lines + 1))
fi

# Step C: parse `git diff --unified=0` for added (+) lines and check whether
# any of their post-image line numbers fall inside [u_start, u_end).
# Hunk header format: @@ -a,b +c,d @@ — extract `c` (post-image start line).
# Uses portable awk constructs (BSD- and GNU-compatible) instead of the
# gawk-only 3-argument match().
added_in_unreleased=$(git diff --unified=0 "$BASE_SHA" "$HEAD_SHA" -- CHANGELOG.md \
  | awk -v u_start="$u_start" -v u_end="$u_end" '
    /^@@/ {
      # Walk fields to find the one that begins with "+" and parse "c[,d]".
      cur = 0
      for (i = 1; i <= NF; i++) {
        if (substr($i, 1, 1) == "+") {
          val = substr($i, 2)
          n = split(val, parts, ",")
          cur = parts[1] + 0
          break
        }
      }
      next
    }
    /^\+\+\+/ { next }
    /^---/ { next }
    /^\+/ {
      if (cur >= u_start && cur < u_end) { found = 1 }
      cur += 1
      next
    }
    /^-/ { next }
    /^ / { cur += 1; next }
    END { if (found) print "yes" }
  ')

if [[ "$added_in_unreleased" == "yes" ]]; then
  exit 0
fi

err "PR touched CHANGELOG.md but added no lines under \`## [Unreleased]\`. Move your bullet up to the [Unreleased] section."
exit 1
