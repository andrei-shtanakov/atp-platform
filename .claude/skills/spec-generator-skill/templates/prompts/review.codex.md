# Code Review

Task: ${TASK_ID} — ${TASK_NAME}

Files changed:
${CHANGED_FILES}

Diff:
${GIT_DIFF}

## Instructions

Review the code for:
1. Bugs and errors
2. Security issues
3. Missing error handling
4. Test coverage

## Required Response Format

You MUST end your response with exactly one of these status codes on a new line:

```
REVIEW_PASSED
```
Use this if the code looks good and has no issues.

```
REVIEW_FIXED
```
Use this if you found and fixed issues.

```
REVIEW_FAILED
```
Use this if there are issues that need manual attention.

Do not add any text after the status code.
