# Code Review Request

## Task Completed: ${TASK_ID} — ${TASK_NAME}

## Changed Files:
${CHANGED_FILES}

## Diff Summary:
${GIT_DIFF}

## Review Instructions:

Launch the following review agents in parallel using the Task tool:

### 1. Quality Agent
Review the code changes for:
- Bugs and logic errors
- Security vulnerabilities
- Error handling gaps

### 2. Implementation Agent
Verify the implementation:
- Code achieves the stated task goals
- All checklist items are properly implemented
- Edge cases are handled

### 3. Testing Agent
Review test coverage:
- New code has adequate test coverage
- Tests are meaningful and not trivial

## Output:

For each issue found, describe it briefly.
If issues are found, fix them and respond with: "REVIEW_FIXED"
If no issues found, respond with: "REVIEW_PASSED"
