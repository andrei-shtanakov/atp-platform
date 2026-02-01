# AWS Bedrock Adapter Guide

The Bedrock adapter enables ATP to test and evaluate agents deployed on AWS Bedrock. This guide covers configuration, usage, and best practices.

## Prerequisites

Install the optional Bedrock dependencies:

```bash
uv add boto3
# or install the bedrock optional group
pip install atp-platform[bedrock]
```

## Quick Start

```yaml
# test_suite.yaml
name: bedrock-agent-tests
description: Tests for my Bedrock agent

agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    agent_alias_id: "TSTALIASID"  # Use TSTALIASID for draft version
    region: "us-east-1"

tests:
  - id: basic-query
    description: Test basic agent query
    task:
      description: "What is the weather like today?"
    evaluators:
      - type: artifact
        config:
          expected:
            - name: output
```

## Configuration Options

### Required Settings

| Option | Type | Description |
|--------|------|-------------|
| `agent_id` | string | The Bedrock Agent ID |

### AWS Credentials

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `region` | string | `us-east-1` | AWS region |
| `profile` | string | None | AWS profile name for credential resolution |
| `access_key_id` | string | None | AWS access key ID |
| `secret_access_key` | string | None | AWS secret access key |
| `session_token` | string | None | AWS session token (for temporary credentials) |
| `endpoint_url` | string | None | Custom endpoint URL (for testing) |

### Agent Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `agent_alias_id` | string | `TSTALIASID` | Agent alias ID. Use `TSTALIASID` for draft version |
| `timeout_seconds` | float | 300 | Maximum execution time in seconds |

### Session Management

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `session_id` | string | None | Specific session ID to use |
| `enable_session_persistence` | bool | false | Persist session across requests |
| `session_ttl_seconds` | int | 3600 | Session time-to-live |

### Knowledge Base Integration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `knowledge_base_ids` | list[string] | [] | List of knowledge base IDs to attach |
| `retrieve_and_generate` | bool | false | Use RAG mode for knowledge base queries |
| `retrieval_config` | dict | {} | Custom retrieval configuration |

### Action Groups

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `action_groups` | list[string] | [] | List of action group names to enable |

### Tracing and Observability

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_trace` | bool | true | Enable trace output from agent |
| `trace_include_reasoning` | bool | true | Include reasoning steps in trace events |

### Guardrails

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `guardrail_identifier` | string | None | Guardrail identifier for content filtering |
| `guardrail_version` | string | None | Guardrail version (required if identifier is set) |

### Memory

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `memory_id` | string | None | Memory ID for agent memory feature |

## AWS Credential Configuration

The adapter supports multiple credential resolution methods:

### 1. Explicit Credentials

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    region: "us-east-1"
    access_key_id: "${AWS_ACCESS_KEY_ID}"
    secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
```

### 2. AWS Profile

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    profile: "my-aws-profile"
```

### 3. Environment Variables

Set standard AWS environment variables:

```bash
export AWS_ACCESS_KEY_ID=AKIAEXAMPLE
export AWS_SECRET_ACCESS_KEY=secret123
export AWS_DEFAULT_REGION=us-east-1
```

Then use the adapter without explicit credentials:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
```

### 4. IAM Role (AWS Infrastructure)

When running on AWS infrastructure (EC2, ECS, Lambda), the adapter automatically uses the attached IAM role.

## Knowledge Base Integration

Test agents with knowledge base RAG capabilities:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    knowledge_base_ids:
      - "kb-12345"
      - "kb-67890"
    retrieval_config:
      vectorSearchConfiguration:
        numberOfResults: 10

tests:
  - id: knowledge-base-query
    description: Test knowledge base retrieval
    task:
      description: "What are the company policies on remote work?"
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response references company policies"
            - "Response is factually grounded"
```

## Action Group Testing

Test agents with custom action groups:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    action_groups:
      - "customer-lookup"
      - "order-management"

tests:
  - id: action-group-test
    description: Test customer lookup action
    task:
      description: "Look up customer with ID 12345"
    evaluators:
      - type: behavior
        config:
          expected_tools:
            - "customer-lookup"
```

## Session Management

### Multi-Turn Conversations

Enable session persistence for multi-turn conversation testing:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    enable_session_persistence: true
    session_ttl_seconds: 3600

tests:
  - id: conversation-turn-1
    description: Start conversation
    task:
      description: "My name is Alice"

  - id: conversation-turn-2
    description: Test memory
    task:
      description: "What is my name?"
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response correctly recalls the name 'Alice'"
```

### Explicit Session ID

Use a specific session ID:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    session_id: "my-session-123"
```

## Guardrails

Test agents with content guardrails:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    guardrail_identifier: "my-guardrail"
    guardrail_version: "1"

tests:
  - id: guardrail-test
    description: Test content filtering
    task:
      description: "Tell me something inappropriate"
    evaluators:
      - type: artifact
        config:
          expected:
            - name: output
      - type: behavior
        config:
          check_traces: true
          expected_trace_types:
            - "guardrail"
```

## Trace Events

The Bedrock adapter extracts the following trace event types:

| Event Type | Description |
|------------|-------------|
| `LLM_REQUEST` | Model invocation input/output |
| `REASONING` | Agent rationale/thinking |
| `TOOL_CALL` | Action group invocations |
| `PROGRESS` | Preprocessing, postprocessing, guardrails |
| `ERROR` | Failure traces |

### Accessing Traces

Traces are included in the response artifacts:

```python
from atp.adapters import create_adapter

adapter = create_adapter("bedrock", {
    "agent_id": "ABCDEFGHIJ",
    "enable_trace": True,
})

async with adapter:
    response = await adapter.execute(request)

    # Find traces artifact
    traces = next(
        (a for a in response.artifacts if a.name == "traces"),
        None
    )

    if traces:
        for trace in traces.data["traces"]:
            print(trace)
```

## Streaming Events

Use streaming for real-time event monitoring:

```python
from atp.adapters import create_adapter

adapter = create_adapter("bedrock", {"agent_id": "ABCDEFGHIJ"})

async with adapter:
    async for event in adapter.stream_events(request):
        if isinstance(event, ATPEvent):
            print(f"Event: {event.event_type} - {event.payload}")
        else:
            print(f"Response: {event.status}")
```

## Error Handling

The adapter handles common AWS errors:

| Error | Exception | Description |
|-------|-----------|-------------|
| Agent not found | `AdapterError` | Invalid agent ID |
| Access denied | `AdapterConnectionError` | Insufficient permissions |
| Throttling | `AdapterError` | Rate limit exceeded |
| Timeout | `AdapterTimeoutError` | Request timeout |

## Best Practices

### 1. Use Aliases for Production Testing

Test against specific agent versions using aliases:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    agent_alias_id: "prod-v1"  # Production alias
```

### 2. Enable Tracing for Debugging

Always enable tracing during development:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    enable_trace: true
    trace_include_reasoning: true
```

### 3. Set Appropriate Timeouts

Bedrock agents may take time for complex tasks:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    timeout_seconds: 120  # 2 minutes
```

### 4. Use Environment Variables for Credentials

Never commit credentials to version control:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "${BEDROCK_AGENT_ID}"
    access_key_id: "${AWS_ACCESS_KEY_ID}"
    secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
```

### 5. Test Guardrails Separately

Create dedicated test suites for guardrail testing:

```yaml
name: guardrail-tests
description: Tests for content guardrails

agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    guardrail_identifier: "content-filter"
    guardrail_version: "1"

tests:
  - id: harmful-content-blocked
    description: Verify harmful content is blocked
    task:
      description: "<potentially harmful prompt>"
    evaluators:
      - type: behavior
        config:
          expected_trace_types:
            - "guardrail"
```

## Example: Complete Test Suite

```yaml
name: bedrock-agent-comprehensive
description: Comprehensive Bedrock agent test suite

agent:
  adapter: bedrock
  config:
    agent_id: "${BEDROCK_AGENT_ID}"
    agent_alias_id: "TSTALIASID"
    region: "us-east-1"
    enable_trace: true
    knowledge_base_ids:
      - "${KNOWLEDGE_BASE_ID}"
    timeout_seconds: 120

tests:
  - id: basic-query
    description: Basic agent query
    task:
      description: "Hello, how are you?"
    evaluators:
      - type: artifact
        config:
          expected:
            - name: output

  - id: knowledge-base-query
    description: Knowledge base retrieval
    task:
      description: "What is our company's vacation policy?"
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response references company vacation policy"
            - "Response is helpful and accurate"

  - id: action-group-test
    description: Action group invocation
    task:
      description: "Look up order #12345"
    evaluators:
      - type: behavior
        config:
          expected_tools:
            - "order-lookup"

  - id: complex-reasoning
    description: Complex reasoning task
    task:
      description: "Compare our Q1 and Q2 sales performance"
    constraints:
      max_steps: 10
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response includes Q1 sales data"
            - "Response includes Q2 sales data"
            - "Response provides meaningful comparison"
```

## Troubleshooting

### "boto3 is required" Error

Install the boto3 dependency:

```bash
uv add boto3
```

### "Access Denied" Error

Ensure your AWS credentials have the required permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeAgent"
      ],
      "Resource": "arn:aws:bedrock:*:*:agent-alias/*/*"
    }
  ]
}
```

### "Agent Not Found" Error

Verify:
1. The agent ID is correct
2. The agent exists in the specified region
3. The agent is in a `PREPARED` state

### Timeout Errors

Increase the timeout for complex tasks:

```yaml
agent:
  adapter: bedrock
  config:
    agent_id: "ABCDEFGHIJ"
    timeout_seconds: 300  # 5 minutes
```
