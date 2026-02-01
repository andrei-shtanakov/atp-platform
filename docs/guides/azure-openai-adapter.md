# Azure OpenAI Adapter Guide

The Azure OpenAI adapter enables ATP to test and evaluate AI models deployed on Azure OpenAI Service. This guide covers configuration, usage, and best practices.

## Prerequisites

Install the optional Azure OpenAI dependencies:

```bash
uv add openai
# For Azure AD authentication, also install:
uv add azure-identity
```

## Quick Start

```yaml
# test_suite.yaml
name: azure-openai-tests
description: Tests for Azure OpenAI deployment

agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"

tests:
  - id: basic-query
    description: Test basic chat completion
    task:
      description: "What is the capital of France?"
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
| `endpoint` | string | Azure OpenAI endpoint URL (e.g., `https://<resource-name>.openai.azure.com`) |
| `deployment_name` | string | Name of the Azure OpenAI model deployment |

### Authentication - API Key

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_key` | string | None | Azure OpenAI API key |

### Authentication - Azure AD

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_azure_ad` | bool | false | Use Azure AD (Entra ID) authentication |
| `tenant_id` | string | None | Azure AD tenant ID |
| `client_id` | string | None | Azure AD client/application ID |
| `client_secret` | string | None | Azure AD client secret |
| `managed_identity_client_id` | string | None | Client ID for user-assigned managed identity |

### API Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_version` | string | `2024-02-15-preview` | Azure OpenAI API version |
| `azure_region` | string | None | Azure region (informational) |
| `timeout_seconds` | float | 300 | Maximum execution time in seconds |

### Model Parameters

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temperature` | float | 0.7 | Temperature for generation (0.0 - 2.0) |
| `max_tokens` | int | 4096 | Maximum tokens in response |
| `top_p` | float | 1.0 | Top-p (nucleus) sampling |
| `frequency_penalty` | float | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | 0.0 | Presence penalty (-2.0 to 2.0) |

### Tool/Function Calling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_function_calling` | bool | true | Enable tool/function calling |
| `tools` | list[dict] | [] | Tool definitions for function calling |
| `tool_choice` | string/dict | None | Tool choice setting ('auto', 'none', 'required') |

### Session Management

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `session_id` | string | None | Specific session ID to use |
| `enable_session_persistence` | bool | false | Persist session across requests |

### Other Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `system_message` | string | None | System message for the assistant |
| `response_format` | dict | None | Response format (e.g., `{"type": "json_object"}`) |
| `seed` | int | None | Seed for deterministic outputs |

## Authentication Methods

### 1. API Key Authentication

The simplest method using an API key:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
```

### 2. Azure AD Default Credential

Uses the DefaultAzureCredential chain (environment variables, managed identity, Azure CLI, etc.):

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    use_azure_ad: true
```

### 3. Service Principal Authentication

Uses a service principal with client credentials:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    use_azure_ad: true
    tenant_id: "${AZURE_TENANT_ID}"
    client_id: "${AZURE_CLIENT_ID}"
    client_secret: "${AZURE_CLIENT_SECRET}"
```

### 4. Managed Identity Authentication

Uses Azure Managed Identity (when running on Azure infrastructure):

```yaml
# System-assigned managed identity
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    use_azure_ad: true

# User-assigned managed identity
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    use_azure_ad: true
    managed_identity_client_id: "${MANAGED_IDENTITY_CLIENT_ID}"
```

## Tool/Function Calling

Test deployments with function calling capabilities:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    tools:
      - type: function
        function:
          name: get_weather
          description: Get the current weather for a location
          parameters:
            type: object
            properties:
              location:
                type: string
                description: The city and state
            required:
              - location
    tool_choice: auto

tests:
  - id: function-calling-test
    description: Test weather function calling
    task:
      description: "What's the weather in Seattle?"
    evaluators:
      - type: behavior
        config:
          expected_tools:
            - "get_weather"
```

## Session Management

### Multi-Turn Conversations

Enable session persistence for multi-turn conversation testing:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    enable_session_persistence: true

tests:
  - id: conversation-turn-1
    description: Start conversation
    task:
      description: "My name is Alice and I live in Seattle."

  - id: conversation-turn-2
    description: Test memory
    task:
      description: "What is my name and where do I live?"
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response correctly recalls the name 'Alice'"
            - "Response correctly recalls 'Seattle'"
```

### Explicit Session ID

Use a specific session ID:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    session_id: "my-session-123"
```

## System Messages

Set up the assistant's behavior with system messages:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    system_message: |
      You are a helpful customer service agent.
      Always be polite and professional.
      If you don't know the answer, say so.

tests:
  - id: customer-service-test
    description: Test customer service responses
    task:
      description: "I need help with my order"
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response is polite and professional"
            - "Response offers to help with the order"
```

## JSON Mode

Request structured JSON responses:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    response_format:
      type: json_object

tests:
  - id: json-output-test
    description: Test JSON output
    task:
      description: |
        Return a JSON object with the following structure:
        {"name": "string", "age": number, "city": "string"}
        Use these values: name=Alice, age=30, city=Seattle
    evaluators:
      - type: code_exec
        config:
          language: python
          code: |
            import json
            output = artifacts['output']['text']
            data = json.loads(output)
            assert data['name'] == 'Alice'
            assert data['age'] == 30
```

## Streaming Events

Use streaming for real-time event monitoring:

```python
from atp.adapters import create_adapter

adapter = create_adapter("azure_openai", {
    "endpoint": "https://my-resource.openai.azure.com",
    "deployment_name": "gpt-4",
    "api_key": "your-api-key",
})

async with adapter:
    async for event in adapter.stream_events(request):
        if isinstance(event, ATPEvent):
            if event.event_type == EventType.PROGRESS:
                print(f"Chunk: {event.payload.get('text', '')}", end="")
            elif event.event_type == EventType.TOOL_CALL:
                print(f"Tool call: {event.payload.get('tool')}")
        else:
            print(f"\nResponse: {event.status}")
```

## Error Handling

The adapter handles common Azure OpenAI errors:

| Error | Exception | Description |
|-------|-----------|-------------|
| Invalid API key | `AdapterConnectionError` | Authentication failed |
| Permission denied | `AdapterConnectionError` | RBAC permissions issue |
| Deployment not found | `AdapterError` | Invalid deployment name |
| Rate limit | `AdapterError` | Rate limit exceeded |
| Timeout | `AdapterTimeoutError` | Request timeout |

## Best Practices

### 1. Use Environment Variables for Secrets

Never commit API keys or secrets to version control:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    deployment_name: "${AZURE_OPENAI_DEPLOYMENT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
```

### 2. Set Appropriate Timeouts

Complex tasks may need longer timeouts:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    timeout_seconds: 120  # 2 minutes
```

### 3. Use Azure AD for Production

For production environments, prefer Azure AD authentication over API keys:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    use_azure_ad: true
```

### 4. Configure Model Parameters for Consistency

For reproducible tests, set temperature and seed:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    temperature: 0.0
    seed: 42
```

### 5. Test Different API Versions

Specify API version for compatibility testing:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    api_version: "2024-02-15-preview"
```

## Example: Complete Test Suite

```yaml
name: azure-openai-comprehensive
description: Comprehensive Azure OpenAI test suite

agent:
  adapter: azure_openai
  config:
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    system_message: "You are a helpful, accurate, and concise assistant."
    temperature: 0.7
    max_tokens: 2048
    timeout_seconds: 120

tests:
  - id: basic-query
    description: Basic chat completion
    task:
      description: "What is 2 + 2?"
    evaluators:
      - type: artifact
        config:
          expected:
            - name: output

  - id: reasoning-test
    description: Test reasoning capabilities
    task:
      description: |
        A farmer has 17 sheep. All but 9 run away.
        How many sheep does the farmer have left?
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response correctly states 9 sheep"
            - "Response explains the reasoning"

  - id: code-generation
    description: Test code generation
    task:
      description: "Write a Python function that calculates the factorial of a number."
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response contains valid Python code"
            - "Function handles edge cases (0, negative numbers)"

  - id: json-output
    description: Test structured output
    task:
      description: |
        Return a JSON object representing a person with name "Alice", age 30, and city "Seattle".
        Only output the JSON, no explanation.
    evaluators:
      - type: code_exec
        config:
          language: python
          code: |
            import json
            data = json.loads(artifacts['output']['text'])
            assert 'name' in data
            assert 'age' in data

  - id: long-context
    description: Test handling of longer contexts
    task:
      description: |
        Summarize the following text in 2-3 sentences:

        The quick brown fox jumps over the lazy dog. This sentence is famous
        for containing every letter of the English alphabet. It has been used
        for typing practice and font demonstrations since the late 19th century.
        The phrase is considered a pangram, which is a sentence using every
        letter of a given alphabet at least once.
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Summary is 2-3 sentences"
            - "Summary captures the key point about pangrams"
```

## Troubleshooting

### "openai is required" Error

Install the openai package:

```bash
uv add openai
```

### "azure-identity is required" Error

Install the azure-identity package for Azure AD authentication:

```bash
uv add azure-identity
```

### "Authentication failed" Error

Verify:
1. API key is correct (if using API key auth)
2. Azure AD credentials are properly configured (if using Azure AD)
3. The identity has the "Cognitive Services OpenAI User" role

### "Deployment not found" Error

Verify:
1. The deployment name matches exactly (case-sensitive)
2. The deployment exists in the specified resource
3. The deployment is in a ready state

### "Rate limit exceeded" Error

Options:
1. Implement retry logic with exponential backoff
2. Increase rate limits in Azure portal
3. Use multiple deployments for load balancing

### Timeout Errors

Increase the timeout for complex tasks:

```yaml
agent:
  adapter: azure_openai
  config:
    endpoint: "https://my-resource.openai.azure.com"
    deployment_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    timeout_seconds: 300  # 5 minutes
```

## Azure RBAC Permissions

For Azure AD authentication, ensure the identity has the appropriate role:

```json
{
  "Name": "Cognitive Services OpenAI User",
  "Description": "Read access to Azure OpenAI resources",
  "Actions": [
    "Microsoft.CognitiveServices/accounts/OpenAI/deployments/chat/completions/action",
    "Microsoft.CognitiveServices/accounts/OpenAI/deployments/completions/action"
  ]
}
```

Assign the role at the resource scope:

```bash
az role assignment create \
  --role "Cognitive Services OpenAI User" \
  --assignee <principal-id> \
  --scope /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<resource>
```
