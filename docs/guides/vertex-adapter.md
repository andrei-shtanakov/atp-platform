# Google Vertex AI Adapter Guide

The Vertex AI adapter enables ATP to test and evaluate AI agents and models deployed on Google Cloud's Vertex AI platform. This guide covers configuration, usage, and best practices.

## Prerequisites

Install the optional Vertex AI dependencies:

```bash
uv add google-cloud-aiplatform
# or install the vertex optional group
pip install atp-platform[vertex]
```

## Quick Start

```yaml
# test_suite.yaml
name: vertex-ai-tests
description: Tests for my Vertex AI model

agent:
  adapter: vertex
  config:
    project_id: "my-gcp-project"
    location: "us-central1"
    model_name: "gemini-1.5-pro"

tests:
  - id: basic-query
    description: Test basic model query
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
| `project_id` | string | Google Cloud project ID |

### Model Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `location` | string | `us-central1` | Google Cloud region for Vertex AI |
| `model_name` | string | `gemini-1.5-pro` | Vertex AI model name |
| `agent_id` | string | None | Vertex AI Agent Builder agent ID (optional) |
| `timeout_seconds` | float | 300 | Maximum execution time in seconds |

### Authentication

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `credentials_path` | string | None | Path to service account JSON key file |
| `service_account_email` | string | None | Service account email for impersonation |

### Session Management

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `session_id` | string | None | Specific session ID to use |
| `enable_session_persistence` | bool | false | Persist session across requests |

### Generation Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temperature` | float | 0.7 | Temperature for generation (0.0 - 2.0) |
| `max_output_tokens` | int | 8192 | Maximum tokens in response |
| `top_p` | float | 0.95 | Top-p (nucleus) sampling parameter |
| `top_k` | int | None | Top-k sampling parameter |

### Function Calling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_function_calling` | bool | true | Enable function calling capabilities |
| `tools` | list[dict] | [] | Tool definitions for function calling |

### Safety Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `safety_settings` | list[dict] | [] | Safety settings per category |
| `block_threshold` | string | None | Default block threshold for all categories |

Valid `block_threshold` values:
- `BLOCK_NONE`
- `BLOCK_LOW_AND_ABOVE`
- `BLOCK_MED_AND_ABOVE`
- `BLOCK_HIGH_AND_ABOVE`

### Grounding

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_grounding` | bool | false | Enable Google Search grounding |
| `grounding_source` | string | None | Grounding source (google_search, vertex_ai_search, or data store ID) |

### System Instruction

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `system_instruction` | string | None | System instruction for the model |

## Google Cloud Authentication

The adapter supports multiple authentication methods:

### 1. Service Account Key File

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    credentials_path: "/path/to/service-account.json"
```

### 2. Service Account Impersonation

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    service_account_email: "my-sa@my-project.iam.gserviceaccount.com"
```

### 3. Application Default Credentials

Set up ADC using gcloud:

```bash
gcloud auth application-default login
```

Then use the adapter without explicit credentials:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
```

### 4. GCE/GKE Service Account

When running on Google Cloud infrastructure (GCE, GKE, Cloud Run), the adapter automatically uses the attached service account.

## Function Calling

Test models with function calling capabilities:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    model_name: "gemini-1.5-pro"
    enable_function_calling: true
    tools:
      - function_declarations:
          - name: "get_weather"
            description: "Get current weather for a location"
            parameters:
              type: "object"
              properties:
                location:
                  type: "string"
                  description: "City name"
                unit:
                  type: "string"
                  enum: ["celsius", "fahrenheit"]
              required: ["location"]

tests:
  - id: function-call-test
    description: Test weather function call
    task:
      description: "What's the weather like in Seattle?"
    evaluators:
      - type: behavior
        config:
          expected_tools:
            - "get_weather"
```

## Google Search Grounding

Enable grounding to use Google Search for fact-checking:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    model_name: "gemini-1.5-pro"
    enable_grounding: true
    grounding_source: "google_search"

tests:
  - id: grounded-response
    description: Test grounded response with sources
    task:
      description: "What were the top news stories today?"
    evaluators:
      - type: artifact
        config:
          expected:
            - name: output
            - name: grounding
```

## Session Management

### Multi-Turn Conversations

Enable session persistence for multi-turn conversation testing:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    enable_session_persistence: true

tests:
  - id: conversation-turn-1
    description: Start conversation
    task:
      description: "My name is Bob"

  - id: conversation-turn-2
    description: Test memory
    task:
      description: "What is my name?"
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response correctly recalls the name 'Bob'"
```

### Explicit Session ID

Use a specific session ID:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    session_id: "my-session-123"
```

## Safety Settings

Configure safety thresholds:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    safety_settings:
      - category: "HARM_CATEGORY_HARASSMENT"
        threshold: "BLOCK_MED_AND_ABOVE"
      - category: "HARM_CATEGORY_HATE_SPEECH"
        threshold: "BLOCK_HIGH_AND_ABOVE"

tests:
  - id: safety-test
    description: Test safety filtering
    task:
      description: "Generate a neutral response"
    evaluators:
      - type: artifact
        config:
          expected:
            - name: output
```

Or use a default threshold for all categories:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    block_threshold: "BLOCK_MED_AND_ABOVE"
```

## System Instructions

Provide system-level instructions:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    system_instruction: |
      You are a helpful customer service assistant.
      Always be polite and professional.
      If you don't know something, say so.

tests:
  - id: persona-test
    description: Test customer service persona
    task:
      description: "How do I reset my password?"
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Response is polite and professional"
            - "Response provides helpful password reset guidance"
```

## Response Artifacts

The Vertex adapter produces the following artifacts:

| Artifact | Content | Always Present |
|----------|---------|----------------|
| `output` | Generated text, session ID, model name | Yes |
| `grounding` | Search citations and sources | When grounding enabled |
| `tool_calls` | Function call details | When functions called |

### Accessing Artifacts

```python
from atp.adapters import create_adapter

adapter = create_adapter("vertex", {
    "project_id": "my-project",
    "enable_grounding": True,
})

async with adapter:
    response = await adapter.execute(request)

    # Get output
    output = next(
        (a for a in response.artifacts if a.name == "output"),
        None
    )
    print(f"Response: {output.data['text']}")

    # Get grounding (if available)
    grounding = next(
        (a for a in response.artifacts if a.name == "grounding"),
        None
    )
    if grounding:
        for chunk in grounding.data.get("grounding_chunks", []):
            print(f"Source: {chunk.get('web', {}).get('uri')}")
```

## Streaming Events

Use streaming for real-time event monitoring:

```python
from atp.adapters import create_adapter

adapter = create_adapter("vertex", {"project_id": "my-project"})

async with adapter:
    async for event in adapter.stream_events(request):
        if isinstance(event, ATPEvent):
            if event.event_type == EventType.PROGRESS:
                print(f"Chunk: {event.payload.get('text', '')}")
            elif event.event_type == EventType.TOOL_CALL:
                print(f"Tool: {event.payload['tool']}")
        else:
            print(f"Response status: {event.status}")
```

## Error Handling

The adapter handles common Google Cloud errors:

| Error | Exception | Description |
|-------|-----------|-------------|
| Permission denied | `AdapterConnectionError` | Insufficient permissions |
| Model not found | `AdapterError` | Invalid model name |
| Quota exceeded | `AdapterError` | Rate limit or quota exceeded |
| Timeout | `AdapterTimeoutError` | Request timeout |

## Best Practices

### 1. Use Appropriate Models

Select models based on your use case:

```yaml
# For fast, cost-effective tasks
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    model_name: "gemini-1.5-flash"

# For complex reasoning
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    model_name: "gemini-1.5-pro"
```

### 2. Set Appropriate Timeouts

Complex tasks may need longer timeouts:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    timeout_seconds: 120  # 2 minutes
```

### 3. Use Environment Variables for Credentials

Never commit credentials to version control:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "${GCP_PROJECT_ID}"
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"
```

### 4. Configure Temperature for Consistency

Lower temperature for more consistent outputs:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    temperature: 0.0  # Deterministic output
```

### 5. Use Grounding for Factual Tasks

Enable grounding for tasks requiring accurate, up-to-date information:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    enable_grounding: true
    grounding_source: "google_search"
```

## Example: Complete Test Suite

```yaml
name: vertex-ai-comprehensive
description: Comprehensive Vertex AI test suite

agent:
  adapter: vertex
  config:
    project_id: "${GCP_PROJECT_ID}"
    location: "us-central1"
    model_name: "gemini-1.5-pro"
    enable_function_calling: true
    enable_grounding: true
    grounding_source: "google_search"
    temperature: 0.7
    max_output_tokens: 4096
    timeout_seconds: 120
    system_instruction: "You are a helpful AI assistant."
    tools:
      - function_declarations:
          - name: "calculate"
            description: "Perform mathematical calculations"
            parameters:
              type: "object"
              properties:
                expression:
                  type: "string"
                  description: "Math expression to evaluate"
              required: ["expression"]

tests:
  - id: basic-query
    description: Basic model query
    task:
      description: "Hello, how are you?"
    evaluators:
      - type: artifact
        config:
          expected:
            - name: output

  - id: grounded-response
    description: Grounded factual query
    task:
      description: "What is the current population of Tokyo?"
    evaluators:
      - type: artifact
        config:
          expected:
            - name: output
            - name: grounding

  - id: function-call-test
    description: Function calling test
    task:
      description: "Calculate 15 * 23"
    evaluators:
      - type: behavior
        config:
          expected_tools:
            - "calculate"

  - id: complex-reasoning
    description: Complex reasoning task
    task:
      description: "Explain the theory of relativity in simple terms"
    constraints:
      max_steps: 5
    evaluators:
      - type: llm_judge
        config:
          criteria:
            - "Explanation is accurate"
            - "Explanation uses simple language"
            - "Explanation covers key concepts"
```

## Troubleshooting

### "google-cloud-aiplatform is required" Error

Install the Vertex AI dependency:

```bash
uv add google-cloud-aiplatform
```

### "Permission Denied" Error

Ensure your credentials have the required permissions:

1. Enable the Vertex AI API in your project
2. Grant the necessary IAM roles:
   - `roles/aiplatform.user` for basic usage
   - `roles/aiplatform.admin` for full access

```bash
gcloud projects add-iam-policy-binding my-project \
  --member="serviceAccount:my-sa@my-project.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### "Model Not Found" Error

Verify:
1. The model name is correct (e.g., `gemini-1.5-pro`, `gemini-1.5-flash`)
2. The model is available in your specified region
3. Your project has access to the model

### Quota Exceeded Error

If you hit quota limits:
1. Request quota increases in the Google Cloud Console
2. Implement retry logic with exponential backoff
3. Use a different region with available quota

### Timeout Errors

Increase the timeout for complex tasks:

```yaml
agent:
  adapter: vertex
  config:
    project_id: "my-project"
    timeout_seconds: 300  # 5 minutes
```

## Model Availability

Common Vertex AI model names:

| Model | Description |
|-------|-------------|
| `gemini-1.5-pro` | Most capable Gemini model |
| `gemini-1.5-flash` | Fast, cost-effective model |
| `gemini-1.0-pro` | Previous generation Gemini |
| `text-bison` | PaLM 2 text model |
| `chat-bison` | PaLM 2 chat model |

Check the [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/overview) for the latest available models.
