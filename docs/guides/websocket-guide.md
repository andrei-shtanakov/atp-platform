# WebSocket Real-Time Updates Guide

This guide explains how to use WebSocket connections for real-time updates in the ATP Dashboard.

## Overview

The ATP Dashboard provides WebSocket support for streaming real-time updates during test execution. This enables:

- **Live test progress** - See test execution progress in real-time
- **Live suite progress** - Monitor overall suite execution status
- **Event streaming** - Receive ATP events as they occur
- **Log streaming** - View logs in real-time
- **Delta updates** - Efficient incremental data updates

## Connection

### WebSocket Endpoint

```
ws://localhost:8000/api/ws/updates
wss://localhost:8000/api/ws/updates  # HTTPS
```

### Client Example (JavaScript)

```javascript
const ws = new ATPWebSocket();

ws.on('test_progress', (data) => {
    console.log('Test progress:', data.progress_percent);
});

ws.on('suite_completed', (data) => {
    console.log('Suite completed:', data.success_rate);
});

ws.subscribe('test:progress', { suite_execution_id: 123 });
ws.connect();
```

### Connection with Client ID

For session persistence across reconnects:

```
ws://localhost:8000/api/ws/updates?client_id=my-client-id
```

---

## Message Types

### Client → Server

| Type | Description | Payload |
|------|-------------|---------|
| `subscribe` | Subscribe to a topic | `{ topic: string, filter?: object }` |
| `unsubscribe` | Unsubscribe from topic | `{ topic: string }` |
| `ping` | Heartbeat ping | `{}` |

### Server → Client

| Type | Description |
|------|-------------|
| `connected` | Connection established, includes `client_id` |
| `subscribed` | Subscription confirmed |
| `unsubscribed` | Unsubscription confirmed |
| `pong` | Heartbeat response |
| `error` | Error message |
| `test_progress` | Test execution progress update |
| `test_completed` | Test execution completed |
| `suite_progress` | Suite execution progress update |
| `suite_completed` | Suite execution completed |
| `log_entry` | Log message |
| `event` | ATP event (tool_call, llm_request, etc.) |
| `delta_update` | Incremental data update |

---

## Topics

Subscribe to topics to receive specific types of updates:

| Topic | Description | Filter Options |
|-------|-------------|----------------|
| `test:progress` | Test progress updates | `suite_execution_id`, `test_id` |
| `test:completed` | Test completion events | `suite_execution_id`, `test_id` |
| `suite:progress` | Suite progress updates | `suite_execution_id` |
| `suite:completed` | Suite completion events | `suite_execution_id` |
| `events:*` | All ATP events | `suite_execution_id`, `event_type` |
| `events:tool_call` | Tool call events | `suite_execution_id` |
| `events:llm_request` | LLM request events | `suite_execution_id` |
| `logs` | Log entries | `suite_execution_id`, `level` |
| `delta` | Delta updates | `resource_type` |

### Subscribe Example

```json
{
  "type": "subscribe",
  "payload": {
    "topic": "test:progress",
    "filter": {
      "suite_execution_id": 123
    }
  }
}
```

---

## Data Payloads

### Test Progress

```json
{
  "type": "test_progress",
  "payload": {
    "suite_execution_id": 123,
    "test_execution_id": 456,
    "test_id": "test-001",
    "test_name": "Login Test",
    "status": "running",
    "progress_percent": 50.0,
    "current_run": 2,
    "total_runs": 5,
    "message": "Executing step 3 of 6"
  }
}
```

### Test Completed

```json
{
  "type": "test_completed",
  "payload": {
    "suite_execution_id": 123,
    "test_execution_id": 456,
    "test_id": "test-001",
    "test_name": "Login Test",
    "success": true,
    "score": 0.95,
    "duration_seconds": 12.5
  }
}
```

### Suite Progress

```json
{
  "type": "suite_progress",
  "payload": {
    "suite_execution_id": 123,
    "suite_name": "Auth Tests",
    "agent_name": "gpt-4",
    "status": "running",
    "progress_percent": 40.0,
    "completed_tests": 4,
    "total_tests": 10,
    "passed_tests": 3,
    "failed_tests": 1
  }
}
```

### Suite Completed

```json
{
  "type": "suite_completed",
  "payload": {
    "suite_execution_id": 123,
    "suite_name": "Auth Tests",
    "agent_name": "gpt-4",
    "success_rate": 0.9,
    "total_tests": 10,
    "passed_tests": 9,
    "failed_tests": 1,
    "duration_seconds": 125.5
  }
}
```

### Log Entry

```json
{
  "type": "log_entry",
  "payload": {
    "suite_execution_id": 123,
    "test_execution_id": 456,
    "level": "info",
    "message": "Starting test execution",
    "timestamp": "2024-01-15T10:30:00Z",
    "source": "runner"
  }
}
```

### ATP Event

```json
{
  "type": "event",
  "payload": {
    "suite_execution_id": 123,
    "test_execution_id": 456,
    "sequence": 15,
    "event_type": "tool_call",
    "timestamp": "2024-01-15T10:30:05Z",
    "payload": {
      "tool": "search",
      "arguments": {"query": "test data"}
    },
    "duration_ms": 150.5
  }
}
```

### Delta Update

```json
{
  "type": "delta_update",
  "payload": {
    "resource_type": "test_execution",
    "resource_id": 456,
    "sequence": 42,
    "changes": {
      "status": "completed",
      "score": 0.95
    }
  }
}
```

---

## JavaScript Client Library

The ATP Dashboard includes a built-in WebSocket client:

```html
<script src="/static/v2/js/websocket.js"></script>
```

### ATPWebSocket Class

```javascript
const ws = new ATPWebSocket({
    url: 'ws://localhost:8000/api/ws/updates',  // Auto-detected if omitted
    autoReconnect: true,                         // Default: true
    reconnectInterval: 1000,                     // Initial retry delay (ms)
    maxReconnectInterval: 30000,                 // Max retry delay (ms)
    reconnectDecay: 1.5,                         // Exponential backoff factor
    maxReconnectAttempts: null,                  // null = infinite
    pingInterval: 25000,                         // Heartbeat interval (ms)
});
```

### Methods

```javascript
// Connect to WebSocket server
await ws.connect();

// Subscribe to a topic
ws.subscribe('test:progress', { suite_execution_id: 123 });

// Unsubscribe from a topic
ws.unsubscribe('test:progress');

// Register event handler
ws.on('test_progress', (data, meta) => {
    console.log('Progress:', data.progress_percent);
    console.log('Sequence:', meta.sequence);
});

// Remove event handler
ws.off('test_progress', handler);

// Disconnect
ws.disconnect();
```

### Events

```javascript
// Connection events
ws.on('connect', () => console.log('Connected'));
ws.on('disconnect', ({ code, reason }) => console.log('Disconnected'));
ws.on('error', ({ event }) => console.error('Error:', event));
ws.on('reconnecting', ({ attempt }) => console.log('Reconnecting...'));
ws.on('reconnect_failed', ({ attempts }) => console.log('Failed to reconnect'));

// Sequence gap detection (for delta updates)
ws.on('sequence_gap', ({ expected, received }) => {
    console.warn(`Missed sequences ${expected} to ${received - 1}`);
    // Trigger full data refresh
});
```

### React Hook

```javascript
function TestProgress({ suiteExecutionId }) {
    const { isConnected, subscribe, on } = useATPWebSocket();
    const [progress, setProgress] = React.useState(0);

    React.useEffect(() => {
        const cleanup = on('test_progress', (data) => {
            if (data.suite_execution_id === suiteExecutionId) {
                setProgress(data.progress_percent);
            }
        });

        subscribe('test:progress', { suite_execution_id: suiteExecutionId });

        return cleanup;
    }, [suiteExecutionId, on, subscribe]);

    return (
        <div>
            <p>Connected: {isConnected ? 'Yes' : 'No'}</p>
            <p>Progress: {progress}%</p>
        </div>
    );
}
```

---

## REST API Endpoints

### Get Connection Info

```http
GET /api/ws/info
```

Returns WebSocket server information:

```json
{
  "url": "ws://localhost:8000/api/ws/updates",
  "ping_interval": 30,
  "ping_timeout": 10
}
```

### List Connected Clients (Admin)

```http
GET /api/ws/clients
Authorization: Bearer <token>
```

Returns list of connected clients:

```json
{
  "clients": [
    {
      "client_id": "abc123",
      "connected_at": "2024-01-15T10:00:00Z",
      "subscriptions": ["test:progress", "logs"],
      "last_activity": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

---

## Reconnection

The JavaScript client handles reconnection automatically with exponential backoff:

1. Initial disconnect detected
2. Wait `reconnectInterval` ms (default: 1000)
3. Attempt reconnection
4. If failed, multiply delay by `reconnectDecay` (default: 1.5)
5. Repeat until `maxReconnectAttempts` reached or connected

On successful reconnection:
- Client ID is preserved (sent as query parameter)
- All subscriptions are automatically restored
- Sequence numbers continue from last received

---

## Best Practices

### 1. Subscribe to Specific Topics

Instead of subscribing to all events, filter by execution ID:

```javascript
// Good - filtered subscription
ws.subscribe('test:progress', { suite_execution_id: 123 });

// Avoid - receives all updates
ws.subscribe('test:progress');
```

### 2. Handle Sequence Gaps

For delta updates, detect and handle sequence gaps:

```javascript
ws.on('sequence_gap', async ({ expected, received }) => {
    // Fetch full data to recover from gap
    const response = await fetch(`/api/executions/${executionId}`);
    const fullData = await response.json();
    updateState(fullData);
});
```

### 3. Implement Heartbeat Monitoring

The client sends pings automatically. Monitor for connection health:

```javascript
ws.on('disconnect', ({ code }) => {
    if (code === 1006) {
        console.warn('Connection lost unexpectedly');
    }
});
```

### 4. Clean Up Subscriptions

Unsubscribe when no longer needed:

```javascript
// In React useEffect cleanup
React.useEffect(() => {
    ws.subscribe('test:progress');
    return () => ws.unsubscribe('test:progress');
}, []);
```

---

## Troubleshooting

### Connection Fails Immediately

1. Check WebSocket URL is correct
2. Verify server is running with WebSocket support
3. Check firewall/proxy settings for WebSocket traffic

### Messages Not Received

1. Verify subscription topic is correct
2. Check filter parameters match data
3. Confirm authentication if required

### High Latency

1. Reduce subscription scope with filters
2. Check network conditions
3. Consider using delta updates instead of full payloads

### Reconnection Loop

1. Check `maxReconnectAttempts` setting
2. Verify server is healthy
3. Check authentication token expiration

---

## See Also

- [Dashboard API Reference](../reference/dashboard-api.md) - REST API endpoints
- [Dashboard Migration Guide](../reference/dashboard-migration.md) - v2 architecture
