# Task: REST API Client

Write an `APIClient` class:

```python
class APIClient:
    def __init__(self, base_url: str, timeout: int = 30):
        ...

    def get(self, path: str, params: dict | None = None) -> dict:
        """GET request, returns the JSON response as a dict."""
        ...

    def post(self, path: str, data: dict | None = None) -> dict:
        """POST request with a JSON body, returns the JSON response."""
        ...
```

Requirements:
- Use the httpx library
- On HTTP errors (4xx, 5xx) raise httpx.HTTPStatusError
- On timeout raise httpx.TimeoutException
- On connection error raise httpx.ConnectError
- The get method adds params as a query string
- The post method sends data as a JSON body
