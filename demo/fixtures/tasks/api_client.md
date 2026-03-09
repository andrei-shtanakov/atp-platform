# Задание: REST API Client

Напиши класс `APIClient`:

```python
class APIClient:
    def __init__(self, base_url: str, timeout: int = 30):
        ...

    def get(self, path: str, params: dict | None = None) -> dict:
        """GET-запрос, возвращает JSON-ответ как dict."""
        ...

    def post(self, path: str, data: dict | None = None) -> dict:
        """POST-запрос с JSON body, возвращает JSON-ответ."""
        ...
```

Требования:
- Используй библиотеку httpx
- При HTTP ошибках (4xx, 5xx) поднимай httpx.HTTPStatusError
- При таймауте поднимай httpx.TimeoutException
- При ошибке соединения поднимай httpx.ConnectError
- Метод get добавляет params как query string
- Метод post отправляет data как JSON body
