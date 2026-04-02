"""ATP SDK client for interacting with the ATP benchmark platform."""


class ATPClient:
    """Client for the ATP benchmark platform API."""

    def __init__(
        self,
        platform_url: str = "http://localhost:8000",
        token: str | None = None,
    ) -> None:
        self.platform_url = platform_url.rstrip("/")
        self.token = token
