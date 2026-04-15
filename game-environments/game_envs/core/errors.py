"""Game-layer exceptions."""


class ValidationError(ValueError):
    """Raised when a game rejects a malformed action."""
