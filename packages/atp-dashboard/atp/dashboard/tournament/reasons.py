"""Cancellation reason enum. Single source of truth, imported by service,
deadline worker, handlers, models, and tests."""

from enum import StrEnum


class CancelReason(StrEnum):
    ADMIN_ACTION = "admin_action"
    PENDING_TIMEOUT = "pending_timeout"
    ABANDONED = "abandoned"
