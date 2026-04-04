"""Webhook delivery for benchmark run notifications.

Sends HTTP POST to configured webhook URLs when runs complete or fail.
Includes SSRF protection and retry with backoff.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import anyio
import httpx

logger = logging.getLogger("atp.dashboard.webhook")

RETRY_DELAYS = [1, 5, 15]  # seconds between retries
REQUEST_TIMEOUT = 10.0  # seconds per attempt

# Background tasks set — prevents GC of in-flight deliveries
_background_tasks: set[asyncio.Task[Any]] = set()

_PRIVATE_HOSTNAMES = {"localhost", "localhost.localdomain"}


def validate_webhook_url(url: str) -> None:
    """Validate webhook URL against SSRF attacks.

    Rejects private IPs, loopback, link-local, and non-HTTP schemes.

    Raises:
        ValueError: If URL is blocked.
    """
    if not url:
        raise ValueError("Webhook URL cannot be empty")

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Webhook URL scheme must be http or https, got: {parsed.scheme}"
        )

    hostname = parsed.hostname or ""

    if hostname.lower() in _PRIVATE_HOSTNAMES:
        raise ValueError(f"Webhook URL points to private host: {hostname}")

    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            raise ValueError(f"Webhook URL points to private IP: {hostname}")
    except ValueError as e:
        if "private" in str(e) or "scheme" in str(e):
            raise
        # Not an IP literal — hostname, which is fine


def build_webhook_payload(
    event: str,
    benchmark: Any,
    run: Any,
    tasks_total: int = 0,
) -> dict[str, Any]:
    """Build webhook notification payload."""
    return {
        "event": event,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "delivery_id": str(uuid.uuid4()),
        "benchmark": {
            "id": benchmark.id,
            "name": benchmark.name,
        },
        "run": {
            "id": run.id,
            "status": str(run.status),
            "total_score": run.total_score,
            "tasks_completed": run.current_task_index,
            "tasks_total": tasks_total,
            "started_at": (run.started_at.isoformat() if run.started_at else None),
            "finished_at": (run.finished_at.isoformat() if run.finished_at else None),
        },
    }


async def deliver_webhook(url: str, payload: dict[str, Any]) -> bool:
    """Deliver webhook with retry.

    Makes up to 3 attempts with backoff (1s, 5s, 15s).
    Returns True if delivery succeeded, False if all attempts failed.
    """
    delivery_id = payload.get("delivery_id", "unknown")
    event = payload.get("event", "unknown")
    headers = {
        "Content-Type": "application/json",
        "X-ATP-Event": event,
        "X-ATP-Delivery": str(delivery_id),
    }

    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code < 400:
                    logger.info(
                        "Webhook delivered: %s to %s (attempt %d)",
                        delivery_id,
                        url,
                        attempt + 1,
                    )
                    return True
                logger.warning(
                    "Webhook HTTP %d: %s to %s (attempt %d)",
                    response.status_code,
                    delivery_id,
                    url,
                    attempt + 1,
                )
        except Exception as e:
            logger.warning(
                "Webhook failed: %s to %s (attempt %d): %s",
                delivery_id,
                url,
                attempt + 1,
                e,
            )

        if attempt < len(RETRY_DELAYS) - 1:
            await anyio.sleep(delay)

    logger.error(
        "Webhook delivery failed after %d attempts: %s to %s",
        len(RETRY_DELAYS),
        delivery_id,
        url,
    )
    return False


def schedule_webhook(url: str, payload: dict[str, Any]) -> None:
    """Schedule webhook delivery as a background task.

    Uses a task set to prevent garbage collection of in-flight
    tasks.
    """
    task = asyncio.create_task(deliver_webhook(url, payload))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
