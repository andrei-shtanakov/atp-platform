"""Device Flow authentication for ATP SDK.

Implements CLI login via GitHub Device Flow:
1. Call POST /auth/device → get user_code + verification_uri
2. Open browser for user to enter the code
3. Poll POST /auth/device/poll until authorized
4. Save token to ~/.atp/config.json
"""

from __future__ import annotations

import json
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import httpx

CONFIG_DIR = Path.home() / ".atp"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_token(platform_url: str | None = None) -> str | None:
    """Load saved token from ~/.atp/config.json.

    Args:
        platform_url: If provided, load token for this specific platform.

    Returns:
        Token string or None if not found.
    """
    if not CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(CONFIG_FILE.read_text())
        if platform_url:
            return data.get("tokens", {}).get(platform_url)
        return data.get("token")
    except (json.JSONDecodeError, KeyError):
        return None


def save_token(token: str, platform_url: str | None = None) -> None:
    """Save token to ~/.atp/config.json.

    Args:
        token: JWT access token.
        platform_url: If provided, save token keyed by platform URL.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
        except json.JSONDecodeError:
            data = {}

    data["token"] = token
    if platform_url:
        tokens = data.setdefault("tokens", {})
        tokens[platform_url] = token

    CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")


def login(
    platform_url: str,
    open_browser: bool = True,
) -> str:
    """Perform Device Flow login against an ATP platform.

    Args:
        platform_url: Base URL of the ATP platform.
        open_browser: Whether to automatically open the browser.

    Returns:
        JWT access token.

    Raises:
        RuntimeError: If login fails or times out.
    """
    base = platform_url.rstrip("/")

    with httpx.Client(timeout=10.0) as http:
        # Step 1: Initiate device flow
        resp = http.post(f"{base}/api/auth/device")
        if resp.status_code == 501:
            raise RuntimeError(
                "GitHub OAuth is not configured on this platform. "
                "Ask the admin to set ATP_GITHUB_CLIENT_ID and "
                "ATP_GITHUB_CLIENT_SECRET."
            )
        resp.raise_for_status()
        data = resp.json()

        device_code = data["device_code"]
        user_code = data["user_code"]
        verification_uri = data["verification_uri"]
        expires_in = data["expires_in"]
        interval = data["interval"]

        # Step 2: Show instructions
        display_code = f"{user_code[:4]}-{user_code[4:]}"
        print(f"\nOpen {verification_uri} and enter code: {display_code}\n")

        if open_browser:
            webbrowser.open(verification_uri)

        # Step 3: Poll until authorized
        deadline = time.monotonic() + expires_in
        while time.monotonic() < deadline:
            time.sleep(interval)
            sys.stdout.write(".")
            sys.stdout.flush()

            resp = http.post(
                f"{base}/api/auth/device/poll",
                json={"device_code": device_code},
            )

            if resp.status_code == 428:
                # Still pending
                continue
            if resp.status_code == 410:
                raise RuntimeError("Device code expired. Please try again.")
            if resp.status_code == 404:
                raise RuntimeError("Device code not found. Please try again.")
            resp.raise_for_status()

            # Success
            token = resp.json()["access_token"]
            print("\nLogin successful!")

            save_token(token, platform_url=base)
            return token

    raise RuntimeError("Login timed out. Please try again.")
