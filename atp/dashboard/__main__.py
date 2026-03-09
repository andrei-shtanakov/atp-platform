"""Entry point for running ATP Dashboard as a module.

Usage:
    python -m atp.dashboard [--host HOST] [--port PORT] [--reload]
"""  # pragma: no cover

import argparse  # pragma: no cover


def main() -> None:  # pragma: no cover
    """Run the ATP Dashboard server."""
    parser = argparse.ArgumentParser(
        description="ATP Dashboard - Web interface for Agent Test Platform results"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config or 8080)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    from atp.core.settings import ATPSettings
    from atp.dashboard import run_server

    settings = ATPSettings()
    host = args.host if args.host is not None else settings.dashboard_host
    port = args.port if args.port is not None else settings.dashboard_port

    print(f"Starting ATP Dashboard at http://{host}:{port}")
    run_server(host=host, port=port, reload=args.reload)


if __name__ == "__main__":  # pragma: no cover
    main()
