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
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    from atp.dashboard import run_server

    print(f"Starting ATP Dashboard at http://{args.host}:{args.port}")
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":  # pragma: no cover
    main()
