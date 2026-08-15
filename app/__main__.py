"""
Application entry point.

Run the FastAPI server:
    uv run python -m app
    # or
    uv run uvicorn app.api.main:app --reload

Run the CLI simulator (text mode):
    uv run python -m cli.simulator
    uv run python -m cli.simulator --mock    # no API key needed
"""

from __future__ import annotations

import uvicorn

from app.config import settings


def main() -> None:
    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
