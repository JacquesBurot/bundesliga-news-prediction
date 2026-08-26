"""Application entry point and command-line error boundary."""

from __future__ import annotations

import httpx

from buli_news.cli.errors import NoMatchesError
from buli_news.cli.parser import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.command_handler(args)
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text.strip()
        response_detail = f": {response_text[:500]}" if response_text else ""
        parser.exit(
            status=1,
            message=(
                f"HTTP request returned status {exc.response.status_code} "
                f"for {exc.request.url}{response_detail}\n"
            ),
        )
    except httpx.HTTPError as exc:
        parser.exit(
            status=1,
            message=f"HTTP request failed: {exc}\n",
        )
    except FileNotFoundError as exc:
        parser.exit(
            status=1,
            message=f"Input file not found: {exc.filename}.\n",
        )
    except NoMatchesError as exc:
        parser.exit(
            status=1,
            message=f"{exc}\n",
        )
    except ValueError as exc:
        parser.exit(
            status=1,
            message=f"Invalid data: {exc}\n",
        )
