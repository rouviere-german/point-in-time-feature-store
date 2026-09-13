"""Container HEALTHCHECK probe using only the standard library (no curl in the image)."""

from __future__ import annotations

import http.client
import os
import sys

__all__ = ["main", "probe"]


def probe(host: str, port: int, path: str = "/healthz", timeout: float = 2.0) -> bool:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", path, headers={"User-Agent": "healthcheck"})
        return connection.getresponse().status == 200
    except OSError:
        return False
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    del argv
    host = os.environ.get("HEALTHCHECK_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    healthy = probe(host, port)
    sys.stderr.write("healthy\n" if healthy else "unhealthy\n")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
