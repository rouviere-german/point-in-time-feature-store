"""structlog JSON logging with PII redaction for both structlog and stdlib (uvicorn) records."""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor

from point_in_time_feature_store.core.redaction import redact_processor

__all__ = ["configure_logging", "get_logger"]


def _shared_processors() -> list[Processor]:
    # Redaction runs after exception formatting so tracebacks are scrubbed too.
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_processor,
    ]


def configure_logging(
    level: str = "INFO", *, json_output: bool = True, stream: TextIO | None = None
) -> None:
    """Configure structlog and route stdlib logging through the same redacting pipeline."""
    level_no = logging.getLevelNamesMapping()[level.upper()]
    output: TextIO = stream if stream is not None else sys.stdout
    renderer: Processor = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    shared = _shared_processors()
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.PrintLoggerFactory(file=output),
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(output)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level_no)
    for noisy in ("uvicorn", "uvicorn.error"):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = True
    # Access logs would print raw query strings; RequestContextMiddleware logs paths instead.
    logging.getLogger("uvicorn.access").disabled = True
    # HTTP client libraries log full request URLs at INFO; URLs can carry entity ids
    # (DELETE /v1/entities/{id}) and tokens, so only their warnings are kept.
    for url_logger in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(url_logger).setLevel(max(level_no, logging.WARNING))


def get_logger(name: str | None = None, **initial_values: Any) -> FilteringBoundLogger:
    """Return a bound structlog logger."""
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    # Binding resolves the configuration immediately; without initial values the proxy stays
    # lazy, so module-level loggers follow a later configure_logging() call.
    if initial_values:
        logger = logger.bind(**initial_values)
    return cast(FilteringBoundLogger, logger)
