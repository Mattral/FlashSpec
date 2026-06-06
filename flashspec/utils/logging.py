"""Structured JSON logger for FlashSpec.

All library code must obtain a logger via ``get_logger(__name__)``.
``print()`` is banned in library code (see AGENTS.md §3.1 and §13.6).

Usage
-----
>>> from flashspec.utils.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Engine initialised", extra={"gamma": 4, "drafter": "llama3-1b"})
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

__all__ = ["get_logger"]

_JSON_FORMAT_VERSION = "1"


class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Each record is a JSON object with keys:
    ``level``, ``logger``, ``message``, ``timestamp``, and any
    ``extra`` fields passed by the caller.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "v": _JSON_FORMAT_VERSION,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, self.datefmt),
        }
        # Merge any extra fields provided via extra={...} in the log call.
        reserved = frozenset(logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in reserved and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that emits structured JSON to stderr.

    Parameters
    ----------
    name : str
        Logger name, conventionally ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    Multiple calls with the same ``name`` return the same logger instance
    (standard Python logging behaviour).  The handler is added only once
    to prevent duplicate log lines.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.debug("Token accepted", extra={"alpha": 0.93})
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
