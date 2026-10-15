"""Optional single-device HTTP server.

Install FastAPI with ``pip install flashspec[serve]`` before calling
:func:`flashspec.serve.app.create_app`.
"""

from flashspec.serve.app import LIMITATIONS, create_app, messages_to_prompt
from flashspec.serve.batching import QueueFullError, RequestBatcher
from flashspec.serve.metrics import ServeMetrics

__all__ = [
    "LIMITATIONS",
    "QueueFullError",
    "RequestBatcher",
    "ServeMetrics",
    "create_app",
    "messages_to_prompt",
]
