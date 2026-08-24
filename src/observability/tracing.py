"""Arize Phoenix OTEL tracing setup and span helpers."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Global tracer — initialized by init_tracing()
_tracer = None


def init_tracing() -> None:
    """
    Initialize Phoenix OpenTelemetry tracing.

    Registers a tracer provider that sends spans to the Phoenix server.
    Call this once at application startup.
    """
    global _tracer

    try:
        from phoenix.otel import register

        from src.config import settings

        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_endpoint,
        )
        _tracer = tracer_provider.get_tracer("hybrid-rag-engine")
        logger.info(
            "Phoenix tracing initialized → %s (project: %s)",
            settings.phoenix_endpoint,
            settings.phoenix_project_name,
        )
    except Exception:
        logger.warning(
            "Phoenix tracing unavailable — continuing without observability",
            exc_info=True,
        )


@contextmanager
def traced_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager that creates a traced span.

    Usage:
        with traced_span("dense_search", {"query": q}) as span_data:
            results = do_search(q)
            span_data["result_count"] = len(results)

    The span_data dict is yielded so callers can add attributes during execution.
    Timing is recorded automatically.
    """
    span_data: dict[str, Any] = {}
    start = time.perf_counter()

    if _tracer is not None:
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            try:
                yield span_data
            finally:
                elapsed = time.perf_counter() - start
                span.set_attribute("duration_seconds", elapsed)
                for k, v in span_data.items():
                    span.set_attribute(k, str(v))
    else:
        # No tracing available — just yield and track timing
        try:
            yield span_data
        finally:
            elapsed = time.perf_counter() - start
            span_data["duration_seconds"] = elapsed


def traced(name: str | None = None):
    """
    Decorator version of traced_span for simple function tracing.

    Usage:
        @traced("my_function")
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with traced_span(span_name):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with traced_span(span_name):
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
