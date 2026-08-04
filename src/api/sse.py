import json
from typing import Any


def format_sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def sources_event(items: list[dict]) -> str:
    return format_sse_event("sources", items)


def delta_event(text: str) -> str:
    return format_sse_event("delta", {"text": text})


def done_event() -> str:
    return format_sse_event("done", {})


def error_event(detail: str) -> str:
    return format_sse_event("error", {"detail": detail})
