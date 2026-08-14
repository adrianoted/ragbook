"""Pure event-to-output mapping for the Search tab streaming handler."""
from dataclasses import dataclass, replace

from src.ui.constants import (
    CHUNK_PREVIEW_MAX_CHARS,
    SEARCH_STATUS_GENERATING,
    SEARCH_STATUS_IDLE,
)


@dataclass(frozen=True)
class SearchOutputs:
    """A snapshot of the Search tab outputs, in the order the handler yields them.

    - answer: the markdown rendered in the answer area, accumulated across deltas
    - sources_rows: the sources table rows (preview, score, filename)
    - last_answer: the hidden state feeding the download button; only populated
                   once the answer is complete, empty while it is still streaming
    - status: the status line text; empty means the line is hidden
    - finished: when True the caller stops consuming the stream
    """

    answer: str
    sources_rows: list[list[str]]
    last_answer: str
    status: str
    finished: bool = False


def initial_outputs() -> SearchOutputs:
    """Return the empty state a search starts from."""
    return SearchOutputs(
        answer="",
        sources_rows=[],
        last_answer="",
        status=SEARCH_STATUS_IDLE,
    )


def next_outputs(event: str, data, previous: SearchOutputs) -> "SearchOutputs | None":
    """Fold one stream event into the previous state.

    Handles the four events emitted by ``POST /api/search/stream``: ``sources``
    raises the "generating" status, the first ``delta`` clears it, and ``done``
    and ``error`` both mark the state finished.

    Returns None for an unknown event, which means the caller should skip the
    yield and leave the UI untouched.
    """
    if event == "sources":
        rows = [
            [
                s["chunk_content"][:CHUNK_PREVIEW_MAX_CHARS],
                f"{s['score']:.4f}",
                s["document_filename"],
            ]
            for s in data
        ]
        return replace(
            previous,
            answer="",
            sources_rows=rows,
            last_answer="",
            status=SEARCH_STATUS_GENERATING,
            finished=False,
        )
    if event == "delta":
        return replace(
            previous,
            answer=previous.answer + data["text"],
            last_answer="",
            status=SEARCH_STATUS_IDLE,
        )
    if event == "done":
        return replace(
            previous,
            last_answer=previous.answer,
            status=SEARCH_STATUS_IDLE,
            finished=True,
        )
    if event == "error":
        detail = data.get("detail", "unknown error") if isinstance(data, dict) else "unknown error"
        return replace(
            previous,
            answer=previous.answer + f"\n\n**Error:** {detail}",
            last_answer="",
            status=SEARCH_STATUS_IDLE,
            finished=True,
        )
    return None
