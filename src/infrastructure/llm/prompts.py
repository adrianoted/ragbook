"""Shared RAG prompt templates with output-mode support.

The user can prefix their query with a slash-command (e.g. ``/summary``)
to control the style of the answer.  Mode parsing lives in
``src.domain.query_mode``; ``build_rag_prompt`` assembles the final prompt.
"""

from __future__ import annotations

from src.domain.query_mode import QUERY_MODES

MAX_CONTEXT_CHARS = 30_000

# ── output-mode instructions ────────────────────────────────────────
_INSTRUCTION_TEXTS: dict[str, str] = {
    "explain": (
        "Provide a thorough, didactic explanation. "
        "Break down concepts step by step, clarify terminology, "
        "and use examples where helpful."
    ),
    "summary": (
        "Provide a concise summary. "
        "Use short sentences or bullet points. "
        "Focus only on the key facts — no extra commentary."
    ),
    "commands": (
        "Reply ONLY with concrete, actionable commands or steps. "
        "Use a numbered list. No introductions or explanations — "
        "just the commands/actions the user needs to execute."
    ),
    "analyze": (
        "Provide a critical analysis. "
        "Highlight strengths, weaknesses, trade-offs, and implications. "
        "Compare alternatives if relevant."
    ),
}

_MODE_INSTRUCTIONS = {m: _INSTRUCTION_TEXTS[m] for m in QUERY_MODES}

_DEFAULT_INSTRUCTION = (
    "Provide a clear, well-structured answer. "
    "Use bullet points or numbered lists when it improves readability."
)

# ── base prompt ─────────────────────────────────────────────────────
_RAG_PROMPT_TEMPLATE = """\
You are a document-consultation assistant. Answer the user's question using \
only the numbered documents below.

Rules:
- Answer exclusively from the documents provided. Do not use external knowledge.
- Cite your sources inline using bracket notation: [1], [2], etc., \
  matching the document numbers above.
- If the documents do not contain the information needed to answer, \
  state explicitly: "The provided documents do not contain information about this."
- {mode_instruction}
- Answer in the same language as the question.

DOCUMENTS:
{context}

QUESTION:
{prompt}

ANSWER:
"""


def build_rag_prompt(query: str, context_text: str, mode: str | None = None) -> str:
    """Build the final prompt string for the LLM."""
    instruction = _MODE_INSTRUCTIONS.get(mode, _DEFAULT_INSTRUCTION) if mode else _DEFAULT_INSTRUCTION
    return _RAG_PROMPT_TEMPLATE.format(
        mode_instruction=instruction,
        context=context_text,
        prompt=query,
    )


def truncate_context(context: list[str]) -> str:
    """Join context chunks into a single string, respecting the char limit."""
    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(context, 1):
        entry = f"[{i}] {chunk}"
        if total + len(entry) > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)
