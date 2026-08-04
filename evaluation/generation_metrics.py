"""Pure metric functions for generation faithfulness evaluation."""

import json
import re

FALLBACK_ANSWER = "No sufficiently relevant results found."


class JudgeParseError(Exception):
    pass


def extract_message_text(content: object) -> str:
    """Flatten an LLM message ``content`` into plain text.

    Chat models may return either a plain string or a list of content blocks
    (text blocks plus non-text ones such as Gemini thought signatures). Only
    text is kept; anything else is dropped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def build_judge_prompt(query: str, answer: str, contexts: list[str]) -> str:
    numbered = "\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))
    return (
        f"You are a faithfulness judge.\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Source contexts:\n{numbered}\n\n"
        f"Extract every factual claim made in the Answer. For each claim, determine "
        f"whether it is supported by the source contexts above.\n"
        f"If the Answer makes no factual claim at all — for example it only states "
        f"that the documents do not contain the information — return an empty "
        f'claims list.\n'
        f'Respond with ONLY valid JSON in this exact format: '
        f'{{"claims": [{{"text": "<claim>", "supported": true|false}}]}}'
    )


def _candidate_payloads(text: str) -> list[str]:
    """Fenced blocks first, then the whole text.

    A judge that reasons inside a fence before answering emits several fences,
    so every fenced block is a candidate — not only the first one.
    """
    fences = [
        m.group(1).strip()
        for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    ]
    return [*fences, text]


def parse_judge_response(raw: str) -> list[dict]:
    text = raw.strip()

    parsed = []
    last_exc: json.JSONDecodeError | None = None
    for candidate in _candidate_payloads(text):
        try:
            parsed.append(json.loads(candidate))
        except json.JSONDecodeError as exc:
            last_exc = exc
    if not parsed:
        raise JudgeParseError(f"Invalid JSON: {last_exc}") from last_exc

    # A reasoning fence may itself be valid JSON: prefer the first payload that
    # actually looks like the judge verdict.
    data = next(
        (d for d in parsed if isinstance(d, dict) and "claims" in d),
        parsed[0],
    )

    if not isinstance(data, dict) or "claims" not in data:
        raise JudgeParseError("Missing 'claims' key in judge response")

    claims = data["claims"]
    if not isinstance(claims, list):
        raise JudgeParseError("'claims' must be a list")

    for item in claims:
        if not isinstance(item, dict):
            raise JudgeParseError("Each claim must be a dict")
        if "text" not in item or "supported" not in item:
            raise JudgeParseError("Each claim must have 'text' and 'supported' keys")
        if not isinstance(item["text"], str):
            raise JudgeParseError("'text' must be a str")
        if not isinstance(item["supported"], bool):
            raise JudgeParseError("'supported' must be a bool")

    return claims


def compute_faithfulness(claims: list[dict]) -> float:
    if not claims:
        return 1.0
    supported = sum(1 for c in claims if c["supported"])
    return supported / len(claims)


def is_fallback_answer(answer: str) -> bool:
    return answer == FALLBACK_ANSWER


def compute_negative_refusal(answer: str) -> float:
    return 1.0 if is_fallback_answer(answer) else 0.0
