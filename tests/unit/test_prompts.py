"""Tests for src/infrastructure/llm/prompts.py — Task A2."""

from __future__ import annotations

import pytest

from src.domain.query_mode import parse_mode
from src.infrastructure.llm.prompts import (
    _MODE_INSTRUCTIONS,
    _RAG_PROMPT_TEMPLATE,
    build_rag_prompt,
    truncate_context,
)


# ── _RAG_PROMPT_TEMPLATE contract ───────────────────────────────────────────

class TestRagPromptTemplate:
    def test_contains_grounding_instruction(self):
        """Template must instruct model to answer only from provided context."""
        template_lower = _RAG_PROMPT_TEMPLATE.lower()
        assert any(phrase in template_lower for phrase in [
            "only",
            "solely",
            "exclusively",
            "do not use",
        ]), "Template must restrict answers to provided context"

    def test_contains_citation_instruction(self):
        """Template must instruct model to cite sources with [n] notation."""
        assert "[n]" in _RAG_PROMPT_TEMPLATE or "[1]" in _RAG_PROMPT_TEMPLATE or (
            "cite" in _RAG_PROMPT_TEMPLATE.lower() or
            "citation" in _RAG_PROMPT_TEMPLATE.lower() or
            "reference" in _RAG_PROMPT_TEMPLATE.lower()
        ), "Template must include citation instructions"
        # Check for bracket notation specifically
        assert "[" in _RAG_PROMPT_TEMPLATE and "]" in _RAG_PROMPT_TEMPLATE

    def test_contains_explicit_fallback(self):
        """Template must tell model to explicitly state when info is not in docs."""
        template_lower = _RAG_PROMPT_TEMPLATE.lower()
        assert any(phrase in template_lower for phrase in [
            "not in",
            "not found",
            "not available",
            "cannot find",
            "cannot answer",
            "not contain",
            "no information",
            "not present",
        ]), "Template must include explicit fallback declaration"

    def test_does_not_contain_supplement_own_knowledge(self):
        """Anti-pattern: model must NOT be invited to go outside context."""
        assert "supplement with your own knowledge" not in _RAG_PROMPT_TEMPLATE.lower()
        assert "supplement" not in _RAG_PROMPT_TEMPLATE.lower()

    def test_contains_language_rule(self):
        """Template must preserve the 'answer in same language as question' rule."""
        template_lower = _RAG_PROMPT_TEMPLATE.lower()
        assert "language" in template_lower, "Template must contain language rule"

    def test_contains_mode_instruction_placeholder(self):
        """Template must still have {mode_instruction} placeholder for modes."""
        assert "{mode_instruction}" in _RAG_PROMPT_TEMPLATE

    def test_contains_context_placeholder(self):
        assert "{context}" in _RAG_PROMPT_TEMPLATE

    def test_contains_prompt_placeholder(self):
        assert "{prompt}" in _RAG_PROMPT_TEMPLATE


# ── build_rag_prompt ─────────────────────────────────────────────────────────

class TestBuildRagPrompt:
    def test_mode_instructions_injected_correctly(self):
        """Verify each mode's instruction appears in the built prompt."""
        for mode, instruction in _MODE_INSTRUCTIONS.items():
            prompt = build_rag_prompt("test query", "context text", mode=mode)
            assert instruction in prompt, f"Mode '{mode}' instruction not found in prompt"

    def test_no_mode_uses_default_instruction(self):
        prompt = build_rag_prompt("test query", "context text", mode=None)
        # Should still produce a valid prompt (no KeyError, no missing placeholders)
        assert "test query" in prompt
        assert "context text" in prompt

    def test_unknown_mode_uses_default_instruction(self):
        prompt = build_rag_prompt("test query", "context text", mode="nonexistent")
        assert "test query" in prompt
        assert "context text" in prompt

    def test_summary_mode_injected(self):
        prompt = build_rag_prompt("what is this?", "doc content", mode="summary")
        assert _MODE_INSTRUCTIONS["summary"] in prompt

    def test_explain_mode_injected(self):
        prompt = build_rag_prompt("what is this?", "doc content", mode="explain")
        assert _MODE_INSTRUCTIONS["explain"] in prompt

    def test_commands_mode_injected(self):
        prompt = build_rag_prompt("how to do X?", "doc content", mode="commands")
        assert _MODE_INSTRUCTIONS["commands"] in prompt

    def test_analyze_mode_injected(self):
        prompt = build_rag_prompt("analyze this", "doc content", mode="analyze")
        assert _MODE_INSTRUCTIONS["analyze"] in prompt


# ── parse_mode ───────────────────────────────────────────────────────────────

class TestParseMode:
    def test_no_prefix_returns_none_and_unchanged_query(self):
        mode, query = parse_mode("what is retrieval augmented generation?")
        assert mode is None
        assert query == "what is retrieval augmented generation?"

    def test_summary_prefix_extracted(self):
        mode, query = parse_mode("/summary what are the main points?")
        assert mode == "summary"
        assert query == "what are the main points?"

    def test_explain_prefix_extracted(self):
        mode, query = parse_mode("/explain how does FAISS work?")
        assert mode == "explain"
        assert query == "how does FAISS work?"

    def test_commands_prefix_extracted(self):
        mode, query = parse_mode("/commands how do I run the server?")
        assert mode == "commands"
        assert query == "how do I run the server?"

    def test_analyze_prefix_extracted(self):
        mode, query = parse_mode("/analyze the architecture")
        assert mode == "analyze"
        assert query == "the architecture"

    def test_case_insensitive_prefix(self):
        mode, query = parse_mode("/SUMMARY what are the main points?")
        assert mode == "summary"
        assert query == "what are the main points?"

    def test_unknown_prefix_not_consumed(self):
        mode, query = parse_mode("/unknown what is this?")
        assert mode is None
        assert query == "/unknown what is this?"

    def test_empty_string(self):
        mode, query = parse_mode("")
        assert mode is None
        assert query == ""

    def test_prefix_only_no_query_text(self):
        mode, query = parse_mode("/summary ")
        assert mode == "summary"
        assert query == ""


# ── truncate_context ─────────────────────────────────────────────────────────

class TestTruncateContext:
    def test_numbers_chunks_starting_at_1(self):
        chunks = ["first chunk", "second chunk", "third chunk"]
        result = truncate_context(chunks)
        assert "[1] first chunk" in result
        assert "[2] second chunk" in result
        assert "[3] third chunk" in result

    def test_sequential_numbering(self):
        """Verify index order matches source order for citation → sources mapping."""
        chunks = ["alpha", "beta", "gamma"]
        result = truncate_context(chunks)
        lines = [part.strip() for part in result.split("\n\n") if part.strip()]
        for i, (expected_prefix, chunk) in enumerate(zip(["[1]", "[2]", "[3]"], chunks), 1):
            assert lines[i - 1].startswith(expected_prefix)
            assert chunk in lines[i - 1]

    def test_single_chunk(self):
        result = truncate_context(["only chunk"])
        assert "[1] only chunk" in result

    def test_empty_list(self):
        result = truncate_context([])
        assert result == ""

    def test_respects_char_limit(self):
        from src.infrastructure.llm.prompts import MAX_CONTEXT_CHARS
        big_chunk = "x" * (MAX_CONTEXT_CHARS + 100)
        result = truncate_context([big_chunk, "should not appear"])
        assert "should not appear" not in result

    def test_chunks_joined_with_double_newline(self):
        result = truncate_context(["a", "b"])
        assert "\n\n" in result


