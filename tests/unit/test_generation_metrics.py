import pytest

from evaluation.generation_metrics import (
    FALLBACK_ANSWER,
    JudgeParseError,
    build_judge_prompt,
    compute_faithfulness,
    compute_negative_refusal,
    extract_message_text,
    is_fallback_answer,
    parse_judge_response,
)


# --- extract_message_text ---


def test_extract_message_text_plain_string():
    assert extract_message_text('{"claims": []}') == '{"claims": []}'


def test_extract_message_text_list_of_blocks():
    content = [
        {"type": "text", "text": '{"claims": '},
        {"type": "text", "text": "[]}"},
    ]
    assert extract_message_text(content) == '{"claims": []}'


def test_extract_message_text_skips_non_text_blocks():
    content = [
        {"type": "thought_signature", "thought_signature": "abc"},
        {"type": "text", "text": '{"claims": []}'},
    ]
    assert extract_message_text(content) == '{"claims": []}'


def test_extract_message_text_list_of_strings():
    assert extract_message_text(["a", "b"]) == "ab"


def test_extract_message_text_parses_end_to_end():
    content = [{"type": "text", "text": '{"claims": [{"text": "x", "supported": true}]}'}]
    assert parse_judge_response(extract_message_text(content)) == [
        {"text": "x", "supported": True}
    ]


# --- parse_judge_response ---


def test_parse_judge_response_valid_json():
    raw = '{"claims": [{"text": "The sky is blue", "supported": true}]}'
    result = parse_judge_response(raw)
    assert result == [{"text": "The sky is blue", "supported": True}]


def test_parse_judge_response_markdown_fence():
    raw = '```json\n{"claims": [{"text": "Water boils at 100C", "supported": false}]}\n```'
    result = parse_judge_response(raw)
    assert result == [{"text": "Water boils at 100C", "supported": False}]


def test_parse_judge_response_uppercase_fence():
    raw = '```JSON\n{"claims": [{"text": "x", "supported": true}]}\n```'
    assert parse_judge_response(raw) == [{"text": "x", "supported": True}]


def test_parse_judge_response_reasoning_fence_before_verdict():
    raw = (
        "```\nlet me think about the claims first\n```\n"
        'Here is the verdict:\n```json\n{"claims": [{"text": "x", "supported": false}]}\n```'
    )
    assert parse_judge_response(raw) == [{"text": "x", "supported": False}]


def test_parse_judge_response_json_fence_after_json_reasoning_fence():
    raw = (
        '```json\n{"scratchpad": ["a", "b"]}\n```\n'
        '```json\n{"claims": [{"text": "x", "supported": true}]}\n```'
    )
    assert parse_judge_response(raw) == [{"text": "x", "supported": True}]


def test_parse_judge_response_claim_text_not_str_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"claims": [{"text": 123, "supported": true}]}')


def test_parse_judge_response_malformed_json_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response("not json at all")


def test_parse_judge_response_missing_claims_key_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"facts": []}')


def test_parse_judge_response_claims_not_list_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"claims": "not a list"}')


def test_parse_judge_response_claim_missing_supported_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"claims": [{"text": "something"}]}')


def test_parse_judge_response_supported_not_bool_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"claims": [{"text": "x", "supported": "yes"}]}')


# --- compute_faithfulness ---


def test_compute_faithfulness_all_supported():
    claims = [
        {"text": "a", "supported": True},
        {"text": "b", "supported": True},
    ]
    assert compute_faithfulness(claims) == pytest.approx(1.0)


def test_compute_faithfulness_mixed():
    claims = [
        {"text": "a", "supported": True},
        {"text": "b", "supported": False},
        {"text": "c", "supported": True},
        {"text": "d", "supported": False},
    ]
    assert compute_faithfulness(claims) == pytest.approx(0.5)


def test_compute_faithfulness_empty_list():
    assert compute_faithfulness([]) == pytest.approx(1.0)


# --- compute_negative_refusal ---


def test_compute_negative_refusal_exact_fallback():
    assert compute_negative_refusal(FALLBACK_ANSWER) == pytest.approx(1.0)


def test_compute_negative_refusal_other_answer():
    assert compute_negative_refusal("Here is some answer.") == pytest.approx(0.0)


def test_compute_negative_refusal_fallback_with_extra_text():
    assert compute_negative_refusal(FALLBACK_ANSWER + " Extra text.") == pytest.approx(0.0)


# --- is_fallback_answer ---


def test_is_fallback_answer_exact_match():
    assert is_fallback_answer(FALLBACK_ANSWER) is True


def test_is_fallback_answer_non_match():
    assert is_fallback_answer("Something else entirely") is False


def test_is_fallback_answer_partial_match():
    assert is_fallback_answer("No sufficiently relevant") is False


# --- build_judge_prompt (smoke — pure function) ---


def test_build_judge_prompt_contains_query_answer_contexts():
    prompt = build_judge_prompt(
        query="What is X?",
        answer="X is Y.",
        contexts=["Context one.", "Context two."],
    )
    assert "What is X?" in prompt
    assert "X is Y." in prompt
    assert "Context one." in prompt
    assert "Context two." in prompt
    assert '{"claims"' in prompt or "claims" in prompt
