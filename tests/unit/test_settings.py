import pytest
from pydantic import ValidationError

from src.config.settings import ConfigurationError, Settings


def make_settings(**kwargs) -> Settings:
    """Build Settings without reading from .env, for test isolation."""
    return Settings(_env_file=None, **kwargs)


def test_validate_llm_missing_provider_and_model():
    s = make_settings()
    with pytest.raises(ConfigurationError):
        s.validate_llm()


def test_validate_llm_unknown_provider():
    s = make_settings(llm_provider="openai", llm_model="gpt-4")
    with pytest.raises(ConfigurationError):
        s.validate_llm()


def test_validate_llm_valid_gemini():
    s = make_settings(llm_provider="gemini", llm_model="gemini-2.0-flash")
    s.validate_llm()  # must not raise


def test_validate_llm_valid_ollama():
    s = make_settings(llm_provider="ollama", llm_model="llama3")
    s.validate_llm()  # must not raise


def test_malformed_env_value_raises_validation_error(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "abc")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_without_env_no_exit():
    s = make_settings()
    assert s.llm_provider is None


def test_cors_origins_list_default():
    s = make_settings()
    assert s.cors_origins_list == ["http://localhost:8000", "http://127.0.0.1:8000"]


def test_cors_origins_list_parses_csv():
    s = make_settings(cors_origins="http://a.com , http://b.com,http://c.com")
    assert s.cors_origins_list == ["http://a.com", "http://b.com", "http://c.com"]
