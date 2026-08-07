"""Tests for OpenAILLM chat-completions (local / OpenAI-compatible server) support.

All tests run offline: the OpenAI SDK client methods are replaced with mocks,
so no Ollama or OpenAI account is needed.
"""

import pytest

from think_reason_learn.core._singleton import SingletonMeta
from think_reason_learn.core.llms._openai.ask import OpenAILLM, get_openai_llm


@pytest.fixture(autouse=True)
def _isolate_openai_singleton(monkeypatch):
    """OpenAILLM is a singleton; reset it and scrub env so tests are hermetic."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    SingletonMeta._instances.pop(OpenAILLM, None)
    yield
    SingletonMeta._instances.pop(OpenAILLM, None)


def test_defaults_to_responses_endpoint_without_base_url():
    llm = OpenAILLM(api_key="sk-test")
    assert llm.endpoint_style == "responses"


def test_auto_detects_chat_completions_when_base_url_set():
    llm = OpenAILLM(api_key="ollama", base_url="http://localhost:11434/v1")
    assert llm.endpoint_style == "chat_completions"


def test_explicit_endpoint_style_beats_auto_detection():
    llm = OpenAILLM(
        api_key="sk-test",
        base_url="http://localhost:8000/v1",
        endpoint_style="responses",
    )
    assert llm.endpoint_style == "responses"


def test_base_url_env_var_enables_chat_completions(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    llm = OpenAILLM(api_key="ollama")
    assert llm.endpoint_style == "chat_completions"
    assert "11434" in str(llm.client.base_url)


def test_get_openai_llm_defaults_api_key_for_local_servers():
    llm = get_openai_llm("", base_url="http://localhost:11434/v1")
    assert llm is not None
    assert llm.client.api_key == "ollama"


def test_get_openai_llm_still_none_without_key_or_base_url():
    assert get_openai_llm("") is None
