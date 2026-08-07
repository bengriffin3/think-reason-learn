"""Tests for OpenAILLM chat-completions (local / OpenAI-compatible server) support.

All tests run offline: the OpenAI SDK client methods are replaced with mocks,
so no Ollama or OpenAI account is needed.
"""

import asyncio
import math
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from think_reason_learn.core._singleton import SingletonMeta
from think_reason_learn.core.llms._openai.ask import OpenAILLM, get_openai_llm


class _Verdict(BaseModel):
    answer: str
    confident: bool


def _fake_completion(content=None, parsed=None, logprob_pairs=None, total_tokens=42):
    """Build a chat-completion response shape (SDK objects are attr-access only)."""
    logprobs = None
    if logprob_pairs is not None:
        logprobs = SimpleNamespace(
            content=[SimpleNamespace(token=t, logprob=lp) for t, lp in logprob_pairs]
        )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, parsed=parsed),
        logprobs=logprobs,
    )
    return SimpleNamespace(
        choices=[choice], usage=SimpleNamespace(total_tokens=total_tokens)
    )


def _chat_llm():
    """OpenAILLM pointed at a local server, with fully mocked SDK clients."""
    llm = OpenAILLM(api_key="ollama", base_url="http://localhost:11434/v1")
    llm.client = MagicMock()
    llm.aclient = MagicMock()
    return llm


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


def test_chat_string_response_preserves_logprobs_sync():
    llm = _chat_llm()
    pairs = [("Hello", -0.05), (" world", -0.2)]
    llm.client.chat.completions.create = MagicMock(
        return_value=_fake_completion(content="Hello world", logprob_pairs=pairs)
    )

    result = llm.respond_sync(model="qwen3:8b", query="Say hello", raise_=True)

    assert result is not None
    assert result.response == "Hello world"
    assert result.logprobs == pairs
    assert result.total_tokens == 42
    assert result.provider_model.model == "qwen3:8b"
    call_kwargs = llm.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["logprobs"] is True
    assert call_kwargs["top_logprobs"] == 5
    assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]
    mock_client = cast(MagicMock, llm.client)
    mock_client.responses.create.assert_not_called()
    mock_client.responses.parse.assert_not_called()


def test_chat_instructions_become_system_message_sync():
    llm = _chat_llm()
    llm.client.chat.completions.create = MagicMock(
        return_value=_fake_completion(content="ok", logprob_pairs=[])
    )

    llm.respond_sync(model="qwen3:8b", query="Q", instructions="Be terse", raise_=True)

    messages = llm.client.chat.completions.create.call_args.kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "Be terse"},
        {"role": "user", "content": "Q"},
    ]


def test_chat_structured_output_uses_parse_sync():
    llm = _chat_llm()
    parsed = _Verdict(answer="yes", confident=True)
    llm.client.chat.completions.parse = MagicMock(
        return_value=_fake_completion(parsed=parsed, logprob_pairs=[("yes", -0.1)])
    )

    result = llm.respond_sync(
        model="qwen3:8b", query="Q", response_format=_Verdict, raise_=True
    )

    assert result is not None
    assert result.response is parsed
    assert result.logprobs == [("yes", -0.1)]
    call_kwargs = llm.client.chat.completions.parse.call_args.kwargs
    assert call_kwargs["response_format"] is _Verdict
    assert call_kwargs["logprobs"] is True


def test_chat_missing_logprobs_yields_empty_list_sync():
    llm = _chat_llm()
    llm.client.chat.completions.create = MagicMock(
        return_value=_fake_completion(content="hi", logprob_pairs=None)
    )

    result = llm.respond_sync(model="qwen3:8b", query="Q", raise_=True)

    assert result is not None
    assert result.logprobs == []


def test_chat_temperature_zero_is_preserved_sync():
    llm = _chat_llm()
    llm.client.chat.completions.create = MagicMock(
        return_value=_fake_completion(content="hi", logprob_pairs=[])
    )

    llm.respond_sync(model="qwen3:8b", query="Q", temperature=0.0, raise_=True)

    assert llm.client.chat.completions.create.call_args.kwargs["temperature"] == 0.0


def test_chat_logprobs_drive_average_confidence():
    llm = _chat_llm()
    pairs = [("a", -0.1), ("b", -0.3)]
    llm.client.chat.completions.create = MagicMock(
        return_value=_fake_completion(content="ab", logprob_pairs=pairs)
    )

    result = llm.respond_sync(model="qwen3:8b", query="Q", raise_=True)

    assert result is not None
    assert result.average_confidence == pytest.approx(math.exp(-0.2))


def test_chat_error_returns_none_unless_raise():
    llm = _chat_llm()
    llm.client.chat.completions.create = MagicMock(side_effect=RuntimeError("boom"))

    assert llm.respond_sync(model="qwen3:8b", query="Q") is None
    with pytest.raises(RuntimeError):
        llm.respond_sync(model="qwen3:8b", query="Q", raise_=True)


def test_chat_string_response_async():
    llm = _chat_llm()
    pairs = [("4", -0.01)]
    llm.aclient.chat.completions.create = AsyncMock(
        return_value=_fake_completion(content="4", logprob_pairs=pairs)
    )

    result = asyncio.run(llm.respond(model="qwen3:8b", query="2+2?", raise_=True))

    assert result is not None
    assert result.response == "4"
    assert result.logprobs == pairs


def test_responses_path_still_used_without_base_url():
    """Backward compat: no base_url means the /v1/responses path, unchanged."""
    llm = OpenAILLM(api_key="sk-test")
    llm.client = MagicMock()
    fake = SimpleNamespace(output_text="cloud", usage=SimpleNamespace(total_tokens=3))
    llm.client.responses.create = MagicMock(return_value=fake)

    result = llm.respond_sync(model="gpt-5", query="Q", raise_=True)

    assert result is not None
    assert result.response == "cloud"
    llm.client.responses.create.assert_called_once()
    llm.client.chat.completions.create.assert_not_called()


def test_dict_llm_priority_record_routes_to_chat_completions():
    """Dict provider records (the GPTree trap) are normalized by LLM and
    reach the chat-completions path intact."""
    from think_reason_learn.core.llms._ask import LLM

    trl_llm = LLM()
    orig_openai = trl_llm.openai_llm
    try:
        chat_llm = _chat_llm()
        chat_llm.client.chat.completions.create = MagicMock(
            return_value=_fake_completion(content="local", logprob_pairs=[("l", -0.4)])
        )
        trl_llm.openai_llm = chat_llm

        result = trl_llm.respond_sync(
            query="Q",
            llm_priority=[{"provider": "openai", "model": "qwen3:8b"}],
            response_format=str,
        )

        assert result.response == "local"
        assert result.logprobs == [("l", -0.4)]
        chat_llm.client.chat.completions.create.assert_called_once()
    finally:
        trl_llm.openai_llm = orig_openai


def test_chat_structured_output_async():
    llm = _chat_llm()
    parsed = _Verdict(answer="no", confident=False)
    llm.aclient.chat.completions.parse = AsyncMock(
        return_value=_fake_completion(parsed=parsed, logprob_pairs=[("no", -0.3)])
    )

    result = asyncio.run(
        llm.respond(model="qwen3:8b", query="Q", response_format=_Verdict, raise_=True)
    )

    assert result is not None
    assert result.response is parsed
    assert result.logprobs == [("no", -0.3)]


def test_settings_read_base_url_and_endpoint_style_from_env(monkeypatch, tmp_path):
    from think_reason_learn.core._config import Settings

    monkeypatch.chdir(tmp_path)  # keep a developer .env file out of the picture
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_ENDPOINT_STYLE", "chat_completions")

    fresh = Settings()

    assert fresh.OPENAI_BASE_URL == "http://localhost:11434/v1"
    assert fresh.OPENAI_ENDPOINT_STYLE == "chat_completions"


def test_settings_reject_invalid_endpoint_style(monkeypatch, tmp_path):
    from pydantic import ValidationError

    from think_reason_learn.core._config import Settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_ENDPOINT_STYLE", "grpc")

    with pytest.raises(ValidationError):
        Settings()


def test_llm_wires_openai_base_url_from_settings(monkeypatch):
    from think_reason_learn.core._config import settings
    from think_reason_learn.core.llms._ask import LLM

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "OPENAI_ENDPOINT_STYLE", "")
    SingletonMeta._instances.pop(LLM, None)
    try:
        trl_llm = LLM()
        assert trl_llm.openai_llm is not None
        assert trl_llm.openai_llm.endpoint_style == "chat_completions"
    finally:
        SingletonMeta._instances.pop(LLM, None)
