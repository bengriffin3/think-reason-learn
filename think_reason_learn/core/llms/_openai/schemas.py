from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel
from openai.types import ChatModel
from openai._types import NOT_GIVEN as NOT_GIVEN, NotGiven as NotGiven

OpenAIChatModel: TypeAlias = str | ChatModel

OpenAIEndpointStyle: TypeAlias = Literal["responses", "chat_completions"]
"""Which OpenAI API surface to call.

``"responses"`` targets ``/v1/responses`` (OpenAI's hosted API).
``"chat_completions"`` targets ``/v1/chat/completions``, the surface
implemented by OpenAI-compatible servers such as Ollama and vLLM.
"""


class OpenAIChoice(BaseModel):
    """An LLM from OpenAI."""

    provider: Literal["openai"] = "openai"
    model: OpenAIChatModel


class OpenAIChoiceDict(TypedDict):
    """An LLM from OpenAI."""

    provider: Literal["openai"]
    model: OpenAIChatModel
