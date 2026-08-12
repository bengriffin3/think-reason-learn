"""Drop-in `LLM` replacement for RRM that requests + returns token logprobs
via OpenAI's `chat.completions` endpoint.

Why: TRL's canonical `OpenAILLM` wrapper uses the newer `/v1/responses`
endpoint and hardcodes `logprobs=[]` on the returned `LLMResponse`. RRM
depends on `response.logprobs` (for rule-perplexity filtering) and
`response.average_confidence` (for the 3-vote ensemble), so it doesn't
work end-to-end through the canonical wrapper. This shim bypasses that
by:

  * calling `client.chat.completions.create(..., logprobs=True,
    top_logprobs=N)` for `response_format=str`,
  * calling `client.beta.chat.completions.parse(..., logprobs=True,
    response_format=<Pydantic>)` for structured outputs,
  * populating the returned `LLMResponse.logprobs` from the token stream.

Routes to whatever endpoint the openai SDK is pointed at — set
`OPENAI_BASE_URL=http://localhost:11434/v1` for local Ollama.

Pass as `ReasonedRuleMining(_llm=LogprobsLLM())`.

Compatibility scope: single-provider (OpenAI-compat). No fallback to
Google/Anthropic — this is a local-first shim. RRM only makes ONE
`respond(...)` call at a time, so single-provider is fine.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, List, Type, TypeVar
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.core.llms._schemas import LLMResponse

logger = logging.getLogger(__name__)
T = TypeVar("T")


class LogprobsLLM:
    """Drop-in for TRL's `LLM` singleton — routes through chat.completions
    with logprobs enabled. Only implements the async `respond(...)` path
    that RRM actually uses.

    Per-call disk cache: if `cache_path` is set, every successful response
    is appended to a JSONL file keyed by a stable hash of the request. On
    subsequent runs the cache is loaded on __init__ and cache hits skip
    the LLM entirely — so a killed multi-hour run resumes from disk with
    zero LLM cost for the parts already done."""

    def __init__(self, top_logprobs: int = 5, cache_path: str | Path | None = None) -> None:
        self._client = AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        )
        self._top_logprobs = top_logprobs
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        self._cache_lock = asyncio.Lock()
        if self._cache_path and self._cache_path.exists():
            n = 0
            for line in self._cache_path.open():
                try:
                    r = json.loads(line)
                    self._cache[r["_key"]] = r
                    n += 1
                except Exception:
                    continue
            logger.info(f"LogprobsLLM: loaded {n} cached responses from {self._cache_path}")

    def _key(self, model: str, messages: list, temperature: float,
             response_format: Any) -> str:
        rf = "str" if not (isinstance(response_format, type) and issubclass(response_format, BaseModel)) else response_format.__name__ + ":" + response_format.model_json_schema().__str__()
        payload = json.dumps({
            "model": model, "messages": messages,
            "t": round(float(temperature), 6), "rf": rf,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def respond(
        self,
        query: str,
        llm_priority: List[Any],
        response_format: Type[T],
        instructions: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse[T]:
        llmp = llm_priority[0]
        provider = llmp["provider"] if isinstance(llmp, dict) else getattr(llmp, "provider", None)
        model = llmp["model"] if isinstance(llmp, dict) else getattr(llmp, "model", None)
        if provider != "openai":
            raise RuntimeError(
                f"LogprobsLLM only supports openai-compat providers, got {llmp!r}")

        messages: List[Any] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": query})

        eff_temp = 1e-6 if (temperature is None or temperature == 0.0) else float(temperature)

        # Structured output → use beta.chat.completions.parse
        wants_pydantic = isinstance(response_format, type) and issubclass(response_format, BaseModel)

        # Cache lookup
        key = self._key(model, messages, eff_temp, response_format) if self._cache_path else None
        if key and key in self._cache:
            c = self._cache[key]
            parsed = response_format.model_validate_json(c["parsed_json"]) if wants_pydantic else c["parsed_text"]
            return LLMResponse(
                response=parsed,
                logprobs=[tuple(lp) for lp in c["logprobs"]],
                total_tokens=c.get("total_tokens"),
                provider_model=OpenAIChoice(model=model),
            )

        try:
            if wants_pydantic:
                resp = await self._client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=eff_temp,
                    logprobs=True,
                    top_logprobs=self._top_logprobs,
                )
                choice = resp.choices[0]
                parsed = choice.message.parsed
            else:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=eff_temp,
                    logprobs=True,
                    top_logprobs=self._top_logprobs,
                )
                choice = resp.choices[0]
                parsed = choice.message.content or ""
        except Exception as e:
            logger.warning(f"LogprobsLLM call failed: {e}", exc_info=True)
            raise

        logprobs_list: List[tuple[str, float | None]] = []
        if choice.logprobs and choice.logprobs.content:
            for tok in choice.logprobs.content:
                logprobs_list.append((tok.token, tok.logprob))

        total_tokens = resp.usage.total_tokens if resp.usage else None

        # Cache write
        if key is not None:
            entry = {
                "_key": key,
                "logprobs": logprobs_list,
                "total_tokens": total_tokens,
            }
            if wants_pydantic:
                entry["parsed_json"] = parsed.model_dump_json()
            else:
                entry["parsed_text"] = parsed
            async with self._cache_lock:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self._cache_path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                self._cache[key] = entry

        return LLMResponse(
            response=parsed,
            logprobs=logprobs_list,
            total_tokens=total_tokens,
            provider_model=OpenAIChoice(model=model),
        )
