"""Restart-safe per-call disk cache for any TRL-compatible LLM.

Lifted out of `logprobs_llm_shim.LogprobsLLM` so the same caching applies
whichever LLM the example ends up using — the shim today, the library's
`OpenAILLM` once PR #81 lands. `llm.py` is the factory that wires it up.

A local qwen-14b pass over VCBench or Movie is a multi-hour run against a
process that can die. Every successful response is appended to a JSONL file
keyed by a stable hash of the request, and reloaded on construction, so a
restarted run replays finished calls from disk at zero LLM cost.

Sampled calls are the exception: with `temperature > 0` the same request is
*meant* to return different completions, and a request-hash cache would
collapse N draws into one identical response — silently flattening RRM's
ensemble vote. Those calls bypass the cache unless you opt in with
`cache_sampled=True` (which buys restart-safety for sampled stages at the cost
of that guarantee).

The key derivation and the on-disk record layout are unchanged from the shim,
so cache files written by earlier runs stay valid.
"""
from __future__ import annotations
import asyncio, hashlib, json, logging
from pathlib import Path
from typing import Any, List, Tuple

from pydantic import BaseModel

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.core.llms._schemas import LLMResponse

logger = logging.getLogger(__name__)


class DiskCache:
    """Wrap an LLM with a JSONL disk cache over its async `respond(...)`.

    Targets the TRL `LLM.respond` contract — `(query, llm_priority,
    response_format, instructions, temperature)` — which is what every method
    in this library calls. Any other attribute access falls through to the
    wrapped object.

    Args:
        llm: The LLM to wrap. Must expose an async `respond(...)`.
        cache_path: JSONL file to read on construction and append to.
        cache_sampled: Also cache `temperature > 0` calls. Off by default;
            see the module docstring for why.
    """

    def __init__(
        self,
        llm: Any,
        cache_path: str | Path,
        *,
        cache_sampled: bool = False,
    ) -> None:
        self._llm = llm
        self._cache_path = Path(cache_path)
        self._cache_sampled = cache_sampled
        self._cache: dict[str, dict] = {}
        self._cache_lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self._cache_path.exists():
            return
        n = 0
        for line in self._cache_path.open():
            try:
                record = json.loads(line)
                self._cache[record["_key"]] = record
                n += 1
            except Exception:
                continue
        logger.info(f"DiskCache: loaded {n} cached responses from {self._cache_path}")

    @staticmethod
    def _key(model: Any, messages: list, temperature: float, response_format: Any) -> str:
        """Stable request hash. Byte-compatible with the shim's own `_key`."""
        rf = "str" if not _is_pydantic(response_format) else response_format.__name__ + ":" + response_format.model_json_schema().__str__()
        payload = json.dumps({
            "model": model, "messages": messages,
            "t": round(float(temperature), 6), "rf": rf,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def respond(
        self,
        query: str = "",
        llm_priority: List[Any] | None = None,
        response_format: Any = str,
        instructions: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        call: dict[str, Any] = {"query": query, "response_format": response_format, **kwargs}
        for name, value in (
            ("llm_priority", llm_priority),
            ("instructions", instructions),
            ("temperature", temperature),
        ):
            if value is not None:
                call[name] = value

        eff_temp = _effective_temperature(temperature)
        if eff_temp > 0.0 and not self._cache_sampled:
            return await self._llm.respond(**call)

        model = _model_of(llm_priority, kwargs)
        messages: List[Any] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": query})
        key = self._key(model, messages, eff_temp or 1e-6, response_format)

        hit = self._cache.get(key)
        if hit is not None:
            parsed = (
                response_format.model_validate_json(hit["parsed_json"])
                if _is_pydantic(response_format)
                else hit["parsed_text"]
            )
            return LLMResponse(
                response=parsed,
                logprobs=[tuple(lp) for lp in hit.get("logprobs", [])],
                total_tokens=hit.get("total_tokens"),
                provider_model=OpenAIChoice(model=model),
            )

        response = await self._llm.respond(**call)
        if response is None:  # library wrappers swallow errors and return None
            return response
        await self._store(key, response, _is_pydantic(response_format))
        return response

    async def _store(self, key: str, response: Any, wants_pydantic: bool) -> None:
        logprobs: List[Tuple[str, float | None]] = list(getattr(response, "logprobs", None) or [])
        entry: dict[str, Any] = {
            "_key": key,
            "logprobs": [list(lp) for lp in logprobs],
            "total_tokens": getattr(response, "total_tokens", None),
        }
        if wants_pydantic:
            entry["parsed_json"] = response.response.model_dump_json()
        else:
            entry["parsed_text"] = response.response
        async with self._cache_lock:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            self._cache[key] = entry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def _is_pydantic(response_format: Any) -> bool:
    return isinstance(response_format, type) and issubclass(response_format, BaseModel)


def _effective_temperature(temperature: Any) -> float:
    """0.0 for "unset or zero"; the sampling temperature otherwise."""
    if temperature is None or not isinstance(temperature, (int, float)):
        return 0.0
    return float(temperature)


def _model_of(llm_priority: List[Any] | None, kwargs: dict) -> Any:
    """Model id from either an `llm_priority` list or a bare `model=` kwarg."""
    if llm_priority:
        choice = llm_priority[0]
        return choice["model"] if isinstance(choice, dict) else getattr(choice, "model", None)
    return kwargs.get("model")
