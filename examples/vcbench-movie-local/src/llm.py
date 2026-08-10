"""LLM factory for the example. Bridges pre-#81 (shim) and post-#81 (library).

Every runner and notebook in this example builds its LLM here, so swapping the
local-Ollama plumbing is a one-file change.

TRL's `OpenAILLM` calls OpenAI's `/v1/responses` endpoint, which Ollama does not
serve, and hardcodes empty `logprobs` — which RRM needs. `logprobs_llm_shim.py`
works around both. The library-side fix is PR #81; until it merges and this repo
depends on a release that includes it, the import below fails and we fall back to
the shim.

Once #81 is in: drop `logprobs_llm_shim.py` and the fallback in `_base_llm`.

    from src.llm import get_local_llm

    llm = get_local_llm(cache_path="results/my_run/llm_cache.jsonl")
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from .disk_cache import DiskCache

OLLAMA_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")

try:
    from think_reason_learn.core.llms import OpenAILLM  # type: ignore[attr-defined]
except ImportError:  # pre-#81: not exported
    OpenAILLM = None


def _base_llm() -> Any:
    """The uncached LLM: the library's if it can talk to Ollama, else the shim."""
    if OpenAILLM is not None:
        try:
            return OpenAILLM(base_url=OLLAMA_BASE_URL)
        except TypeError:  # pre-#81: no base_url parameter
            pass
    from .logprobs_llm_shim import LogprobsLLM

    # The shim builds its own AsyncOpenAI from OPENAI_BASE_URL. Without this it
    # silently falls back to api.openai.com instead of Ollama. `setdefault`
    # keeps an explicit environment setting authoritative.
    os.environ.setdefault("OPENAI_BASE_URL", OLLAMA_BASE_URL)
    return LogprobsLLM()


def get_local_llm(
    cache_path: str | Path | None = None, *, cache_sampled: bool = False
) -> Any:
    """Build the local-Ollama LLM, optionally wrapped in a restart-safe cache.

    Args:
        cache_path: JSONL cache file. Pass one for any run long enough that you
            would not want to redo it from scratch; omit for one-off calls.
        cache_sampled: Cache `temperature > 0` calls too. Off by default because
            a request-hash cache collapses repeated draws into one response —
            see `disk_cache.py`. `scripts/run_rrm.py` turns it on: RRM reasons
            at temperature 1.0 during fit, and without this an interrupted fit
            restarts from zero.
    """
    llm = _base_llm()
    return DiskCache(llm, cache_path, cache_sampled=cache_sampled) if cache_path else llm
