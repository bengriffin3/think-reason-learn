"""Configuration for tests."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _ensure_main_thread_event_loop():
    """Guarantee a current event loop exists in the main thread for every test.

    pytest-asyncio 1.x closes and clears the current loop after each
    ``@pytest.mark.asyncio`` test, leaving subsequent synchronous tests without
    one. Libraries like ``grpc.aio`` (used by ``xai_sdk``) call
    ``asyncio.get_event_loop()`` during client construction and error out with
    ``RuntimeError: There is no current event loop`` when it returns nothing.

    This autouse fixture restores a current loop before every test if none is
    set. Cheap, side-effect-free, keeps CI ordering-independent.
    """
    try:
        asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield
