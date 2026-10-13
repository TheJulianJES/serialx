"""Compatibility shim for pyserial's ``serial.serialutil`` module."""

from __future__ import annotations

# Only the shim exports this, in `serialx` it is an alias of `TimeoutError`
from serialx.common import SerialTimeoutException
from serialx.serialutil import *  # noqa: F401,F403
from serialx.serialutil import __all__ as _serialutil_all

__all__ = list(_serialutil_all)
__all__ += ["SerialTimeoutException"]
