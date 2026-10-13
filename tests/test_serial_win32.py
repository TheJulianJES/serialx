"""Tests for the Win32 serial port implementation."""

from __future__ import annotations

import errno
import sys

import pytest

if sys.platform != "win32":
    pytest.skip("Windows-only tests", allow_module_level=True)

import pywintypes
from winerror import ERROR_INVALID_HANDLE

from serialx import PinState
from serialx.platforms import _win32api, serial_win32
from serialx.platforms.serial_win32 import Win32Serial

# `INVALID_HANDLE_VALUE`. Every comm call rejects it; `CloseHandle` accepts it.
INVALID_HANDLE = -1


@pytest.mark.parametrize("name", ["GetCommModemStatus", "ClearCommError"])
def test_win32_failure_raises_os_error(name: str) -> None:
    """A real pywin32 failure surfaces as `OSError` with the Win32 code in `winerror`."""
    func = getattr(_win32api, name)

    with pytest.raises(OSError) as raised:
        func(INVALID_HANDLE)

    assert not isinstance(raised.value, pywintypes.error)
    assert raised.value.winerror == ERROR_INVALID_HANDLE
    assert raised.value.errno == errno.EBADF


@pytest.mark.parametrize("wrap_exceptions", [False, True])
def test_close_tolerates_failing_win32_calls(wrap_exceptions: bool) -> None:
    """Closing a port whose device is gone does not raise."""

    # pySerial compatibility mode re-raises errors of public methods as
    # `SerialException`, which must not leak out of the close path
    port = Win32Serial(_wrap_exceptions=wrap_exceptions)

    # An unplugged device makes `EscapeCommFunction` fail on close. An invalid handle
    # fails it the same way, with a real `pywintypes.error` from pywin32.
    port._handle = INVALID_HANDLE
    port.close()

    assert not port.is_open


def test_close_tracks_modem_pins_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DTR/RTS levels set on close are remembered, as Windows can't read them."""
    monkeypatch.setattr(serial_win32, "EscapeCommFunction", lambda handle, func: None)

    port = Win32Serial(dtr_on_close=PinState.HIGH, rts_on_close=PinState.LOW)

    # Set at runtime, before closing
    port._dtr_state = PinState.LOW
    port._rts_state = PinState.HIGH

    port._handle = INVALID_HANDLE
    port.close()

    assert not port.is_open
    assert port._dtr_state is PinState.HIGH
    assert port._rts_state is PinState.LOW
