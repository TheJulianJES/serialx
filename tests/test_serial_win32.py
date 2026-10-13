"""Tests for the Win32 serial port implementation."""

from __future__ import annotations

import errno
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

if sys.platform != "win32":
    pytest.skip("Windows-only tests", allow_module_level=True)

import pywintypes
from win32con import (
    DTR_CONTROL_DISABLE,
    DTR_CONTROL_ENABLE,
    RTS_CONTROL_DISABLE,
    RTS_CONTROL_ENABLE,
)
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


def test_close_tolerates_failing_win32_calls() -> None:
    """Closing a port whose device is gone does not raise."""

    port = Win32Serial()

    # An unplugged device makes `EscapeCommFunction` fail on close. An invalid handle
    # fails it the same way, with a real `pywintypes.error` from pywin32.
    port._handle = INVALID_HANDLE
    port.close()

    assert not port.is_open


@pytest.mark.parametrize(
    ("setting", "pin", "control_attr", "disable", "enable"),
    [
        ("rtscts", "rts", "fRtsControl", RTS_CONTROL_DISABLE, RTS_CONTROL_ENABLE),
        ("dsrdtr", "dtr", "fDtrControl", DTR_CONTROL_DISABLE, DTR_CONTROL_ENABLE),
    ],
)
def test_reconfigure_keeps_modem_pin_levels(
    setting: str, pin: str, control_attr: str, disable: int, enable: int
) -> None:
    """The DCB DTR/RTS control mode follows the last written level, not on-open."""
    port = Win32Serial("COM99", baudrate=115200)
    port._handle = INVALID_HANDLE
    written_dcbs: list[Any] = []

    # As left by opening with the default `dtr_on_open=rts_on_open=HIGH`
    dcb = SimpleNamespace(
        fRtsControl=RTS_CONTROL_ENABLE, fDtrControl=DTR_CONTROL_ENABLE
    )

    with (
        patch.object(serial_win32, "SetCommTimeouts"),
        patch.object(serial_win32, "ClearCommError"),
        patch.object(serial_win32, "EscapeCommFunction"),
        patch.object(serial_win32, "GetCommModemStatus", return_value=0),
        patch.object(
            serial_win32,
            "GetCommState",
            side_effect=lambda handle: SimpleNamespace(**vars(dcb)),
        ),
        patch.object(
            serial_win32,
            "SetCommState",
            side_effect=lambda handle, new_dcb: written_dcbs.append(new_dcb),
        ),
    ):
        port.set_modem_pins(**{pin: False})

        port.reconfigure_port(baudrate=9600)
        assert getattr(written_dcbs[-1], control_attr) == disable

        port.reconfigure_port(**{setting: True})
        port.reconfigure_port(**{setting: False})
        assert getattr(written_dcbs[-1], control_attr) == disable
        assert getattr(port.get_modem_pins(), pin) is PinState.LOW

        port.set_modem_pins(**{pin: True})
        port.reconfigure_port(baudrate=115200)
        assert getattr(written_dcbs[-1], control_attr) == enable

    port._handle = None
