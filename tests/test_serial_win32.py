"""Tests for the Win32 serial port implementation."""

from __future__ import annotations

from collections.abc import Callable
import errno
import sys

import pytest

if sys.platform != "win32":
    pytest.skip("Windows-only tests", allow_module_level=True)

import pywintypes
from win32file import OVERLAPPED
from winerror import ERROR_INVALID_HANDLE

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


def test_read_and_write_keep_each_others_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read and a write from two threads do not reset each other's timeouts."""

    port = Win32Serial()
    port._handle = INVALID_HANDLE
    port._overlapped_read = OVERLAPPED()
    port._overlapped_write = OVERLAPPED()

    kernel_timeouts: list[tuple[int, ...]] = [()]
    seen_by_io: dict[str, tuple[int, ...]] = {}
    other_thread_io: list[Callable[[], object]] = []

    def set_comm_timeouts(handle: int, timeouts: tuple[int, ...]) -> None:
        kernel_timeouts[0] = tuple(timeouts)

    def reset_event(event: object) -> None:
        # Right before `ReadFile`/`WriteFile`, as a second thread using the port could
        if other_thread_io:
            other_thread_io.pop()()

    def read_file(
        handle: int, buf: bytearray, overlapped: object
    ) -> tuple[int, bytearray]:
        seen_by_io["read"] = kernel_timeouts[0]
        return 0, buf

    def write_file(handle: int, data: bytes, overlapped: object) -> tuple[int, int]:
        seen_by_io["write"] = kernel_timeouts[0]
        return 0, len(data)

    def get_overlapped_result(handle: int, overlapped: object, wait: bool) -> int:
        return 4 if overlapped is port._overlapped_write else 0

    monkeypatch.setattr(serial_win32, "SetCommTimeouts", set_comm_timeouts)
    monkeypatch.setattr(serial_win32, "ResetEvent", reset_event)
    monkeypatch.setattr(serial_win32, "ReadFile", read_file)
    monkeypatch.setattr(serial_win32, "WriteFile", write_file)
    monkeypatch.setattr(serial_win32, "GetOverlappedResult", get_overlapped_result)

    # A write without a timeout must not drop the total timeout of the read
    other_thread_io.append(lambda: port.write(b"data"))
    port.readinto(bytearray(4), timeout=0.5)
    assert seen_by_io["read"] == (10, 0, 500, 0, 0)

    # A read without a timeout must not drop the total timeout of the write
    other_thread_io.append(lambda: port.readinto(bytearray(4)))
    port.write(b"data", timeout=2.0)
    assert seen_by_io["write"] == (10, 0, 0, 0, 2000)
