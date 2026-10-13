"""POSIX serial port tests."""

import sys

import pytest

if sys.platform in ("win32", "emscripten"):
    pytest.skip("POSIX-only tests", allow_module_level=True)

import array
import asyncio
import contextlib
import errno
import os
import termios
from typing import Any
from unittest.mock import patch

from serialx import ModemPins, PinState, create_serial_connection
from serialx.platforms.serial_posix import PosixSerial


def test_num_unread_bytes_uses_native_int_ioctl_buffer() -> None:
    """FIONREAD writes a native C int, not a little-endian byte string."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        assert request == termios.FIONREAD
        assert isinstance(arg, array.array)
        assert arg.typecode == "i"
        arg[0] = 123
        return 0

    serial = PosixSerial(fileno=1)
    with patch("serialx.platforms.serial_posix.fcntl.ioctl", side_effect=ioctl):
        assert serial.num_unread_bytes() == 123


def test_num_unwritten_bytes_uses_native_int_ioctl_buffer() -> None:
    """TIOCOUTQ writes a native C int, not a little-endian byte string."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        assert request == termios.TIOCOUTQ
        assert isinstance(arg, array.array)
        assert arg.typecode == "i"
        arg[0] = 456
        return 0

    serial = PosixSerial(fileno=1)
    with patch("serialx.platforms.serial_posix.fcntl.ioctl", side_effect=ioctl):
        assert serial.num_unwritten_bytes() == 456


def test_get_modem_pins_uses_native_int_ioctl_buffer() -> None:
    """TIOCMGET writes a native C int, not a little-endian byte string."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        assert request == termios.TIOCMGET
        assert isinstance(arg, array.array)
        assert arg.typecode == "i"
        arg[0] = termios.TIOCM_DTR
        return 0

    serial = PosixSerial(fileno=1)
    with patch("serialx.platforms.serial_posix.fcntl.ioctl", side_effect=ioctl):
        pins = serial.get_modem_pins()

    assert pins.dtr is PinState.HIGH
    assert pins.rts is PinState.LOW


def test_set_modem_pins_uses_native_int_ioctl_buffer() -> None:
    """TIOCMSET reads a native C int, not a little-endian byte string."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        assert request == termios.TIOCMSET
        assert isinstance(arg, array.array)
        assert arg.typecode == "i"
        assert arg[0] & termios.TIOCM_DTR
        assert arg[0] & termios.TIOCM_RTS
        return 0

    serial = PosixSerial(fileno=1)
    with patch("serialx.platforms.serial_posix.fcntl.ioctl", side_effect=ioctl):
        serial.set_modem_pins(
            ModemPins(
                le=PinState.LOW,
                dtr=PinState.HIGH,
                rts=PinState.HIGH,
                st=PinState.LOW,
                sr=PinState.LOW,
                cts=PinState.LOW,
                car=PinState.LOW,
                rng=PinState.LOW,
                dsr=PinState.LOW,
            )
        )


def test_partial_set_modem_pins_uses_native_int_ioctl_buffer() -> None:
    """TIOCMBIS/TIOCMBIC read native C ints, not little-endian byte strings."""

    seen_requests = []

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        assert isinstance(arg, array.array)
        assert arg.typecode == "i"
        seen_requests.append(request)
        if request == termios.TIOCMBIS:
            assert arg[0] == termios.TIOCM_DTR
        elif request == termios.TIOCMBIC:
            assert arg[0] == termios.TIOCM_RTS
        else:
            raise OSError(errno.ENOTTY, "unexpected ioctl")
        return 0

    serial = PosixSerial(fileno=1)
    with patch("serialx.platforms.serial_posix.fcntl.ioctl", side_effect=ioctl):
        serial.set_modem_pins(dtr=True, rts=False)

    assert seen_requests == [termios.TIOCMBIS, termios.TIOCMBIC]


def test_termios_error_raised_as_os_error_bad_descriptor() -> None:
    """`termios.error` are re-raised as `OSError` with the errno from the C call."""

    # Pass bad file descriptor directly, _open() would raise ValueError
    serial = PosixSerial(fileno=1)
    with pytest.raises(OSError) as raised:
        serial.reconfigure_port()

    assert raised.value.errno is not None
    assert raised.value.strerror is not None


class DataProtocol(asyncio.Protocol):
    """Protocol that records received data and connection loss."""

    def __init__(self) -> None:
        """Initialize the protocol."""
        self.received = bytearray()
        self.lost = asyncio.get_running_loop().create_future()

    def data_received(self, data: bytes) -> None:
        """Record received data."""
        self.received += data

    def connection_lost(self, exc: Exception | None) -> None:
        """Record connection loss."""
        self.lost.set_result(exc)


@pytest.mark.parametrize("write_eof", [False, True])
async def test_async_close_stops_reading_while_draining(write_eof: bool) -> None:
    """No data is received after `close()`, even while the write buffer drains."""
    master, slave = os.openpty()
    os.set_blocking(master, False)

    try:
        transport, protocol = await create_serial_connection(
            asyncio.get_running_loop(), DataProtocol, os.ttyname(slave), 115200
        )
        assert isinstance(protocol, DataProtocol)

        # Nothing reads the PTY master, so this stays in the write buffer
        transport.write(b"x" * 200_000)
        assert transport.get_write_buffer_size() > 0

        if write_eof:
            transport.write_eof()

        transport.close()
        os.write(master, b"after close")
        await asyncio.sleep(0.1)

        assert transport.get_write_buffer_size() > 0
        assert not protocol.received

        # Let the write buffer drain so the transport finishes closing
        for _ in range(5000):
            if protocol.lost.done():
                break
            with contextlib.suppress(BlockingIOError):
                while os.read(master, 65536):
                    pass
            await asyncio.sleep(0.001)

        await asyncio.wait_for(transport.wait_closed(), 5)
        assert await protocol.lost is None
        assert not protocol.received
    finally:
        os.close(slave)
        os.close(master)
