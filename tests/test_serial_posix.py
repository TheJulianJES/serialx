"""POSIX serial port tests."""

import sys

import pytest

if sys.platform in ("win32", "emscripten"):
    pytest.skip("POSIX-only tests", allow_module_level=True)

import array
import errno
import os
import termios
import time
import tty
from typing import Any
from unittest.mock import patch

from serialx import ModemPins, PinState
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


def test_reconfigure_port_keeps_buffers_and_modem_pins() -> None:
    """Only opening applies the on-open modem pins and flushes, not reconfiguring."""
    master, slave = os.openpty()
    tty.setraw(master)

    try:
        with PosixSerial(os.ttyname(slave), baudrate=115200, read_timeout=1) as serial:
            os.write(master, b"received before reconfigure")
            time.sleep(0.1)

            with (
                patch.object(serial, "_set_modem_pins") as set_modem_pins,
                patch("serialx.platforms.serial_posix.tcflush") as tcflush,
            ):
                serial.reconfigure_port(baudrate=9600)

            set_modem_pins.assert_not_called()
            tcflush.assert_not_called()
            assert serial.readexactly(27) == b"received before reconfigure"
    finally:
        os.close(master)
        os.close(slave)


def test_open_sets_modem_pins_then_flushes() -> None:
    """Opening applies settings, then the on-open modem pins, then flushes."""
    master, slave = os.openpty()
    calls: list[tuple[object, ...]] = []

    try:
        serial = PosixSerial(os.ttyname(slave), baudrate=115200)

        with (
            patch.object(
                serial,
                "_reconfigure_port",
                side_effect=lambda update: calls.append(("configure", update)),
            ),
            patch.object(
                serial,
                "_set_modem_pins",
                side_effect=lambda pins: calls.append(("pins", pins.dtr, pins.rts)),
            ),
            patch(
                "serialx.platforms.serial_posix.tcflush",
                side_effect=lambda fd, queue: calls.append(("flush", queue)),
            ),
        ):
            serial.open()

        serial.close()
    finally:
        os.close(master)
        os.close(slave)

    assert calls == [
        ("configure", serial._all_settings()),
        ("pins", PinState.HIGH, PinState.HIGH),
        ("flush", termios.TCIOFLUSH),
    ]
