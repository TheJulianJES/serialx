"""POSIX serial port tests."""

import sys

import pytest

if sys.platform in ("win32", "emscripten"):
    pytest.skip("POSIX-only tests", allow_module_level=True)

import array
import errno
import os
import termios
from typing import Any
from unittest.mock import patch

from serialx import ModemPins, PinState
from serialx.platforms.serial_darwin import IOSSIOSPEED, DarwinSerial
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


@pytest.mark.parametrize("err", [errno.ENOTTY, errno.EINVAL, errno.EOPNOTSUPP])
def test_modem_pins_unsupported_ignored(err: int) -> None:
    """Devices without modem control lines report no modem pins."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        raise OSError(err, os.strerror(err))

    serial = PosixSerial(fileno=1)
    with patch("serialx.platforms.serial_posix.fcntl.ioctl", side_effect=ioctl):
        serial.set_modem_pins(dtr=True, rts=False)
        assert serial._get_modem_pins() == ModemPins()


def test_get_modem_pins_raises_other_errors() -> None:
    """Errors other than "unsupported", e.g. after an unplug, are not swallowed."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        raise OSError(errno.EIO, "Input/output error")

    serial = PosixSerial(fileno=1)
    with patch("serialx.platforms.serial_posix.fcntl.ioctl", side_effect=ioctl):
        with pytest.raises(OSError) as raised:
            serial.get_modem_pins()

    assert raised.value.errno == errno.EIO


@pytest.mark.parametrize(
    "modem_pins",
    [
        # TIOCMBIS/TIOCMBIC
        ModemPins(dtr=PinState.HIGH, rts=PinState.LOW),
        # TIOCMSET
        ModemPins(**dict.fromkeys(ModemPins.__annotations__, PinState.LOW)),
    ],
)
def test_set_modem_pins_raises_other_errors(modem_pins: ModemPins) -> None:
    """Errors other than "unsupported", e.g. after an unplug, are not swallowed."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        raise OSError(errno.EIO, "Input/output error")

    serial = PosixSerial(fileno=1)
    with patch("serialx.platforms.serial_posix.fcntl.ioctl", side_effect=ioctl):
        with pytest.raises(OSError) as raised:
            serial.set_modem_pins(modem_pins)

    assert raised.value.errno == errno.EIO

    # The failed write is not recorded as the current state
    assert serial._dtr_state is PinState.UNDEFINED


def test_darwin_non_posix_baudrate_rejected_raises() -> None:
    """A driver rejecting a custom baudrate fails instead of keeping the fallback."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        assert request == IOSSIOSPEED
        raise OSError(errno.EINVAL, "Invalid argument")

    serial = DarwinSerial(fileno=1, baudrate=3_000_000)
    with patch("serialx.platforms.serial_darwin.fcntl.ioctl", side_effect=ioctl):
        with pytest.raises(OSError) as raised:
            serial._set_non_posix_baudrate(3_000_000)

    assert raised.value.errno == errno.EINVAL


def test_darwin_non_posix_baudrate_not_a_tty_ignored() -> None:
    """Custom baudrates are ignored on devices that are not serial ports."""

    def ioctl(fd: int, request: int, arg: Any = 0, mutate_flag: bool = True) -> int:
        raise OSError(errno.ENOTTY, "Inappropriate ioctl for device")

    serial = DarwinSerial(fileno=1, baudrate=3_000_000)
    with patch("serialx.platforms.serial_darwin.fcntl.ioctl", side_effect=ioctl):
        serial._set_non_posix_baudrate(3_000_000)


def test_termios_error_raised_as_os_error_bad_descriptor() -> None:
    """`termios.error` are re-raised as `OSError` with the errno from the C call."""

    # Pass bad file descriptor directly, _open() would raise ValueError
    serial = PosixSerial(fileno=1)
    with pytest.raises(OSError) as raised:
        serial.reconfigure_port()

    assert raised.value.errno is not None
    assert raised.value.strerror is not None
