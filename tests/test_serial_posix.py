"""POSIX serial port tests."""

import sys

import pytest

if sys.platform in ("win32", "emscripten"):
    pytest.skip("POSIX-only tests", allow_module_level=True)

import array
import asyncio
from collections.abc import AsyncGenerator
import contextlib
import errno
import os
import termios
import threading
import time
from typing import Any
from unittest.mock import patch

from serialx import ModemPins, PinState, create_serial_connection
from serialx.descriptor_transport import _safe_close
from serialx.platforms.serial_posix import PosixSerial, PosixSerialTransport


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


class FlowProtocol(asyncio.Protocol):
    """Protocol that records write flow control."""

    def __init__(self) -> None:
        """Initialize the protocol."""
        self.paused = False
        self.resume_writing_calls = 0

    def pause_writing(self) -> None:
        """Record that writing was paused."""
        self.paused = True

    def resume_writing(self) -> None:
        """Record that writing was resumed."""
        self.paused = False
        self.resume_writing_calls += 1


PtyTransport = tuple[PosixSerialTransport, FlowProtocol, int]


@pytest.fixture
async def pty_transport() -> AsyncGenerator[PtyTransport]:
    """Async transport opened on the slave side of a fresh PTY."""
    master, slave = os.openpty()
    os.set_blocking(master, False)

    transport, protocol = await create_serial_connection(
        asyncio.get_running_loop(), FlowProtocol, os.ttyname(slave), 115200
    )
    assert isinstance(transport, PosixSerialTransport)
    assert isinstance(protocol, FlowProtocol)

    try:
        yield transport, protocol, master
    finally:
        transport.abort()

        # Closing the slave blocks until the kernel's output queue drains
        closed = asyncio.ensure_future(transport.wait_closed())
        while not closed.done():
            _drain_pty_master(master)
            await asyncio.sleep(0.001)

        os.close(slave)
        os.close(master)


def _drain_pty_master(master: int) -> None:
    """Read and discard everything the PTY master has buffered."""
    with contextlib.suppress(BlockingIOError):
        while os.read(master, 65536):
            pass


async def _drain_pty_until_empty(transport: PosixSerialTransport, master: int) -> None:
    """Drain the PTY master until the transport's write buffer is empty."""
    for _ in range(5000):
        if transport.get_write_buffer_size() == 0:
            return
        _drain_pty_master(master)
        await asyncio.sleep(0.001)

    pytest.fail("Write buffer did not drain")


async def test_async_flush_concurrent(pty_transport: PtyTransport) -> None:
    """Concurrent `flush()` calls all complete once the write buffer drains."""
    transport, _, master = pty_transport
    transport.write(b"x" * 200_000)
    assert transport.get_write_buffer_size() > 0

    with patch.object(transport._serial, "flush") as mock_tcdrain:
        flushes = [asyncio.create_task(transport.flush()) for _ in range(3)]
        await asyncio.sleep(0.05)
        assert not any(flush.done() for flush in flushes)

        await _drain_pty_until_empty(transport, master)
        await asyncio.wait_for(asyncio.gather(*flushes), 5)

    assert mock_tcdrain.call_count == 3


async def test_async_flush_write_during_tcdrain(pty_transport: PtyTransport) -> None:
    """Data written and drained while `flush()` waits for `tcdrain` is not lost."""
    transport, protocol, master = pty_transport
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _, ctx: loop_errors.append(ctx))

    in_tcdrain = threading.Event()
    release_tcdrain = threading.Event()

    def slow_tcdrain() -> None:
        in_tcdrain.set()
        release_tcdrain.wait(5)

    transport.write(b"x" * 200_000)

    with patch.object(transport._serial, "flush", side_effect=slow_tcdrain):
        flush = asyncio.create_task(transport.flush())
        await _drain_pty_until_empty(transport, master)
        assert await loop.run_in_executor(None, in_tcdrain.wait, 5)
        protocol.resume_writing_calls = 0

        # The protocol is paused by this write and resumed once it drains
        transport.write(b"y" * 200_000)
        assert protocol.paused
        await _drain_pty_until_empty(transport, master)
        assert not protocol.paused
        assert protocol.resume_writing_calls == 1

        release_tcdrain.set()
        await asyncio.wait_for(flush, 5)

    assert not loop_errors

    # The transport still closes gracefully
    transport.write(b"z" * 200_000)
    transport.close()
    await _drain_pty_until_empty(transport, master)
    await asyncio.wait_for(transport.wait_closed(), 5)


async def test_async_flush_abort(pty_transport: PtyTransport) -> None:
    """`abort()` while `flush()` waits for the write buffer to drain fails it."""
    transport, _, _ = pty_transport
    transport.write(b"x" * 200_000)

    with patch.object(transport._serial, "flush") as mock_tcdrain:
        flush = asyncio.create_task(transport.flush())
        await asyncio.sleep(0.05)
        assert not flush.done()

        # The buffered data is discarded, so the flush did not succeed
        transport.abort()
        with pytest.raises(ConnectionAbortedError):
            await asyncio.wait_for(flush, 5)

        await transport.wait_closed()

        # Nothing is left to flush once closed
        await asyncio.wait_for(transport.flush(), 5)

    # The background close owns the file descriptor, it must not be drained
    mock_tcdrain.assert_not_called()


async def test_async_flush_graceful_close(pty_transport: PtyTransport) -> None:
    """`flush()` pending during a graceful `close()` returns once the port is closed."""
    transport, _, master = pty_transport
    transport.write(b"x" * 200_000)

    def slow_close(fd: int) -> None:
        # Drivers wait for the output to drain when the port is closed
        time.sleep(0.2)
        _safe_close(fd)

    with (
        patch.object(transport._serial, "flush") as mock_tcdrain,
        patch("serialx.descriptor_transport._safe_close", side_effect=slow_close),
    ):
        flush = asyncio.create_task(transport.flush())
        await asyncio.sleep(0.05)
        assert not flush.done()

        transport.close()
        await _drain_pty_until_empty(transport, master)

        # The write buffer is empty but the port is still closing
        await asyncio.sleep(0.05)
        assert not flush.done()

        await asyncio.wait_for(flush, 5)
        assert transport._closed_waiter.done()

    # The background close owns the file descriptor, it must not be drained
    mock_tcdrain.assert_not_called()


async def test_async_flush_cancelled(pty_transport: PtyTransport) -> None:
    """Cancelled `flush()` calls do not leave their waiters behind."""
    transport, _, master = pty_transport
    transport.write(b"x" * 200_000)

    for _ in range(10):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(transport.flush(), 0.001)

    assert not transport._empty_waiters

    with patch.object(transport._serial, "flush"):
        flush = asyncio.create_task(transport.flush())
        await _drain_pty_until_empty(transport, master)
        await asyncio.wait_for(flush, 5)


async def test_async_flush_fatal_error(pty_transport: PtyTransport) -> None:
    """A fatal error while `flush()` waits for the write buffer fails it."""
    transport, _, _ = pty_transport
    transport.write(b"x" * 200_000)

    flush = asyncio.create_task(transport.flush())
    await asyncio.sleep(0.05)
    assert not flush.done()

    transport._fatal_error(OSError(errno.EIO, "device disconnected"))

    with pytest.raises(OSError) as raised:
        await asyncio.wait_for(flush, 5)

    assert raised.value.errno == errno.EIO
