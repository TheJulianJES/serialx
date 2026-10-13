"""Pyodide serial port tests."""

import asyncio
import errno
import sys

import pytest

if sys.platform != "emscripten":
    pytest.skip(
        "PyodideSerialTransport is only available under Pyodide",
        allow_module_level=True,
    )

from typing import Any

from serialx import Parity, StopBits, create_serial_connection, open_serial_connection
from serialx.common import SerialException, UnsupportedSetting
from tests.common import create_pyodide_pair


async def test_pyodide_one_point_five_stopbits_unsupported() -> None:
    """1.5 stop bits is not supported."""
    with create_pyodide_pair() as (left, _):
        with pytest.raises(UnsupportedSetting, match="stopbits"):
            await create_serial_connection(
                asyncio.get_running_loop(),
                asyncio.Protocol,
                url=left,
                baudrate=115200,
                stopbits=StopBits.ONE_POINT_FIVE,
            )


async def test_pyodide_mark_parity_unsupported() -> None:
    """Mark parity is not supported."""
    with create_pyodide_pair() as (left, _):
        with pytest.raises(UnsupportedSetting, match="parity"):
            await create_serial_connection(
                asyncio.get_running_loop(),
                asyncio.Protocol,
                url=left,
                baudrate=115200,
                parity=Parity.MARK,
            )


async def test_pyodide_invalid_byte_size_unsupported() -> None:
    """Byte sizes other than 7 or 8 are not supported."""
    with create_pyodide_pair() as (left, _):
        with pytest.raises(UnsupportedSetting, match="byte_size"):
            await create_serial_connection(
                asyncio.get_running_loop(),
                asyncio.Protocol,
                url=left,
                baudrate=115200,
                byte_size=5,
            )


async def test_pyodide_unregistered_port_raises() -> None:
    """Connecting to a URL with no registered JS port raises."""
    with pytest.raises(SerialException, match="No JS serial port registered"):
        await create_serial_connection(
            asyncio.get_running_loop(),
            asyncio.Protocol,
            url="pyodide://not-registered",
            baudrate=115200,
        )


class _FailingWriter:
    """Stand-in for a `WritableStreamDefaultWriter` whose writes fail."""

    def __init__(self, writer: Any) -> None:
        self._writer = writer

    async def write(self, data: Any) -> None:
        raise RuntimeError("write failed")

    def releaseLock(self) -> None:  # noqa: N802
        self._writer.releaseLock()


async def test_pyodide_flush_after_write_error() -> None:
    """`flush` raises the write error instead of waiting for unwritten data."""
    with create_pyodide_pair() as (left, _):
        transport, _ = await create_serial_connection(
            asyncio.get_running_loop(), asyncio.Protocol, url=left, baudrate=115200
        )
        transport._js_writer = _FailingWriter(transport._js_writer)

        for _ in range(3):
            transport.write(b"x" * 10)

        with pytest.raises(OSError, match="write failed") as exc_info:
            await asyncio.wait_for(transport.flush(), 1)
        assert exc_info.value.errno == errno.EIO

        await asyncio.wait_for(transport.wait_closed(), 1)
        assert transport.get_write_buffer_size() == 0

        # The transport stays broken
        with pytest.raises(OSError, match="write failed"):
            await asyncio.wait_for(transport.flush(), 1)


async def test_pyodide_reconfigure_after_drain_timeout() -> None:
    """Writes still go out after a reconfigure had to discard stalled writes."""
    with create_pyodide_pair() as (left, right):
        right_reader, right_writer = await open_serial_connection(
            url=right, baudrate=115200
        )
        # Hold the left side's CTS low so its writes stall
        await right_writer.transport.set_modem_pins(rts=False)

        transport, _ = await create_serial_connection(
            asyncio.get_running_loop(),
            asyncio.Protocol,
            url=left,
            baudrate=115200,
            rtscts=True,
            close_timeout=0.1,
        )

        try:
            transport.write(b"stalled")
            transport.write(b"discarded")
            await transport.reconfigure_port(baudrate=9600)

            await right_writer.transport.set_modem_pins(rts=True)
            transport.write(b"after")
            await asyncio.wait_for(transport.flush(), 1)
            assert await asyncio.wait_for(right_reader.readexactly(5), 1) == b"after"
        finally:
            transport.close()
            right_writer.close()
            await asyncio.wait_for(transport.wait_closed(), 1)
            await asyncio.wait_for(right_writer.wait_closed(), 1)
