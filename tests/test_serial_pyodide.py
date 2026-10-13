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

import js  # type: ignore[import-not-found]

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


@pytest.mark.parametrize(
    "error", ["BreakError", "FramingError", "ParityError", "BufferOverrunError"]
)
async def test_pyodide_non_fatal_read_error_keeps_reading(error: str) -> None:
    """Line errors only replace `port.readable`, reading continues on the new one."""
    left_port, right_port = js.create_fake_serial_pair()
    left_reader, left_writer = await open_serial_connection(
        url="pyodide://left", baudrate=115200, js_port=left_port
    )
    _, right_writer = await open_serial_connection(
        url="pyodide://right", baudrate=115200, js_port=right_port
    )

    try:
        right_writer.write(b"before")
        assert await asyncio.wait_for(left_reader.readexactly(6), 1) == b"before"

        left_port.errorReadable(error)
        await asyncio.sleep(0.1)
        assert not left_writer.transport.is_closing()

        right_writer.write(b"after")
        assert await asyncio.wait_for(left_reader.readexactly(5), 1) == b"after"
    finally:
        left_writer.close()
        right_writer.close()
        await left_writer.wait_closed()
        await right_writer.wait_closed()


async def test_pyodide_disconnect_closes_transport() -> None:
    """A fatal read error (unplugged device) closes the transport with an error."""
    left_port, _ = js.create_fake_serial_pair()
    left_reader, left_writer = await open_serial_connection(
        url="pyodide://left", baudrate=115200, js_port=left_port
    )

    left_port.disconnect()
    await asyncio.wait_for(left_writer.transport.wait_closed(), 1)
    assert left_writer.transport.is_closing()

    with pytest.raises(OSError) as exc_info:
        await left_reader.read(1)
    assert exc_info.value.errno == errno.EIO
