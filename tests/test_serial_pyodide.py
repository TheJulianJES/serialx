"""Pyodide serial port tests."""

import asyncio
import sys

import pytest

if sys.platform != "emscripten":
    pytest.skip(
        "PyodideSerialTransport is only available under Pyodide",
        allow_module_level=True,
    )

from pyodide.ffi import JsException  # type: ignore[import-not-found]

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


async def test_pyodide_open_failure_raises() -> None:
    """A port that fails to open raises and leaves the JS port alone."""
    with create_pyodide_pair() as (left, right):
        left_reader, left_writer = await open_serial_connection(
            url=left, baudrate=115200
        )
        _, right_writer = await open_serial_connection(url=right, baudrate=115200)

        try:
            # The JS port is already open, so `port.open()` rejects
            with pytest.raises(JsException, match="InvalidStateError"):
                await asyncio.wait_for(
                    create_serial_connection(
                        asyncio.get_running_loop(),
                        asyncio.Protocol,
                        url=left,
                        baudrate=115200,
                    ),
                    timeout=1,
                )

            # The failed connection did not close the port out from under the first
            right_writer.write(b"still open")
            assert await asyncio.wait_for(left_reader.readexactly(10), 1) == (
                b"still open"
            )
        finally:
            left_writer.close()
            right_writer.close()
            await asyncio.wait_for(left_writer.wait_closed(), 1)
            await asyncio.wait_for(right_writer.wait_closed(), 1)


async def test_pyodide_close_after_failed_reconfigure() -> None:
    """Closing does not hang when reopening the port during a reconfigure failed."""
    with create_pyodide_pair() as (left, _):
        transport, _ = await create_serial_connection(
            asyncio.get_running_loop(), asyncio.Protocol, url=left, baudrate=115200
        )

        with pytest.raises(UnsupportedSetting, match="stopbits"):
            await transport.reconfigure_port(stopbits=StopBits.ONE_POINT_FIVE)

        transport.close()
        await asyncio.wait_for(transport.wait_closed(), 1)
