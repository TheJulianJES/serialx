"""Pyodide serial port tests."""

import asyncio
import sys
from typing import Any

import pytest

if sys.platform != "emscripten":
    pytest.skip(
        "PyodideSerialTransport is only available under Pyodide",
        allow_module_level=True,
    )

import js  # type: ignore[import-not-found]
from pyodide.ffi import JsException  # type: ignore[import-not-found]

from serialx import Parity, StopBits, create_serial_connection, open_serial_connection
from serialx.common import SerialException, UnsupportedSetting
from serialx.platforms.serial_pyodide import PyodideSerialTransport
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


class _SlowOpenPort:
    """Wraps a JS port so that `open()` only resolves once `opened` is set."""

    def __init__(self, port: Any) -> None:
        self._port = port
        self.opened = asyncio.Event()

    def open(self, **kwargs: Any) -> asyncio.Future[None]:
        # Like a JS promise, cancelling the caller does not stop the port opening
        return asyncio.shield(self._open(**kwargs))

    async def _open(self, **kwargs: Any) -> None:
        await self.opened.wait()
        await self._port.open(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._port, name)


@pytest.mark.parametrize("how", ["close", "cancel", "cancel twice"])
async def test_pyodide_close_while_opening(how: str) -> None:
    """A port that finishes opening after the transport was closed is closed again."""
    slow_port = _SlowOpenPort(js.create_fake_serial_pair()[0])
    transport = PyodideSerialTransport(asyncio.get_running_loop(), asyncio.Protocol())
    connect_task = asyncio.create_task(
        transport.connect(path="pyodide://slow", baudrate=115200, js_port=slow_port)
    )
    await asyncio.sleep(0.1)

    if how == "close":
        transport.close()
    else:
        connect_task.cancel()
        if how == "cancel twice":
            await asyncio.sleep(0.01)
            connect_task.cancel()
    slow_port.opened.set()

    with pytest.raises((SerialException, asyncio.CancelledError)):
        await asyncio.wait_for(connect_task, 1)
    await asyncio.wait_for(transport.wait_closed(), 1)

    # The port is closed again in the background once it has opened
    await asyncio.sleep(0.1)

    # The JS port was closed again, so a new connection can open it
    _, writer = await asyncio.wait_for(
        open_serial_connection(
            url="pyodide://slow", baudrate=115200, js_port=slow_port
        ),
        1,
    )
    writer.close()
    await asyncio.wait_for(writer.wait_closed(), 1)
