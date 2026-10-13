"""RFC2217 serial port tests."""

import asyncio
import sys

import pytest

if sys.platform == "emscripten":
    pytest.skip(
        "RFC2217 transport isn't available under Pyodide",
        allow_module_level=True,
    )

from serialx import Serial, SerialException, create_serial_connection
from serialx.common import PortSettingsUpdate, UnsupportedSetting, measure_time
from serialx.platforms.serial_rfc2217 import Rfc2217
from serialx.platforms.serial_rfc2217.types import (
    FlowcontrolResumeCmd,
    FlowcontrolSuspendCmd,
    LineStateFlag,
    ModemStateFlag,
    NotifyLinestateCmd,
    NotifyModemstateCmd,
    Rfc2217CmdId,
    SignatureCmd,
    TelnetOption,
    WillCmd,
)

from .common import HUB4COM_BINARY, SerialBackend, SerialPair, create_hub4com_pair
from .rfc2217_server import create_rfc2217_server
from .socket_relay import create_accept_then_close_server, create_silent_server


def test_unknown_telnet_option() -> None:
    """Test that unknown telnet option codes are accepted."""
    opt = TelnetOption(99)
    assert opt.value == 99
    assert "UNKNOWN" in opt.name


def test_will_cmd_roundtrip() -> None:
    """Test WillCmd to_bytes/from_bytes round-trip."""
    cmd = WillCmd(option=TelnetOption.BINARY)
    assert WillCmd.from_bytes(cmd.to_bytes()) == cmd


def test_signature_roundtrip() -> None:
    """Test SignatureCmd to_bytes/from_bytes round-trip."""
    cmd = SignatureCmd(signature=b"serialx")
    assert cmd.to_bytes() == b"serialx"
    assert SignatureCmd.from_bytes(b"serialx") == cmd


def test_notify_linestate_roundtrip() -> None:
    """Test NotifyLinestateCmd to_bytes/from_bytes round-trip."""
    cmd = NotifyLinestateCmd(linestate=LineStateFlag(0x1E))
    assert NotifyLinestateCmd.from_bytes(cmd.to_bytes()) == cmd


def test_notify_modemstate_roundtrip() -> None:
    """Test NotifyModemstateCmd to_bytes/from_bytes round-trip."""
    cmd = NotifyModemstateCmd(modemstate=ModemStateFlag(0xB0))
    assert NotifyModemstateCmd.from_bytes(cmd.to_bytes()) == cmd


def test_flowcontrol_suspend_roundtrip() -> None:
    """Test FlowcontrolSuspendCmd to_bytes/from_bytes round-trip."""
    cmd = FlowcontrolSuspendCmd()
    assert cmd.to_bytes() == b""
    assert FlowcontrolSuspendCmd.from_bytes(b"") == cmd


def test_flowcontrol_resume_roundtrip() -> None:
    """Test FlowcontrolResumeCmd to_bytes/from_bytes round-trip."""
    cmd = FlowcontrolResumeCmd()
    assert cmd.to_bytes() == b""
    assert FlowcontrolResumeCmd.from_bytes(b"") == cmd


def test_invalid_byte_size_rejected_before_sending() -> None:
    """An invalid byte size fails locally instead of waiting for a server ack."""
    engine = Rfc2217()
    with pytest.raises(UnsupportedSetting):
        list(engine.build_port_config_commands(PortSettingsUpdate(byte_size=123)))


def test_sync_negotiate_timeout_silent_server() -> None:
    """Sync RFC2217 negotiation times out when the server never responds."""
    with create_silent_server() as addr:
        with measure_time() as elapsed:
            with pytest.raises(TimeoutError):
                with Serial.from_url(
                    f"rfc2217://{addr}", baudrate=115200, connect_timeout=0.1
                ):
                    pass

        assert 0.09 <= elapsed() < 1.1


async def test_async_negotiate_timeout_silent_server() -> None:
    """Async RFC2217 negotiation times out when the server never responds."""
    with create_silent_server() as addr:
        with measure_time() as elapsed:
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await create_serial_connection(
                    asyncio.get_running_loop(),
                    asyncio.Protocol,
                    url=f"rfc2217://{addr}",
                    baudrate=115200,
                    connect_timeout=0.1,
                )

        assert 0.1 <= elapsed() < 1.0


async def test_async_data_before_connection_made_is_delivered() -> None:
    """Serial data arriving while the port is configured is delivered after connect."""
    events: list[str | bytes] = []
    received = asyncio.Event()

    class Protocol(asyncio.Protocol):
        def connection_made(self, transport: asyncio.BaseTransport) -> None:
            events.append("connection_made")

        def data_received(self, data: bytes) -> None:
            events.append(data)
            if b"".join(e for e in events if isinstance(e, bytes)) == b"early":
                received.set()

    # The server sends serial data in the same segment as a port configuration ack
    with create_rfc2217_server(
        after_ack={Rfc2217CmdId.SET_CONTROL: b"early"}
    ) as server:
        transport, _ = await create_serial_connection(
            asyncio.get_running_loop(), Protocol, url=server.url, baudrate=115200
        )

        try:
            await asyncio.wait_for(received.wait(), timeout=2)
            assert events[0] == "connection_made"
        finally:
            transport.close()
            await transport.wait_closed()


# Serial data sent along with two different port configuration acks
EARLY_DATA: dict[int, bytes] = {
    Rfc2217CmdId.SET_BAUDRATE: b"first",
    Rfc2217CmdId.SET_CONTROL: b"early",
}


async def test_async_data_before_connection_made_honors_pause() -> None:
    """Buffered early data waits for `resume_reading()` if `connection_made` paused."""
    received = bytearray()

    class Protocol(asyncio.Protocol):
        def connection_made(self, transport: asyncio.BaseTransport) -> None:
            assert isinstance(transport, asyncio.ReadTransport)
            transport.pause_reading()

        def data_received(self, data: bytes) -> None:
            received.extend(data)

    with create_rfc2217_server(after_ack=EARLY_DATA) as server:
        transport, _ = await create_serial_connection(
            asyncio.get_running_loop(), Protocol, url=server.url, baudrate=115200
        )

        try:
            await asyncio.sleep(0.1)
            assert received == b""

            transport.resume_reading()

            while len(received) < len(b"firstearly"):
                await asyncio.sleep(0.01)

            assert received == b"firstearly"
        finally:
            transport.close()
            await transport.wait_closed()


@pytest.mark.parametrize("method", ["close", "abort"])
async def test_async_data_before_connection_made_dropped_on_close(
    method: str,
) -> None:
    """Buffered early data isn't delivered once `connection_made` closed the port."""
    events: list[str | bytes] = []

    class Protocol(asyncio.Protocol):
        def connection_made(self, transport: asyncio.BaseTransport) -> None:
            events.append("connection_made")
            getattr(transport, method)()

        def data_received(self, data: bytes) -> None:
            events.append(data)

        def connection_lost(self, exc: Exception | None) -> None:
            events.append("connection_lost")

    with create_rfc2217_server(after_ack=EARLY_DATA) as server:
        transport, _ = await create_serial_connection(
            asyncio.get_running_loop(), Protocol, url=server.url, baudrate=115200
        )
        await transport.wait_closed()

    assert events == ["connection_made", "connection_lost"]


@pytest.mark.parametrize("paused", [False, True])
async def test_async_data_before_connection_made_protocol_error(paused: bool) -> None:
    """A `data_received()` error while replaying early data fails the transport."""
    loop = asyncio.get_running_loop()
    lost: asyncio.Future[Exception | None] = loop.create_future()
    errors: list[dict[str, object]] = []

    class Protocol(asyncio.Protocol):
        def connection_made(self, transport: asyncio.BaseTransport) -> None:
            if paused:
                assert isinstance(transport, asyncio.ReadTransport)
                transport.pause_reading()

        def data_received(self, data: bytes) -> None:
            raise RuntimeError("protocol error")

        def connection_lost(self, exc: Exception | None) -> None:
            lost.set_result(exc)

    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))

    try:
        with create_rfc2217_server(after_ack=EARLY_DATA) as server:
            transport, _ = await create_serial_connection(
                loop, Protocol, url=server.url, baudrate=115200
            )

            if paused:
                transport.resume_reading()

            exc = await asyncio.wait_for(lost, timeout=2)
            await transport.wait_closed()
    finally:
        loop.set_exception_handler(old_handler)

    assert isinstance(exc, RuntimeError)
    assert [context["exception"] for context in errors] == [exc]


def test_sync_peer_close_during_negotiation_raises() -> None:
    """Peer-side TCP close surfaces as OSError, not a silent 0-byte read."""
    with create_accept_then_close_server() as addr:
        with pytest.raises(OSError, match="RFC 2217 connection closed by server"):
            with Serial.from_url(
                f"rfc2217://{addr}", baudrate=115200, connect_timeout=2.0
            ):
                pass


@pytest.mark.skipif(not HUB4COM_BINARY, reason="hub4com not available")
def test_sync_negotiate_comport_rejected_by_hub4com(serial_pair: SerialPair) -> None:
    """Sync RFC2217 negotiation fails when the server responds with `Wont`."""  # codespell:ignore wont
    if serial_pair.backends != (SerialBackend.ADAPTER,):
        pytest.skip("Requires a bare adapter pair")

    with create_hub4com_pair(serial_pair.left, serial_pair.right, comport="no") as (
        rfc_left,
        _rfc_right,
    ):
        with pytest.raises(SerialException, match="COM-PORT-OPTION"):
            with Serial.from_url(rfc_left, baudrate=115200, connect_timeout=1.0):
                pass


@pytest.mark.skipif(not HUB4COM_BINARY, reason="hub4com not available")
async def test_async_negotiate_comport_rejected_by_hub4com(
    serial_pair: SerialPair,
) -> None:
    """Async RFC2217 negotiation fails when the server responds with `Wont`."""  # codespell:ignore wont
    if serial_pair.backends != (SerialBackend.ADAPTER,):
        pytest.skip("Requires a bare adapter pair")

    with create_hub4com_pair(serial_pair.left, serial_pair.right, comport="no") as (
        rfc_left,
        _rfc_right,
    ):
        with pytest.raises(SerialException, match="COM-PORT-OPTION"):
            await create_serial_connection(
                asyncio.get_running_loop(),
                asyncio.Protocol,
                url=rfc_left,
                baudrate=115200,
                connect_timeout=1.0,
            )
