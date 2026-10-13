"""RFC2217 serial port tests."""

import asyncio
import sys
import threading
import time

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


@pytest.mark.parametrize(
    ("subneg", "cmd_id"),
    [
        (
            bytes([0xFF, 0xFA, 44, 105, 20, 0xFF, 0xF0]),
            Rfc2217CmdId.SET_CONTROL,
        ),
        (
            bytes([0xFF, 0xFA, 44, 102, 0xFF, 0xF0]),
            Rfc2217CmdId.SET_DATASIZE,
        ),
    ],
    ids=["set_control_unassigned_value", "set_datasize_empty"],
)
def test_malformed_server_response(subneg: bytes, cmd_id: Rfc2217CmdId) -> None:
    """A malformed response fails its request without losing surrounding data."""
    engine = Rfc2217()

    assert engine.feed(b"AB" + subneg + b"CD") == (b"AB" + b"CD", [])
    assert engine.feed(b"EF") == (b"EF", [])

    with pytest.raises(SerialException, match=cmd_id.name):
        engine.pop_pending_rfc2217(cmd_id)

    # Later responses are still parsed
    baudrate_ack = bytes([0xFF, 0xFA, 44, 101]) + (115200).to_bytes(4, "big")
    assert engine.feed(baudrate_ack + bytes([0xFF, 0xF0])) == (b"", [])
    assert engine.pop_pending_rfc2217(Rfc2217CmdId.SET_BAUDRATE) is not None


def test_sync_malformed_ack_fails_request() -> None:
    """A malformed ack fails its request, the connection keeps working."""
    with create_rfc2217_server() as server:
        with Serial.from_url(server.url, baudrate=115200, read_timeout=2) as serial:
            server.drop_acks(Rfc2217CmdId.SET_CONTROL)
            num_commands = len(server.commands)

            def reply_malformed() -> None:
                deadline = time.monotonic() + 5

                while len(server.commands) == num_commands:
                    assert time.monotonic() < deadline, "request never received"
                    time.sleep(0.01)

                server.send(b"AB")
                server.send_ack(Rfc2217CmdId.SET_CONTROL, bytes([20]))
                server.send(b"CD")

            thread = threading.Thread(target=reply_malformed, daemon=True)
            thread.start()

            with pytest.raises(SerialException, match="SET_CONTROL"):
                serial.set_modem_pins(dtr=True)

            thread.join(timeout=5)
            assert not thread.is_alive()

            serial.set_modem_pins(dtr=False)
            assert serial.read(4) == b"ABCD"


async def test_async_malformed_ack_fails_request() -> None:
    """A malformed ack fails its request, the connection keeps working."""
    received = bytearray()

    class Protocol(asyncio.Protocol):
        def data_received(self, data: bytes) -> None:
            received.extend(data)

    with create_rfc2217_server() as server:
        transport, _ = await create_serial_connection(
            asyncio.get_running_loop(), Protocol, url=server.url, baudrate=115200
        )

        try:
            server.drop_acks(Rfc2217CmdId.SET_CONTROL)
            num_commands = len(server.commands)
            request = asyncio.ensure_future(transport.set_modem_pins(dtr=True))

            async def request_received() -> None:
                while len(server.commands) == num_commands:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(request_received(), timeout=5)

            server.send(b"AB")
            server.send_ack(Rfc2217CmdId.SET_CONTROL, bytes([20]))
            server.send(b"CD")

            with pytest.raises(SerialException, match="SET_CONTROL"):
                await request

            await transport.set_modem_pins(dtr=False)
            assert not transport.is_closing()

            while len(received) < 4:
                await asyncio.sleep(0.01)

            assert received == b"ABCD"
        finally:
            transport.close()
            await transport.wait_closed()


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
