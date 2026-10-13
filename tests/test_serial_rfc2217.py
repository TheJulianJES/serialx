"""RFC2217 serial port tests."""

import asyncio
from collections.abc import Iterator
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
from .rfc2217_server import FakeRfc2217Server, create_rfc2217_server
from .socket_relay import create_accept_then_close_server, create_silent_server


@pytest.fixture
def fake_rfc2217_server() -> Iterator[FakeRfc2217Server]:
    """Fake RFC 2217 server for a single client."""
    with create_rfc2217_server() as server:
        yield server


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


def test_sync_open_without_state_mask_acks(
    fake_rfc2217_server: FakeRfc2217Server,
) -> None:
    """Servers that don't ack the state mask commands, like pyserial's, work."""
    fake_rfc2217_server.unacked_cmd_ids = {
        Rfc2217CmdId.SET_LINESTATE_MASK,
        Rfc2217CmdId.SET_MODEMSTATE_MASK,
    }

    with Serial.from_url(
        fake_rfc2217_server.url, baudrate=115200, connect_timeout=5.0
    ) as serial:
        serial.write(b"data")
        serial.flush()


async def test_async_open_without_state_mask_acks(
    fake_rfc2217_server: FakeRfc2217Server,
) -> None:
    """Servers that don't ack the state mask commands, like pyserial's, work."""
    fake_rfc2217_server.unacked_cmd_ids = {
        Rfc2217CmdId.SET_LINESTATE_MASK,
        Rfc2217CmdId.SET_MODEMSTATE_MASK,
    }

    transport, _ = await asyncio.wait_for(
        create_serial_connection(
            asyncio.get_running_loop(),
            asyncio.Protocol,
            url=fake_rfc2217_server.url,
            baudrate=115200,
            connect_timeout=5.0,
        ),
        2,
    )

    transport.write(b"data")
    await asyncio.wait_for(transport.flush(), 2)

    transport.close()
    await transport.wait_closed()
