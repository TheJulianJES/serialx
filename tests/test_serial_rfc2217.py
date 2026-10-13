"""RFC2217 serial port tests."""

import asyncio
from collections.abc import Iterator
import sys
import threading

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


@pytest.mark.parametrize("read_timeout", [0, 0.01])
def test_sync_commands_ignore_read_timeout(
    fake_rfc2217_server: FakeRfc2217Server, read_timeout: float
) -> None:
    """Commands wait for their acks with the connect timeout, not the read timeout."""
    with Serial.from_url(
        fake_rfc2217_server.url, baudrate=115200, read_timeout=read_timeout
    ) as serial:
        fake_rfc2217_server.hold_acks = True
        stop = threading.Event()

        def release_acks() -> None:
            while not stop.wait(0.1):
                fake_rfc2217_server.release_acks()

        thread = threading.Thread(target=release_acks, daemon=True)
        thread.start()

        try:
            serial.write(b"data")
            serial.flush()
            serial.dtr = False
            serial.rts = True
            serial.reset_read_buffer()
            serial.reset_write_buffer()
        finally:
            stop.set()
            thread.join()


@pytest.mark.parametrize("write_timeout", [None, 1.0])
def test_sync_flush_waits_for_slow_ack(
    fake_rfc2217_server: FakeRfc2217Server, write_timeout: float | None
) -> None:
    """flush() waits for the server to drain to the device, not `connect_timeout`."""
    with Serial.from_url(
        fake_rfc2217_server.url,
        baudrate=115200,
        connect_timeout=0.2,
        write_timeout=write_timeout,
    ) as serial:
        fake_rfc2217_server.hold_acks = True
        serial.write(b"data")

        # The server only acks the flush once the device took the data
        timer = threading.Timer(0.5, fake_rfc2217_server.release_acks)
        timer.start()

        try:
            serial.flush()
        finally:
            timer.join()


def test_sync_flush_honors_write_timeout(
    fake_rfc2217_server: FakeRfc2217Server,
) -> None:
    """flush() gives up after `write_timeout` when the server never acks it."""
    with Serial.from_url(
        fake_rfc2217_server.url, baudrate=115200, write_timeout=0.2
    ) as serial:
        fake_rfc2217_server.unacked_cmd_ids = {Rfc2217CmdId.SET_BAUDRATE}
        serial.write(b"data")

        with measure_time() as elapsed:
            with pytest.raises(TimeoutError):
                serial.flush()

        assert 0.15 <= elapsed() < 1.0


def test_sync_command_timeout_is_a_deadline(
    fake_rfc2217_server: FakeRfc2217Server,
) -> None:
    """Serial data arriving while waiting for an ack doesn't extend the wait."""
    with Serial.from_url(
        fake_rfc2217_server.url, baudrate=115200, connect_timeout=0.3
    ) as serial:
        fake_rfc2217_server.unacked_cmd_ids = {Rfc2217CmdId.SET_CONTROL}
        stop = threading.Event()

        def stream_data() -> None:
            while not stop.wait(0.05):
                fake_rfc2217_server.send(b"x")

        thread = threading.Thread(target=stream_data, daemon=True)
        thread.start()

        try:
            with measure_time() as elapsed:
                with pytest.raises(TimeoutError):
                    serial.dtr = True
        finally:
            stop.set()
            thread.join()

        assert 0.25 <= elapsed() < 1.0
