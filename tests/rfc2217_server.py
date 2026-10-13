"""Minimal scriptable RFC 2217 access server for tests."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import socket
import threading

from serialx.platforms.serial_rfc2217.types import TelnetCmdId, TelnetOption

# Server-to-client RFC 2217 command codes are offset by 100
SERVER_CMD_OFFSET = 100


class FakeRfc2217Server:
    """RFC 2217 server that accepts every option and acks every command.

    Received COM-PORT-OPTION commands are recorded in `commands` as
    `(client command code, payload)` and serial data in `data`, both before the
    command is acked, so a client that waited for the ack can inspect them.
    """

    def __init__(self, *, after_ack: dict[int, bytes] | None = None) -> None:
        """Initialize the server.

        `after_ack` maps a client command code to raw bytes that are sent once,
        in the same write as the next ack of that command.
        """
        self.after_ack = dict(after_ack or {})
        self.commands: list[tuple[int, bytes]] = []
        self.data = bytearray()

        self._server = socket.create_server(("127.0.0.1", 0))
        self._conn: socket.socket | None = None
        self._connected = threading.Event()
        self._enabled: set[tuple[int, int]] = set()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def url(self) -> str:
        """Return the `rfc2217://` URL of the server."""
        return f"rfc2217://127.0.0.1:{self._server.getsockname()[1]}"

    def send(self, data: bytes) -> None:
        """Send raw bytes to the connected client."""
        assert self._connected.wait(timeout=5)
        assert self._conn is not None
        self._conn.sendall(data)

    def start(self) -> None:
        """Start serving a single client in a background thread."""
        self._thread.start()

    def close(self) -> None:
        """Stop the server and close the client connection."""
        self._server.close()
        if self._conn is not None:
            with contextlib.suppress(OSError):
                self._conn.shutdown(socket.SHUT_RDWR)
            self._conn.close()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        try:
            conn, _ = self._server.accept()
        except OSError:
            return

        self._conn = conn
        self._connected.set()
        state = "data"
        option_cmd = 0
        subneg = bytearray()

        while True:
            try:
                chunk = conn.recv(4096)
            except OSError:
                return

            if not chunk:
                return

            for byte in chunk:
                if state == "data":
                    if byte == TelnetCmdId.IAC:
                        state = "iac"
                    else:
                        self.data.append(byte)
                elif state == "iac":
                    if byte == TelnetCmdId.IAC:
                        self.data.append(0xFF)
                        state = "data"
                    elif byte == TelnetCmdId.SB:
                        subneg.clear()
                        state = "subneg"
                    elif byte in (
                        TelnetCmdId.WILL,
                        TelnetCmdId.WONT,  # codespell:ignore wont
                        TelnetCmdId.DO,
                        TelnetCmdId.DONT,
                    ):
                        option_cmd = byte
                        state = "option"
                    else:
                        state = "data"
                elif state == "option":
                    self._handle_option(conn, option_cmd, byte)
                    state = "data"
                elif state == "subneg":
                    if byte == TelnetCmdId.IAC:
                        state = "subneg_iac"
                    else:
                        subneg.append(byte)
                elif state == "subneg_iac":
                    if byte == TelnetCmdId.IAC:
                        subneg.append(0xFF)
                        state = "subneg"
                    else:
                        if byte == TelnetCmdId.SE:
                            self._handle_subneg(conn, bytes(subneg))
                        state = "data"

    def _handle_option(self, conn: socket.socket, cmd: int, option: int) -> None:
        # Only answer requests that change an option's state, so that the client
        # re-confirming an option doesn't loop forever
        if cmd not in (TelnetCmdId.WILL, TelnetCmdId.DO):
            return

        if (cmd, option) in self._enabled:
            return

        self._enabled.add((cmd, option))
        reply = TelnetCmdId.DO if cmd == TelnetCmdId.WILL else TelnetCmdId.WILL
        conn.sendall(bytes([TelnetCmdId.IAC, reply, option]))

    def _handle_subneg(self, conn: socket.socket, subneg: bytes) -> None:
        if len(subneg) < 2 or subneg[0] != TelnetOption.COM_PORT_OPTION:
            return

        cmd, payload = subneg[1], subneg[2:]
        self.commands.append((cmd, payload))

        ack = (
            bytes(
                [
                    TelnetCmdId.IAC,
                    TelnetCmdId.SB,
                    TelnetOption.COM_PORT_OPTION,
                    cmd + SERVER_CMD_OFFSET,
                ]
            )
            + payload.replace(b"\xff", b"\xff\xff")
            + bytes([TelnetCmdId.IAC, TelnetCmdId.SE])
        )
        conn.sendall(ack + self.after_ack.pop(cmd, b""))


@contextlib.contextmanager
def create_rfc2217_server(
    *, after_ack: dict[int, bytes] | None = None
) -> Iterator[FakeRfc2217Server]:
    """Create a fake RFC 2217 server for a single client."""
    server = FakeRfc2217Server(after_ack=after_ack)
    server.start()
    try:
        yield server
    finally:
        server.close()
