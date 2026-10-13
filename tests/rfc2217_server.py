"""Minimal scriptable RFC 2217 access server for tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import select
import socket
import threading

from serialx.platforms.serial_rfc2217.types import TelnetCmdId, TelnetOption

# Server-to-client RFC 2217 command codes are offset by 100
SERVER_CMD_OFFSET = 100

# How often blocking socket calls wake up to check whether the server is stopping
POLL_INTERVAL = 0.1

OPTION_CMDS = frozenset(
    {
        TelnetCmdId.WILL,
        TelnetCmdId.WONT,  # codespell:ignore wont
        TelnetCmdId.DO,
        TelnetCmdId.DONT,
    }
)


def encode_ack(cmd_id: int, payload: bytes) -> bytes:
    """Encode a server response to the client command `cmd_id`."""
    return (
        bytes(
            [
                TelnetCmdId.IAC,
                TelnetCmdId.SB,
                TelnetOption.COM_PORT_OPTION,
                cmd_id + SERVER_CMD_OFFSET,
            ]
        )
        + payload.replace(b"\xff", b"\xff\xff")
        + bytes([TelnetCmdId.IAC, TelnetCmdId.SE])
    )


class FakeRfc2217Server:
    """RFC 2217 server for a single client that accepts every option.

    Every COM-PORT-OPTION command is acked by echoing it back. Received commands
    are recorded in `commands` as `(client command code, payload)` and serial data
    in `data`, both before the command is acked, so a client that waited for the
    ack can inspect them.
    """

    def __init__(self, *, after_ack: Mapping[int, bytes] | None = None) -> None:
        """Initialize the server.

        `after_ack` maps a client command code to raw bytes that are sent once,
        glued to the next ack of that command.
        """
        self.commands: list[tuple[int, bytes]] = []
        self.data = bytearray()
        self.after_ack: dict[int, bytes] = dict(after_ack or {})
        # Commands that are recorded but never acked
        self.unacked_cmd_ids: set[int] = set()
        # Queue acks until `release_acks()` instead of sending them right away
        self.hold_acks = False

        self._held_acks: list[bytes] = []
        self._acks_to_drop: dict[int, int] = {}
        self._enabled_options: set[tuple[int, int]] = set()

        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(POLL_INTERVAL)
        self._conn: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._connected = threading.Event()
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def url(self) -> str:
        """Return the `rfc2217://` URL of the server."""
        return f"rfc2217://127.0.0.1:{self._server.getsockname()[1]}"

    @property
    def num_held_acks(self) -> int:
        """Return the number of acks queued while `hold_acks` is set."""
        with self._send_lock:
            return len(self._held_acks)

    def start(self) -> None:
        """Start serving a single client in a background thread."""
        self._thread.start()

    def close(self) -> None:
        """Stop the server and close the client connection."""
        self._stopping.set()

        with self._send_lock:
            conn = self._conn

        if conn is not None:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)

        if self._thread.ident is not None:
            self._thread.join(timeout=5)

        self._server.close()

        if conn is not None:
            conn.close()

    def wait_connected(self, timeout: float = 5) -> None:
        """Wait until a client has connected."""
        if not self._connected.wait(timeout):
            raise TimeoutError("No RFC 2217 client connected")

    def disconnect_client(self) -> None:
        """Close the connection to the client."""
        with self._send_lock:
            conn = self._conn

        if conn is not None:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)

    def send(self, data: bytes) -> None:
        """Send raw bytes to the client."""
        self.wait_connected()

        with self._send_lock:
            assert self._conn is not None
            self._conn.sendall(data)

    def send_ack(self, cmd_id: int, payload: bytes) -> None:
        """Send a response to the client command `cmd_id` with an arbitrary payload."""
        self.send(encode_ack(cmd_id, payload))

    def release_acks(self, count: int | None = None) -> None:
        """Send the oldest `count` held acks (all by default) in a single write."""
        with self._send_lock:
            num = len(self._held_acks) if count is None else count
            acks, self._held_acks = self._held_acks[:num], self._held_acks[num:]

            if acks:
                assert self._conn is not None
                self._conn.sendall(b"".join(acks))

    def drop_acks(self, cmd_id: int, count: int = 1) -> None:
        """Don't ack the next `count` commands `cmd_id`, they are still recorded."""
        with self._send_lock:
            self._acks_to_drop[cmd_id] = self._acks_to_drop.get(cmd_id, 0) + count

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return

            conn.settimeout(None)

            with self._send_lock:
                if self._stopping.is_set():
                    conn.close()
                    return

                self._conn = conn

            self._connected.set()
            self._serve(conn)
            return

    def _serve(self, conn: socket.socket) -> None:
        state = "data"
        option_cmd = 0
        subneg = bytearray()

        while not self._stopping.is_set():
            try:
                readable, _, _ = select.select([conn], [], [], POLL_INTERVAL)
                if not readable:
                    continue

                chunk = conn.recv(4096)
            except (OSError, ValueError):
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
                    elif byte in OPTION_CMDS:
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
                    elif byte == TelnetCmdId.SE:
                        self._handle_subneg(conn, bytes(subneg))
                        state = "data"
                    else:
                        state = "data"

    def _handle_option(self, conn: socket.socket, cmd: int, option: int) -> None:
        # Only answer requests that change an option's state, so that the client
        # re-confirming an option doesn't loop forever
        if cmd not in (TelnetCmdId.WILL, TelnetCmdId.DO):
            return

        if (cmd, option) in self._enabled_options:
            return

        self._enabled_options.add((cmd, option))
        reply = TelnetCmdId.DO if cmd == TelnetCmdId.WILL else TelnetCmdId.WILL

        with self._send_lock:
            conn.sendall(bytes([TelnetCmdId.IAC, reply, option]))

    def _handle_subneg(self, conn: socket.socket, subneg: bytes) -> None:
        if len(subneg) < 2 or subneg[0] != TelnetOption.COM_PORT_OPTION:
            return

        cmd_id, payload = subneg[1], subneg[2:]
        self.commands.append((cmd_id, payload))

        if cmd_id in self.unacked_cmd_ids:
            return

        with self._send_lock:
            if self._acks_to_drop.get(cmd_id, 0) > 0:
                self._acks_to_drop[cmd_id] -= 1
                return

            ack = encode_ack(cmd_id, payload) + self.after_ack.pop(cmd_id, b"")

            if self.hold_acks:
                self._held_acks.append(ack)
            else:
                conn.sendall(ack)


@contextlib.contextmanager
def create_rfc2217_server(
    *, after_ack: Mapping[int, bytes] | None = None
) -> Iterator[FakeRfc2217Server]:
    """Create a fake RFC 2217 server for a single client."""
    server = FakeRfc2217Server(after_ack=after_ack)
    server.start()
    try:
        yield server
    finally:
        server.close()
