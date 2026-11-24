"""Magic Home / Zengge (LEDENET) Wi-Fi lamp control that works with a full-tunnel VPN running.

Why the VPN matters: AmneziaVPN (and many WireGuard clients) install their own route for the
local subnet (e.g. 192.168.0.0/24) with metric 0, so ordinary sockets send LAN traffic into the
tunnel and the lamp is unreachable. Windows uses the strong host model for sending, so binding a
socket to the PC's *LAN* IPv4 address forces the packets out through the physical/LAN adapter.
Every socket here is bound that way; no routes or VPN settings are touched.
"""
from __future__ import annotations

import ipaddress
import json
import os
import socket
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import psutil

PORT = 5577
DISCOVERY_PORT = 48899
DISCOVERY_MSG = b"HF-A11ASSISTHREAD"
CACHE_FILE = Path(__file__).with_name("lamp_cache.json")

# Interface names that belong to tunnels; never use them as the LAN source.
VPN_HINTS = ("vpn", "wireguard", "amnezia", "wg", "tun", "tap", "openvpn", "outline", "nord", "proton")

# Colour temperature range assumed for the lamp's warm/cold white LEDs.
KELVIN_MIN, KELVIN_MAX = 2700, 6500


# ----------------------------------------------------------------------------- protocol helpers

def checksum(payload: list[int] | bytes) -> bytes:
    b = bytes(payload)
    return b + bytes([sum(b) & 0xFF])


def cmd_state() -> bytes:
    return checksum([0x81, 0x8A, 0x8B])


def cmd_power(on: bool) -> bytes:
    return checksum([0x71, 0x23 if on else 0x24, 0x0F])


PERSIST, NO_PERSIST = 0x0F, 0xF0   # last flag byte: 0x0F saves to flash, 0xF0 is temporary


def cmd_rgb(r: int, g: int, b: int, persist: bool = True) -> bytes:
    # 9-byte LEDENET format. NOTE: the 8-byte variant blanks this lamp (model 0x1E), never use it.
    # Use persist=False for high-rate effects so the lamp's flash is not rewritten constantly.
    return checksum([0x31, _b(r), _b(g), _b(b), 0, 0, 0xF0, PERSIST if persist else NO_PERSIST])


def cmd_white(warm: int, cold: int, persist: bool = True) -> bytes:
    return checksum([0x31, 0, 0, 0, _b(warm), _b(cold), 0x0F, PERSIST if persist else NO_PERSIST])


def _b(v: int) -> int:
    return max(0, min(255, int(round(v))))


def kelvin_to_white(kelvin: int, brightness_pct: float) -> tuple[int, int]:
    """Split a colour temperature into warm/cold channel values.

    The lamp caps warm+cold at 255 (it scales both down proportionally otherwise), so 100 %
    brightness means warm+cold == 255.
    """
    k = max(KELVIN_MIN, min(KELVIN_MAX, kelvin))
    frac_cold = (k - KELVIN_MIN) / (KELVIN_MAX - KELVIN_MIN)
    total = _b(255 * max(0.0, min(100.0, brightness_pct)) / 100)
    cold = _b(total * frac_cold)
    return total - cold, cold


def white_to_kelvin(warm: int, cold: int) -> int | None:
    if warm + cold == 0:
        return None
    return int(round(KELVIN_MIN + (KELVIN_MAX - KELVIN_MIN) * cold / (warm + cold)))


@dataclass
class LampState:
    power: bool
    mode: str            # "rgb" | "white" | "other"
    rgb: tuple[int, int, int]
    warm: int
    cold: int
    brightness_pct: int
    kelvin: int | None
    model: str
    raw: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_state(d: bytes) -> LampState:
    if len(d) < 14 or d[0] != 0x81:
        raise ValueError(f"unexpected state reply: {d.hex(' ')}")
    rgb = (d[6], d[7], d[8])
    warm, cold = d[9], d[11]
    mask = d[12]
    if mask == 0xF0:
        mode, bright = "rgb", max(rgb) / 255
    elif mask == 0x0F:
        mode, bright = "white", min(1.0, (warm + cold) / 255)
    else:
        mode, bright = "other", max(max(rgb), warm + cold) / 255
    return LampState(
        power=d[2] == 0x23, mode=mode, rgb=rgb, warm=warm, cold=cold,
        brightness_pct=int(round(min(1.0, bright) * 100)),
        kelvin=white_to_kelvin(warm, cold) if mode == "white" else None,
        model=f"0x{d[1]:02x}", raw=d.hex(" "),
    )


# ----------------------------------------------------------------------------- network / VPN

@dataclass
class LanInterface:
    name: str
    ip: str
    network: ipaddress.IPv4Network


def lan_interfaces() -> list[LanInterface]:
    """Up, non-VPN, non-link-local IPv4 interfaces with a real subnet."""
    stats = psutil.net_if_stats()
    out = []
    for name, addrs in psutil.net_if_addrs().items():
        if any(h in name.lower() for h in VPN_HINTS):
            continue
        if name in stats and not stats[name].isup:
            continue
        for a in addrs:
            if a.family != socket.AF_INET or not a.netmask:
                continue
            ip = ipaddress.IPv4Address(a.address)
            if ip.is_loopback or ip.is_link_local:
                continue
            net = ipaddress.IPv4Network(f"{a.address}/{a.netmask}", strict=False)
            if net.prefixlen >= 31:      # point-to-point tunnel style address
                continue
            out.append(LanInterface(name, a.address, net))
    return out


def source_ip_for(target_ip: str) -> LanInterface:
    override = os.environ.get("LAMP_SOURCE_IP")
    ifaces = lan_interfaces()
    if override:
        for i in ifaces:
            if i.ip == override:
                return i
        return LanInterface("override", override, ipaddress.IPv4Network(f"{override}/32"))
    t = ipaddress.IPv4Address(target_ip)
    for i in ifaces:
        if t in i.network:
            return i
    raise ConnectionError(
        f"No local (non-VPN) interface is in the same subnet as {target_ip}. "
        f"LAN interfaces: {[(i.name, str(i.network)) for i in ifaces]}"
    )


def discover(timeout: float = 2.0) -> list[dict]:
    """Broadcast the Magic Home discovery probe on every LAN interface (bound, so it bypasses VPN)."""
    found: dict[str, dict] = {}
    for iface in lan_interfaces():
        if not iface.network.is_private:
            continue
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.bind((iface.ip, 0))
            s.settimeout(0.25)
            s.sendto(DISCOVERY_MSG, (str(iface.network.broadcast_address), DISCOVERY_PORT))
            end = time.time() + timeout
            while time.time() < end:
                try:
                    data, addr = s.recvfrom(1024)
                except (socket.timeout, ConnectionResetError):
                    continue
                parts = data.decode(errors="replace").strip().split(",")
                if len(parts) >= 3 and parts[1] != "":
                    found[parts[1].upper()] = {
                        "ip": parts[0], "mac": parts[1].upper(), "model": parts[2],
                        "interface": iface.name, "source_ip": iface.ip,
                    }
        except OSError:
            pass
        finally:
            s.close()
    return list(found.values())


# ----------------------------------------------------------------------------- persistent link

# Measured on this lamp (2026-11-24): one persistent connection handles 10 cmd/s, 20 cmd/s freezes
# it for seconds; opening a fresh TCP connection per command exhausts its slots after ~5 connects.
# The lamp fades to each new value itself in ~0.3-0.6 s, so > ~5 cmd/s brings nothing visible.
SAFE_RATE_HZ = 5.0
MAX_RATE_HZ = 10.0


class LampLink:
    """One long-lived TCP connection: frame parsing, automatic reconnect, rate limiting."""

    def __init__(self, host: str, rate_hz: float = SAFE_RATE_HZ):
        self.host = host
        self.min_interval = 1.0 / max(0.5, min(MAX_RATE_HZ, rate_hz))
        self._sock: socket.socket | None = None
        self._buf = b""
        self._last_send = 0.0
        self._lock = threading.Lock()

    # -- connection
    def _connect(self) -> socket.socket:
        if self._sock is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(4)
            s.bind((source_ip_for(self.host).ip, 0))      # LAN source address -> bypasses the VPN
            s.connect((self.host, PORT))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock, self._buf = s, b""
        return self._sock

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

    # -- incoming frames: 0x81 state replies and 0xB0B1B2B3 wrapped push notifications
    def _pop_state_frame(self) -> bytes | None:
        while self._buf:
            if self._buf.startswith(b"\xb0\xb1\xb2\xb3"):
                if len(self._buf) < 10:
                    return None
                n = 10 + int.from_bytes(self._buf[8:10], "big") + 1
                if len(self._buf) < n:
                    return None
                self._buf = self._buf[n:]
            elif self._buf[0] == 0x81:
                if len(self._buf) < 14:
                    return None
                f, self._buf = self._buf[:14], self._buf[14:]
                if sum(f[:13]) & 0xFF == f[13]:
                    return f
            else:
                self._buf = self._buf[1:]
        return None

    def _drain(self, s: socket.socket) -> None:
        s.setblocking(False)
        try:
            while True:
                chunk = s.recv(1024)
                if not chunk:
                    raise ConnectionError("lamp closed the connection")
                self._buf += chunk
        except (BlockingIOError, InterruptedError):
            pass
        finally:
            s.setblocking(True)
            s.settimeout(4)

    # -- outgoing
    def _wait_rate(self) -> None:
        dt = time.monotonic() - self._last_send
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)

    def _send_locked(self, msg: bytes, attempts: int) -> socket.socket:
        last: Exception | None = None
        for i in range(attempts):
            try:
                s = self._connect()
                self._drain(s)                 # keep the receive buffer from filling with pushes
                self._wait_rate()
                s.sendall(msg)
                self._last_send = time.monotonic()
                return s
            except (OSError, ConnectionError) as e:
                last = e
                self._drop()
                time.sleep(1.0 + i)            # a frozen lamp recovers in a few seconds
        raise ConnectionError(f"lamp {self.host} unreachable: {last}")

    def send(self, msg: bytes, attempts: int = 4) -> None:
        with self._lock:
            self._send_locked(msg, attempts)

    def try_send(self, msg: bytes) -> bool:
        """Non-blocking for effects: skip the frame instead of waiting if we're over the rate."""
        with self._lock:
            if time.monotonic() - self._last_send < self.min_interval:
                return False
            try:
                self._send_locked(msg, attempts=1)
                return True
            except ConnectionError:
                return False

    def state(self, attempts: int = 3) -> LampState:
        with self._lock:
            for i in range(attempts):
                s = self._send_locked(cmd_state(), 2)
                end = time.monotonic() + 2.5
                try:
                    while time.monotonic() < end:
                        f = self._pop_state_frame()
                        if f:
                            return parse_state(f)
                        chunk = s.recv(1024)
                        if not chunk:
                            raise ConnectionError("closed")
                        self._buf += chunk
                except (OSError, ConnectionError):
                    self._drop()
            raise ConnectionError(f"no state reply from {self.host}")


# ----------------------------------------------------------------------------- lamp client

class Lamp:
    def __init__(self, host: str | None = None, mac: str | None = None):
        cache = self._load_cache()
        self.mac = (mac or os.environ.get("LAMP_MAC") or cache.get("mac") or "").upper().replace(":", "").replace("-", "") or None
        self.host = host or cache.get("ip") or os.environ.get("LAMP_HOST")
        self._link: LampLink | None = None
        self._link_lock = threading.Lock()

    # -- cache
    @staticmethod
    def _load_cache() -> dict:
        try:
            return json.loads(CACHE_FILE.read_text("utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        try:
            CACHE_FILE.write_text(json.dumps({"ip": self.host, "mac": self.mac}), "utf-8")
        except OSError:
            pass

    # -- locating the lamp
    def rediscover(self) -> dict:
        lamps = discover()
        pick = None
        if self.mac:
            pick = next((l for l in lamps if l["mac"] == self.mac), None)
        elif len(lamps) == 1:
            pick = lamps[0]
        if not pick:
            hint = " Several lamps found: set LAMP_MAC to choose one." if len(lamps) > 1 and not self.mac else ""
            raise ConnectionError(f"Lamp not found by discovery (mac={self.mac}); seen: {lamps}.{hint}")
        self.host, self.mac = pick["ip"], pick["mac"]
        self._save_cache()
        return pick

    # -- transport: one shared persistent connection (see LampLink for why)
    @property
    def link(self) -> LampLink:
        with self._link_lock:
            if not self.host:
                self.rediscover()
            if self._link is None or self._link.host != self.host:
                if self._link:
                    self._link.close()
                self._link = LampLink(self.host)
            return self._link

    def _with_rediscovery(self, op):
        try:
            return op(self.link)
        except ConnectionError:
            # maybe the lamp got a new IP from DHCP: find it again by MAC and retry once
            old = self.host
            self.rediscover()
            if self.host == old:
                raise
            return op(self.link)

    def send(self, msg: bytes) -> None:
        self._with_rediscovery(lambda l: l.send(msg))

    def close(self) -> None:
        with self._link_lock:
            if self._link:
                self._link.close()
                self._link = None

    # -- high level
    def state(self) -> LampState:
        return self._with_rediscovery(lambda l: l.state())

    def _apply_and_verify(self, msg: bytes, check, timeout: float = 4.0) -> tuple[LampState, bool]:
        """The lamp fades to a new value over ~0.3-0.6 s, so poll its state until it matches."""
        self.send(msg)
        end = time.monotonic() + timeout
        st = None
        while time.monotonic() < end:
            time.sleep(0.4)
            st = self.state()
            if check(st):
                return st, True
        return st or self.state(), False

    def power(self, on: bool) -> tuple[LampState, bool]:
        return self._apply_and_verify(cmd_power(on), lambda s: s.power == on)

    def set_rgb(self, r: int, g: int, b: int) -> tuple[LampState, bool]:
        if not self.state().power:
            self.send(cmd_power(True))
        target = (_b(r), _b(g), _b(b))
        return self._apply_and_verify(
            cmd_rgb(*target),
            lambda s: s.mode == "rgb" and all(abs(x - y) <= 5 for x, y in zip(s.rgb, target)))

    def set_white(self, kelvin: int, brightness_pct: float) -> tuple[LampState, bool]:
        if not self.state().power:
            self.send(cmd_power(True))
        warm, cold = kelvin_to_white(kelvin, brightness_pct)
        return self._apply_and_verify(
            cmd_white(warm, cold),
            lambda s: s.mode == "white" and abs(s.warm - warm) <= 5 and abs(s.cold - cold) <= 5)

    def set_brightness(self, pct: float) -> tuple[LampState, bool]:
        st = self.state()
        pct = max(1.0, min(100.0, pct))
        if st.mode == "white":
            return self.set_white(st.kelvin or 4000, pct)
        peak = max(st.rgb) or 1
        k = (255 * pct / 100) / peak
        return self.set_rgb(*(c * k for c in st.rgb))


# ----------------------------------------------------------------------------- CLI

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Magic Home lamp helper (VPN-safe)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover", help="find Magic Home lamps on the LAN")
    sub.add_parser("network", help="show LAN interfaces used to bypass the VPN")
    p = sub.add_parser("state", help="read a lamp's state")
    p.add_argument("--host", help="lamp IP (default: LAMP_HOST / cache / discovery)")
    a = ap.parse_args()
    if a.cmd == "discover":
        lamps = discover()
        if not lamps:
            print("No lamps found. Is the PC on the same Wi-Fi/LAN as the lamp?")
        for l in lamps:
            print(f"ip={l['ip']}  mac={l['mac']}  model={l['model']}  via {l['interface']} ({l['source_ip']})")
    elif a.cmd == "network":
        for i in lan_interfaces():
            print(f"{i.name:40} {i.ip:16} {i.network}")
    else:
        lamp = Lamp(host=a.host)
        print(json.dumps({"lamp_ip": lamp.host, **lamp.state().to_dict()}, ensure_ascii=False, indent=2))
        lamp.close()


if __name__ == "__main__":
    _cli()
