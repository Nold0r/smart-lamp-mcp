"""MCP server for a Magic Home / Zengge Wi-Fi lamp, VPN-safe (see magichome.py)."""
from __future__ import annotations

import threading

import anyio
from mcp.server.mcpserver import MCPServer

import magichome as mh

lamp = mh.Lamp()
_show = None                      # active lightshow.LightShow, if any
_show_lock = threading.Lock()

server = MCPServer(
    name="smart-lamp",
    instructions=(
        "Controls the user's Magic Home (Zengge) Wi-Fi lamp on the home LAN. Works while a "
        "full-tunnel VPN is active by binding sockets to the PC's LAN address. Use set_color for "
        "RGB colours, set_white for white light (kelvin 2700-6500), set_brightness to dim or "
        "brighten keeping the current colour, set_scene for named moods, start_light_music / "
        "stop_light_music for a music-reactive show driven by what the PC is playing. Every "
        "change tool returns the state the lamp itself reports afterwards and whether it matched. "
        "Any colour/white/power/scene change stops a running light show first."
    ),
)

# Named moods, tuned by eye on the real lamp.
SCENES: dict[str, dict] = {
    "rain":     {"rgb": (80, 160, 200),  "desc": "fresh cool blue-teal, like a rainy day"},
    "sunset":   {"rgb": (255, 90, 25),   "desc": "warm orange with a red tint"},
    "golden":   {"rgb": (255, 150, 40),  "desc": "golden hour, lighter and yellower than sunset"},
    "night":    {"rgb": (255, 60, 10),   "desc": "dim deep amber night light", "brightness": 15},
    "daylight": {"white": (5600, 100),   "desc": "neutral-cool daylight white"},
    "morning":  {"white": (5000, 100),   "desc": "soft neutral white, slightly warmer than daylight"},
    "reading":  {"white": (5000, 69),    "desc": "soft neutral white at ~70 %, comfortable for reading"},
    "cozy":     {"white": (2700, 50),    "desc": "warm white at half brightness"},
}


def _report(result: tuple[mh.LampState, bool], stopped_show: bool = False) -> dict:
    st, ok = result
    out = {"verified": ok, "state": st.to_dict(), "lamp_ip": lamp.host}
    if stopped_show:
        out["light_music"] = "stopped (replaced by this change)"
    return out


async def _run(fn, *args):
    return await anyio.to_thread.run_sync(lambda: fn(*args))


def _stop_show(restore: bool) -> dict | None:
    global _show
    with _show_lock:
        show, _show = _show, None
    if show is None:
        return None
    return show.stop(restore=restore)


async def _change(fn, *args) -> dict:
    """Run a lamp change; stop a running light show first (without restoring the old light)."""
    stopped = await _run(_stop_show, False) is not None
    return _report(await _run(fn, *args), stopped)


@server.tool()
async def get_state() -> dict:
    """Read the lamp's current state (power, mode, RGB, white channels, brightness %, kelvin)."""
    st = await _run(lamp.state)
    return {"state": st.to_dict(), "lamp_ip": lamp.host, "light_music_running": _show is not None}


@server.tool()
async def power(on: bool) -> dict:
    """Turn the lamp on (true) or off (false)."""
    return await _change(lamp.power, on)


@server.tool()
async def set_color(red: int, green: int, blue: int, brightness: float = 100) -> dict:
    """Set an RGB colour (each channel 0-255). brightness (1-100 %) scales the colour down."""
    k = max(1.0, min(100.0, brightness)) / 100
    return await _change(lamp.set_rgb, red * k, green * k, blue * k)


@server.tool()
async def set_color_hex(hex_color: str, brightness: float = 100) -> dict:
    """Set an RGB colour from a hex string such as '#50A0C8'. brightness 1-100 %."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError("hex_color must look like #RRGGBB")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return await set_color(r, g, b, brightness)


@server.tool()
async def set_white(kelvin: int = 5000, brightness: float = 100) -> dict:
    """White light from the dedicated white LEDs. kelvin 2700 (warm) .. 6500 (cold), brightness 1-100 %."""
    return await _change(lamp.set_white, kelvin, brightness)


@server.tool()
async def set_brightness(brightness: float) -> dict:
    """Change brightness (1-100 %) while keeping the current colour or white temperature."""
    return await _change(lamp.set_brightness, brightness)


@server.tool()
async def list_scenes() -> dict:
    """List the named scenes accepted by set_scene."""
    return {name: s["desc"] for name, s in SCENES.items()}


@server.tool()
async def set_scene(name: str) -> dict:
    """Apply a named mood: rain, sunset, golden, night, daylight, morning, reading, cozy."""
    s = SCENES.get(name.lower().strip())
    if not s:
        raise ValueError(f"unknown scene {name!r}; choose from {sorted(SCENES)}")
    if "white" in s:
        return await set_white(*s["white"])
    return await set_color(*s["rgb"], brightness=s.get("brightness", 100))


@server.tool()
async def start_light_music(palette: str = "rainbow", rate_hz: float = 5,
                            min_brightness: float = 0.12, device: str | None = None) -> dict:
    """Start a music-reactive light show from the audio the PC is playing (system output loopback,
    not the microphone). palette: rainbow, warm, cool, rain, sunset. rate_hz: lamp commands per
    second (default 5, max 10; the lamp freezes at ~20). min_brightness 0-0.9 for quiet parts.
    device: optional speaker name substring. The light from before the show is restored by
    stop_light_music."""
    import lightshow  # lazy: audio libs only load when needed

    def start():
        global _show
        # restarting (e.g. another palette) must keep the light from before the *first* show
        original = _show.before if _show is not None else None
        _stop_show(restore=False)
        show = lightshow.LightShow(lamp.link, palette, rate_hz, min_brightness, device)
        show.start()
        if original is not None:
            show.before = original
        with _show_lock:
            _show = show
        return show

    show = await _run(start)
    await anyio.sleep(1.5)
    st = show.status()
    if st["error"]:
        await _run(_stop_show, True)
        raise RuntimeError(st["error"])
    return {"started": True, **st}


@server.tool()
async def stop_light_music(restore: bool = True) -> dict:
    """Stop the light show. restore=true puts back the light from before the show."""
    res = await _run(_stop_show, restore)
    if res is None:
        return {"running": False, "note": "no light show was running"}
    return res


@server.tool()
async def light_music_status() -> dict:
    """Stats of the running light show: beats detected, frames sent, current colour, errors."""
    if _show is None:
        return {"running": False}
    return _show.status()


@server.tool()
async def discover_lamps() -> dict:
    """Search the LAN (bypassing the VPN) for Magic Home lamps and re-pin the configured one."""
    found = await _run(mh.discover)
    try:
        pinned = await _run(lamp.rediscover)
    except ConnectionError as e:
        pinned = str(e)
    return {"found": found, "pinned": pinned}


@server.tool()
async def network_info() -> dict:
    """Diagnostics: LAN interfaces considered, and which local address is used to reach the lamp."""
    ifaces = [{"name": i.name, "ip": i.ip, "network": str(i.network)} for i in mh.lan_interfaces()]
    try:
        src = mh.source_ip_for(lamp.host) if lamp.host else None
        src = {"name": src.name, "ip": src.ip} if src else None
    except ConnectionError as e:
        src = str(e)
    return {"lamp_ip": lamp.host, "lamp_mac": lamp.mac, "source": src, "lan_interfaces": ifaces}


if __name__ == "__main__":
    try:
        server.run("stdio")
    finally:
        # Claude Code closed the server: don't leave the lamp on a random show colour
        try:
            _stop_show(restore=True)
        finally:
            lamp.close()
