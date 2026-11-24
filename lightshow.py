"""Light music for the Magic Home lamp: captures what the PC is playing (WASAPI loopback) and
drives the lamp colour from it. VPN-safe (LAN-bound socket) and lamp-safe (one persistent
connection, <= rate limit, non-persistent commands so the lamp's flash is not worn out).

    .venv\\Scripts\\python.exe lightshow.py                 # until Ctrl+C
    .venv\\Scripts\\python.exe lightshow.py --palette cool --rate 5 --seconds 60

The previous light is restored on exit. The MCP server runs the same LightShow class.
"""
from __future__ import annotations

import argparse
import colorsys
import os
import sys
import threading
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", module="soundcard")
import soundcard as sc  # noqa: E402

import magichome as mh  # noqa: E402

SR, BLOCK, FFT_N = 48000, 1024, 2048
BANDS = {"bass": (30, 160), "mid": (160, 2000), "high": (2000, 10000)}

# hue ranges (0..1) per palette; rainbow wanders freely
PALETTES = {
    "rainbow": None,
    "warm": (0.94, 0.13),     # red -> orange -> yellow (wraps around 1.0)
    "cool": (0.45, 0.78),     # teal -> blue -> violet
    "rain": (0.50, 0.62),     # the "rainy day" blue-teal family
    "sunset": (0.96, 0.08),
}


def hue_in_palette(h: float, pal) -> float:
    if pal is None:
        return h % 1.0
    lo, hi = pal
    span = (hi - lo) % 1.0 or 1.0
    t = h % 2.0                       # ping-pong inside the range so colours stay in the family
    t = t if t <= 1.0 else 2.0 - t
    return (lo + span * t) % 1.0


class LightShow:
    def __init__(self, link: mh.LampLink, palette: str = "rainbow", rate_hz: float = mh.SAFE_RATE_HZ,
                 min_brightness: float = 0.12, device: str | None = None):
        if palette not in PALETTES:
            raise ValueError(f"unknown palette {palette!r}; choose from {sorted(PALETTES)}")
        self.link, self.palette, self.device = link, palette, device
        self.rate_hz = max(0.5, min(mh.MAX_RATE_HZ, rate_hz))
        self.min_brightness = max(0.0, min(0.9, min_brightness))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._flock = threading.Lock()
        self.level = {k: 0.0 for k in BANDS}
        self.rms = 0.0
        self._beat_counter = 0
        self.before: mh.LampState | None = None
        self.error: str | None = None
        self.speaker: str | None = None
        self.stats = {"sent": 0, "beats": 0, "rgb": (0, 0, 0), "started": 0.0}

    # -- lifecycle
    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self) -> None:
        self.before = self.link.state()
        if not self.before.power:
            self.link.send(mh.cmd_power(True))
        self._saved_interval = self.link.min_interval
        self.link.min_interval = 1.0 / self.rate_hz
        self.stats["started"] = time.monotonic()
        self._threads = [threading.Thread(target=self._audio_loop, daemon=True, name="ls-audio"),
                         threading.Thread(target=self._render_loop, daemon=True, name="ls-render")]
        for t in self._threads:
            t.start()

    def stop(self, restore: bool = True) -> dict:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        self.link.min_interval = getattr(self, "_saved_interval", self.link.min_interval)
        restored = None
        if restore and self.before is not None:
            st = self.before
            time.sleep(0.3)
            if st.mode == "white":
                self.link.send(mh.cmd_white(st.warm, st.cold))
            else:
                self.link.send(mh.cmd_rgb(*st.rgb))
            if not st.power:
                self.link.send(mh.cmd_power(False))
            restored = {"mode": st.mode, "rgb": st.rgb, "warm": st.warm, "cold": st.cold, "power": st.power}
        return {**self.status(), "restored": restored}

    def status(self) -> dict:
        dur = (time.monotonic() - self.stats["started"]) if self.stats["started"] else 0.0
        return {"running": self.running, "palette": self.palette, "rate_hz": self.rate_hz,
                "speaker": self.speaker, "seconds": round(dur, 1), "beats": self.stats["beats"],
                "frames_sent": self.stats["sent"],
                "frames_per_s": round(self.stats["sent"] / dur, 2) if dur else 0.0,
                "current_rgb": self.stats["rgb"], "error": self.error}

    # -- audio analysis
    def _audio_loop(self) -> None:
        try:
            spk = sc.get_speaker(self.device) if self.device else sc.default_speaker()
            self.speaker = spk.name
            mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
            freqs = np.fft.rfftfreq(FFT_N, 1 / SR)
            idx = {k: (freqs >= lo) & (freqs < hi) for k, (lo, hi) in BANDS.items()}
            window = np.hanning(FFT_N)
            ring = np.zeros(FFT_N, dtype=np.float32)
            peak = {k: 1e-6 for k in BANDS}          # slowly decaying peak = auto gain
            bass_avg, last_beat = 0.0, 0.0
            with mic.recorder(samplerate=SR, blocksize=BLOCK) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=BLOCK)
                    mono = data.mean(axis=1) if data.ndim == 2 else data
                    ring = np.roll(ring, -len(mono)); ring[-len(mono):] = mono
                    spec = np.abs(np.fft.rfft(ring * window))
                    raw = {k: float(np.sqrt((spec[m] ** 2).mean())) for k, m in idx.items()}
                    rms = float(np.sqrt((mono ** 2).mean()))
                    now = time.monotonic()
                    lv = {}
                    for k, v in raw.items():
                        peak[k] = max(v, peak[k] * 0.995)   # ~4 s decay at ~47 blocks/s
                        lv[k] = min(1.0, v / peak[k]) if peak[k] > 1e-4 else 0.0
                    is_beat = (raw["bass"] > 1.45 * bass_avg and lv["bass"] > 0.55
                               and rms > 0.004 and now - last_beat > 0.22)
                    bass_avg = 0.92 * bass_avg + 0.08 * raw["bass"]
                    with self._flock:
                        self.level, self.rms = lv, rms
                        if is_beat:
                            self._beat_counter += 1; last_beat = now
        except Exception as e:
            self.error = f"audio: {type(e).__name__}: {e}"
            self._stop.set()

    # -- colour rendering + sending
    def _render_loop(self) -> None:
        pal = PALETTES[self.palette]
        hue, bright, seen, flash = 0.0, 0.0, 0, 0.0
        last_sent = None
        t_prev = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic(); dt = now - t_prev; t_prev = now
                with self._flock:
                    lv, rms, beats = dict(self.level), self.rms, self._beat_counter
                new_beat = beats != seen
                if new_beat:
                    self.stats["beats"] += beats - seen; seen = beats
                    hue += 0.17 if pal is None else 0.23      # jump to a clearly different colour
                    flash = 1.0
                hue += dt * (0.015 + 0.06 * lv["high"])       # slow drift, faster with bright treble
                flash = max(0.0, flash - dt * 2.5)
                if rms < 0.002:                               # silence: dim, slow drift
                    target = self.min_brightness * 0.8
                else:
                    energy = 0.65 * lv["bass"] + 0.35 * lv["mid"]
                    target = self.min_brightness + (1 - self.min_brightness) * max(energy, flash)
                bright += (target - bright) * min(1.0, dt * 12)
                sat = 1.0 - 0.25 * lv["high"] * (1 - flash)
                rgb = tuple(int(round(255 * c)) for c in
                            colorsys.hsv_to_rgb(hue_in_palette(hue, pal), sat, max(0.0, min(1.0, bright))))
                changed = last_sent is None or sum(abs(x - y) for x, y in zip(rgb, last_sent)) > 8
                if (changed or new_beat) and self.link.try_send(mh.cmd_rgb(*rgb, persist=False)):
                    self.stats["sent"] += 1; self.stats["rgb"] = rgb; last_sent = rgb
                time.sleep(0.02)
        except Exception as e:
            self.error = f"render: {type(e).__name__}: {e}"
            self._stop.set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="lamp IP (default: LAMP_HOST / LAMP_MAC / cache / discovery)")
    ap.add_argument("--rate", type=float, default=mh.SAFE_RATE_HZ,
                    help=f"max commands per second (default {mh.SAFE_RATE_HZ}, hard cap {mh.MAX_RATE_HZ})")
    ap.add_argument("--palette", choices=sorted(PALETTES), default="rainbow")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--device", help="speaker name substring to capture (default: system default output)")
    ap.add_argument("--min-brightness", type=float, default=0.12)
    a = ap.parse_args()

    lamp = mh.Lamp(host=a.host)
    link = lamp.link
    show = LightShow(link, a.palette, a.rate, a.min_brightness, a.device)
    show.start()
    print(f"lamp {lamp.host} via {mh.source_ip_for(lamp.host).ip}. Ctrl+C to stop.")
    t0 = time.monotonic()
    try:
        while show.running and not (a.seconds and time.monotonic() - t0 > a.seconds):
            s = show.status()
            bar = "#" * int(show.level["bass"] * 20)
            print(f"\r rgb={s['current_rgb']!s:<16} bass {bar:<20} beats={s['beats']:<4} sent={s['frames_sent']:<5}",
                  end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    res = show.stop(restore=True)
    print(f"\nstopped: {res}")
    time.sleep(1.5)
    print("lamp now:", link.state().raw)
    link.close()
    return 1 if res["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
