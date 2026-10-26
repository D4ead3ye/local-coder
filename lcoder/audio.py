"""The Audio tool: prompt -> sound, merged from SFX Forge.

The engine (parser -> recipe -> render) is pure numpy DSP and lives in
`lcoder.sfx`. This is the controller the UI drives: it holds the player, the
current recipe and its rendered audio, generates variations, and exports WAV.
Generation runs on a worker thread so the window never stutters.

Music / advanced neural generation (Stable Audio Open, MusicGen) will plug in
here as alternative generators alongside the procedural one.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from .paths import ROOT
from .sfx import parser, player as player_mod, export

# UI cue roles plus a free "geral" for arbitrary sounds
ROLES = ["geral", "click", "hover", "ok", "warn", "error", "toast"]

AI_VENV_PY = ROOT / "ai-venv" / "Scripts" / "python.exe"
NEURAL_SERVER = Path(__file__).resolve().parent / "sfx" / "neural_server.py"
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _read_wav(path) -> "np.ndarray | None":
    """Read a 16-bit WAV into (N, 2) float32 using the stdlib, so the app's
    Python needs no soundfile/torch."""
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except Exception:
        return None
    if sw != 2:
        return None
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch >= 2:
        a = a.reshape(-1, ch)[:, :2]
    else:
        a = np.repeat(a[:, None], 2, axis=1)
    return a


def _free_port(start=8130):
    for p in [start] + [start + i for i in range(1, 40)]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p)); return p
        except OSError:
            continue
        finally:
            s.close()
    return 0


class NeuralAudio:
    """Drives the Stable Audio Open server running in the project's CUDA venv.
    Starts it on first use (model load is slow), speaks its HTTP API."""

    def __init__(self):
        self.port = 0
        self.base = ""
        self.proc = None
        self.state = "off"      # off | starting | loading | ready | error | missing
        self.error = ""
        self.device = ""

    @property
    def available(self) -> bool:
        return AI_VENV_PY.exists()

    def _get(self, path, timeout=2):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read())

    def ensure(self):
        """Start the server if it is not up. Returns False if the venv is
        missing (neural generation unavailable)."""
        if not self.available:
            self.state, self.error = "missing", "ai-venv ausente (torch não instalado)"
            return False
        if self.proc and self.proc.poll() is None:
            return True
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.state, self.error = "starting", ""
        self.proc = subprocess.Popen(
            [str(AI_VENV_PY), str(NEURAL_SERVER), "--port", str(self.port), "--preload"],
            cwd=str(NEURAL_SERVER.parent), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=FLAGS)
        for _ in range(60):
            try:
                self._get("/health", timeout=1)
                return True
            except Exception:
                time.sleep(0.5)
        self.state, self.error = "error", "servidor neural não respondeu"
        return False

    def refresh_state(self):
        if not (self.proc and self.proc.poll() is None):
            if self.state not in ("off", "missing", "error"):
                self.state = "off"
            return
        try:
            s = self._get("/status")
            self.state = s.get("status", "?")
            self.device = s.get("device", "")
            if s.get("detail"):
                self.error = s["detail"]
        except Exception:
            pass

    def generate(self, prompt, mode, seconds, steps, seed, out_path) -> bool:
        if not self.ensure():
            return False
        body = json.dumps({"prompt": prompt, "mode": mode, "seconds": seconds,
                           "steps": steps, "seed": seed, "out": str(out_path)}).encode()
        req = urllib.request.Request(self.base + "/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                json.loads(r.read())
            return True
        except Exception as e:
            self.error = str(e)
            return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.state = "off"


class AudioTool:
    def __init__(self):
        self.player = player_mod.Player()
        self.renderer = player_mod.Renderer()
        self.neural = NeuralAudio()
        self.prompt = ""
        self.seed = 0
        self.role = "geral"
        self.volume = 0.8
        self.backend = "procedural"       # procedural | sfx | music
        self.seconds = 3.0                # neural clip length
        self.steps = 50                   # neural diffusion steps
        self.recipe = None
        self.audio: np.ndarray | None = None
        self.vars: list = []              # variation recipes
        self.var_cache: dict = {}         # id(recipe) -> audio
        self.busy = False
        self.error = ""
        self.on_update = None

    def _role(self):
        return None if self.role == "geral" else self.role

    # -- generation ------------------------------------------------------

    def generate(self):
        if self.busy or not self.prompt.strip():
            return
        self.busy = True
        self.error = ""
        threading.Thread(target=self._gen, daemon=True).start()

    def _gen(self):
        try:
            if self.backend == "procedural":
                self.recipe = parser.parse(self.prompt.strip(), seed=self.seed, role=self._role())
                self.audio = np.asarray(self.renderer.get(self.recipe))
            else:
                self._gen_neural()
            if self.audio is not None:
                self.play()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.busy = False
            if self.on_update:
                self.on_update()

    def _gen_neural(self):
        out = ROOT / "ai-venv" / "_gen.wav"
        secs = max(1.0, min(47.0, self.seconds if self.backend == "music" else min(self.seconds, 11.0)))
        ok = self.neural.generate(self.prompt.strip(), self.backend, secs,
                                  self.steps, self.seed, out)
        if not ok:
            self.error = self.neural.error or "geração neural falhou"
            return
        self.audio = _read_wav(out)
        self.recipe = None

    def variations(self, count=6):
        if self.busy or not self.prompt.strip():
            return
        self.busy = True
        self.error = ""
        threading.Thread(target=self._vars, args=(count,), daemon=True).start()

    def _vars(self, count):
        try:
            self.vars = parser.variations(self.prompt.strip(), count,
                                          seed=self.seed, role=self._role())
            self.var_cache = {}
            for r in self.vars:
                self.var_cache[id(r)] = np.asarray(self.renderer.get(r))
            if self.vars:
                self.recipe = self.vars[0]
                self.audio = self.var_cache[id(self.vars[0])]
                self.play()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.busy = False
            if self.on_update:
                self.on_update()

    def pick_variation(self, i):
        if 0 <= i < len(self.vars):
            self.recipe = self.vars[i]
            self.audio = self.var_cache.get(id(self.recipe))
            self.play()

    # -- playback / export ----------------------------------------------

    def play(self, audio=None):
        a = self.audio if audio is None else audio
        if a is not None:
            self.player.volume = self.volume
            self.player.play(a)

    def stop(self):
        self.player.stop()

    def export_wav(self, path) -> str:
        if self.audio is None:
            return "nada para exportar"
        export.write_wav(str(path), self.audio)
        return f"salvo: {path}"

    @property
    def duration(self) -> float:
        return 0.0 if self.audio is None else len(self.audio) / 44100.0

    def waveform(self, points: int):
        """Downsampled peak envelope for drawing, in [-1, 1]."""
        if self.audio is None or len(self.audio) == 0:
            return None
        mono = self.audio.mean(axis=1) if self.audio.ndim == 2 else self.audio
        n = len(mono)
        if n <= points:
            return mono
        step = n // points
        chunks = mono[:step * points].reshape(points, step)
        # peak per chunk, signed by the louder sample
        hi = chunks.max(axis=1)
        lo = chunks.min(axis=1)
        return np.where(np.abs(hi) >= np.abs(lo), hi, lo)
