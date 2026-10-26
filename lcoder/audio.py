"""The Audio tool: prompt -> sound, merged from SFX Forge.

The engine (parser -> recipe -> render) is pure numpy DSP and lives in
`lcoder.sfx`. This is the controller the UI drives: it holds the player, the
current recipe and its rendered audio, generates variations, and exports WAV.
Generation runs on a worker thread so the window never stutters.

Music / advanced neural generation (Stable Audio Open, MusicGen) will plug in
here as alternative generators alongside the procedural one.
"""

from __future__ import annotations

import threading

import numpy as np

from .sfx import parser, player as player_mod, export

# UI cue roles plus a free "geral" for arbitrary sounds
ROLES = ["geral", "click", "hover", "ok", "warn", "error", "toast"]


class AudioTool:
    def __init__(self):
        self.player = player_mod.Player()
        self.renderer = player_mod.Renderer()
        self.prompt = ""
        self.seed = 0
        self.role = "geral"
        self.volume = 0.8
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
            self.recipe = parser.parse(self.prompt.strip(), seed=self.seed, role=self._role())
            self.audio = np.asarray(self.renderer.get(self.recipe))
            self.play()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.busy = False
            if self.on_update:
                self.on_update()

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
