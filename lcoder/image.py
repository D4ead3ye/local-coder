"""The Image tool: prompt -> picture, local diffusion on the GPU.

Mirrors the audio tool. The heavy torch/diffusers stack lives in the project's
CUDA venv (ai-venv) and runs image_server.py; this controller spawns that
server and drives it over HTTP, so the app's own Python stays torch-free.
Generation runs on a worker thread; the result is a PNG loaded with Pillow.
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

from .paths import ROOT

AI_VENV_PY = ROOT / "ai-venv" / "Scripts" / "python.exe"
IMAGE_SERVER = Path(__file__).resolve().parent / "sfx" / "image_server.py"
OUT_DIR = ROOT / "output" / "images"
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# key -> label for the UI model picker (kept in sync with image_server.MODELS)
MODELS = [
    ("sdxl", "SDXL 1.0 (1024, qualidade)"),
    ("sdxl-turbo", "SDXL Turbo (512, rápido)"),
    ("sd15", "SD 1.5 (512, leve)"),
]
# sensible default size per model
SIZES = {"sdxl": 1024, "sdxl-turbo": 512, "sd15": 512}


def _free_port(start=8140):
    for p in [start] + [start + i for i in range(1, 40)]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p)); return p
        except OSError:
            continue
        finally:
            s.close()
    return 0


class ImageServer:
    """Drives image_server.py in the CUDA venv (same venv as audio)."""

    def __init__(self):
        self.port = 0
        self.base = ""
        self.proc = None
        self.state = "off"      # off | starting | loading | ready | generating | error | missing
        self.error = ""
        self.device = ""
        self.model = "sdxl"

    @property
    def available(self) -> bool:
        return AI_VENV_PY.exists()

    def provision(self, log=print) -> bool:
        from . import neural_setup
        return neural_setup.provision(log=log)

    def _get(self, path, timeout=2):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read())

    def ensure(self, model="sdxl"):
        if not self.available:
            self.state = "missing"
            self.error = "ai-venv ausente — rode: python -m lcoder.neural_setup"
            return False
        if self.proc and self.proc.poll() is None:
            return True
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.state, self.error = "starting", ""
        self.proc = subprocess.Popen(
            [str(AI_VENV_PY), str(IMAGE_SERVER), "--port", str(self.port),
             "--model", model, "--preload"],
            cwd=str(IMAGE_SERVER.parent), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=FLAGS)
        for _ in range(60):
            try:
                self._get("/health", timeout=1)
                return True
            except Exception:
                time.sleep(0.5)
        self.state, self.error = "error", "servidor de imagem não respondeu"
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
            self.model = s.get("model", self.model)
            if s.get("detail"):
                self.error = s["detail"]
        except Exception:
            pass

    def generate(self, prompt, model, negative, steps, guidance, width, height, seed, out_path):
        if not self.ensure(model):
            return False
        body = json.dumps({"prompt": prompt, "model": model, "negative": negative,
                           "steps": steps, "guidance": guidance, "width": width,
                           "height": height, "seed": seed, "out": str(out_path)}).encode()
        req = urllib.request.Request(self.base + "/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                json.loads(r.read())
            return True
        except Exception as e:
            self.error = str(e)
            return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.state = "off"


class ImageTool:
    def __init__(self):
        self.server = ImageServer()
        self.prompt = ""
        self.negative = ""
        self.model = "sdxl"
        self.steps = 30
        self.guidance = 7.0
        self.size = 1024
        self.seed = 0
        self.busy = False
        self.error = ""
        self.last_path: Path | None = None
        self.image = None              # PIL.Image of the last result
        self.tex = None                # (texture_id, w, h) cached by the UI
        self.on_update = None

    def set_model(self, key):
        self.model = key
        self.size = SIZES.get(key, 1024)
        if key == "sdxl-turbo":
            self.steps, self.guidance = 4, 0.0
        elif key == "sd15":
            self.steps, self.guidance = 28, 7.5
        else:
            self.steps, self.guidance = 30, 7.0

    def generate(self):
        if self.busy or not self.prompt.strip():
            return
        self.busy = True
        self.error = ""
        threading.Thread(target=self._gen, daemon=True).start()

    def _gen(self):
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            seed = self.seed if self.seed else int(time.time() * 1000) % 2**31
            out = OUT_DIR / f"img_{int(time.time())}_{seed}.png"
            ok = self.server.generate(self.prompt.strip(), self.model, self.negative.strip(),
                                      self.steps, self.guidance, self.size, self.size, seed, out)
            if not ok:
                self.error = self.server.error or "geração de imagem falhou"
                return
            self.last_path = out
            self.tex = None            # force the UI to re-upload the texture
            try:
                from PIL import Image
                self.image = Image.open(out).convert("RGBA")
            except Exception as e:
                self.error = f"abrir PNG: {e}"
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.busy = False
            if self.on_update:
                self.on_update()

    def stop(self):
        self.server.stop()
