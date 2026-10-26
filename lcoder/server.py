"""Owns the llama-server process: start, stop, health, and its log.

The server speaks the OpenAI chat API on localhost. If one is already
listening on the port (started by hand, or left over), it is adopted rather
than fighting it for the port.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .paths import ROOT
LLAMA_DIR = ROOT / "llama"
MODELS_DIR = ROOT / "models"
NCMOE = 24


@dataclass
class Model:
    key: str
    label: str
    file: str
    note: str
    mmproj: str = ""       # vision projector; without it the model is text-only
    mtp: bool = False      # carries multi-token-prediction heads
    moe: bool = True       # MoE (expert offload) vs dense (per-layer offload)
    bytes_: int = 0        # expected complete size, to tell a finished file from a partial one
    ctx: int = 65536       # preferred context for this model

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.file

    @property
    def present(self) -> bool:
        p = self.path
        if not p.exists():
            return False
        # a file still downloading looks present but is useless; accept only a
        # file within 1% of the known complete size (or >13 GB if unknown)
        n = p.stat().st_size
        return n >= self.bytes_ * 0.99 if self.bytes_ else n > 13e9

    @property
    def vision(self) -> bool:
        p = MODELS_DIR / self.mmproj
        return bool(self.mmproj) and p.exists() and p.stat().st_size > 8e8


MODELS = [
    Model("normal", "Qwen3.6 35B-A3B",
          "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
          "Melhor qualidade. Recusa raramente em código.",
          mmproj="mmproj-qwen3.6-F16.gguf", bytes_=22360456160),
    Model("abliterated", "Qwen3.6 35B-A3B abliterated",
          "Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K.gguf",
          "Sem recusas. Um pouco menos preciso.",
          mmproj="mmproj-huihui-abliterated-f16.gguf", mtp=True, bytes_=21712409824),
]


def model(key: str) -> Model:
    for m in MODELS:
        if m.key == key:
            return m
    return MODELS[0]


def _bindable(port: int) -> bool:
    """True if we can actually bind 127.0.0.1:port right now. Catches both a
    port in use AND a Windows *reserved* port range (Hyper-V/WSL grab ranges,
    after which the OS refuses the bind and the real cause is invisible)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _free_port(preferred: int) -> int:
    """The preferred port if usable, else the next usable one, else an
    OS-assigned ephemeral port."""
    for p in [preferred] + [preferred + i for i in range(1, 40)]:
        if _bindable(p):
            return p
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class Server:
    def __init__(self, port: int = 8089):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.state = "off"          # off | starting | ready | error | external
        self.error = ""
        self.model_key = ""
        self.ctx = 0
        self.vision = False
        self._fallback = None
        self.log: deque[str] = deque(maxlen=3000)
        self._log_seq = 0
        self._lock = threading.Lock()

    # -- queries ---------------------------------------------------------

    def healthy(self, timeout=0.6) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def props(self) -> dict:
        try:
            with urllib.request.urlopen(self.base + "/props", timeout=2) as r:
                return json.loads(r.read())
        except Exception:
            return {}

    @property
    def ready(self) -> bool:
        return self.state in ("ready", "external")

    def log_since(self, seq: int):
        """Lines added after `seq`, and the new seq."""
        with self._lock:
            n = self._log_seq - seq
            lines = list(self.log)[-n:] if 0 < n <= len(self.log) else (
                list(self.log) if n > 0 else [])
            return lines, self._log_seq

    def _push(self, line: str):
        with self._lock:
            self.log.append(line)
            self._log_seq += 1

    # -- lifecycle -------------------------------------------------------

    def adopt_existing(self) -> bool:
        if self.proc is None and self.healthy():
            self.state = "external"
            p = self.props()
            self.ctx = int(p.get("default_generation_settings", {}).get("n_ctx", 0) or 0)
            self.vision = bool(p.get("modalities", {}).get("vision"))
            self._push(f"[local coder] servidor já ativo em {self.base}, usando ele")
            return True
        return False

    def start(self, model_key: str, ctx: int = 65536, mtp: bool = True, ncmoe: int = NCMOE):
        """`ncmoe` = expert layers kept on the CPU. 0 lets llama.cpp's --fit
        decide; measured on the RTX 5070 with 64k context and the vision
        projector loaded, the hand-picked 24 is ~13% faster than --fit."""
        self.stop()
        m = model(model_key)
        exe = LLAMA_DIR / "llama-server.exe"
        if not exe.exists():
            self.state, self.error = "error", f"não achei {exe}"
            return
        if not m.present:
            self.state, self.error = "error", f"modelo ausente ou incompleto: {m.file}"
            return
        # the saved port may now be in a Windows reserved range (nothing binds
        # it, the server would just exit "couldn't bind"); move to a free one
        if not _bindable(self.port):
            new = _free_port(self.port + 1)
            self._push(f"[local coder] porta {self.port} indisponível (reservada/ocupada); usando {new}")
            self.port = new
            self.base = f"http://127.0.0.1:{self.port}"
        args = [str(exe), "-m", str(m.path),
                "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(ctx),
                "-fa", "on",
                # q8 KV halves the cache with no measurable loss, which is
                # what lets 64k context sit next to the weights in 12 GB
                "-ctk", "q8_0", "-ctv", "q8_0",
                "--jinja",
                "--reasoning-format", "deepseek",
                "--no-webui",
                "-a", m.key,
                "-np", "1",          # one agent, one slot: no idle KV/compute buffers
                "-t", str(max(1, (os.cpu_count() or 12) // 2))]
        if m.moe and ncmoe > 0:
            args += ["-ngl", "99", "-ncmoe", str(ncmoe), "--fit", "off"]
        else:
            # dense models have no experts to offload; --fit sizes -ngl to fit
            # the GPU. (Also the fallback path when the fixed split overflows.)
            args += ["--fit-target", "768"]
        self._fallback = (model_key, ctx, mtp) if (m.moe and ncmoe > 0) else None
        if m.vision:
            args += ["--mmproj", str(MODELS_DIR / m.mmproj)]
        else:
            args += ["--no-mmproj"]
        if m.mtp and mtp:
            args += ["--spec-type", "draft-mtp"]
        self.model_key, self.ctx, self.vision = m.key, ctx, m.vision
        self.state, self.error = "starting", ""
        self._push("[local coder] " + " ".join(f'"{a}"' if " " in a else a for a in args))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                args, cwd=str(LLAMA_DIR), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=flags, text=True, encoding="utf-8", errors="replace")
        except OSError as e:
            self.state, self.error = "error", str(e)
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._wait_ready, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self._push(line.rstrip())
        code = proc.wait()
        if proc is self.proc:
            self.proc = None
            if self.state == "starting" and self._fallback:
                # the fixed split did not fit (another app holding VRAM?):
                # let --fit size it to whatever is free right now
                key, ctx, mtp = self._fallback
                self._push("[local coder] não coube na VRAM com o ajuste fixo; tentando o automático (--fit)")
                self.start(key, ctx, mtp, ncmoe=0)
                return
            if self.state != "off":
                self.state = "error"
                self.error = f"llama-server saiu (código {code}); veja a aba Servidor"

    def _wait_ready(self, proc):
        while proc is self.proc and proc.poll() is None:
            if self.healthy(timeout=1.0):
                self.state = "ready"
                self._push("[local coder] modelo carregado")
                return
            time.sleep(0.5)

    def stop(self):
        p, self.proc = self.proc, None
        if self.state != "external":
            self.state = "off"
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
            self._push("[local coder] servidor parado")
