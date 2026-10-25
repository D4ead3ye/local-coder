"""Background system monitor: GPU VRAM/util via nvidia-smi, system RAM via the
Win32 API. Sampled on a thread so the UI never blocks on it.

This exists mostly to make ONE failure legible: when the KV cache grows past the
free VRAM, llama.cpp spills it to shared system memory and generation collapses
from ~30 tok/s to a crawl. Seeing VRAM pinned at the limit explains that
instantly, where before it just looked like the model got slow.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from ctypes import wintypes

FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _ram():
    try:
        m = _MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            used = m.ullTotalPhys - m.ullAvailPhys
            return used / 2**30, m.ullTotalPhys / 2**30
    except Exception:
        pass
    return 0.0, 0.0


def _gpu():
    """(used_gb, total_gb, util_pct) from nvidia-smi, or zeros if unavailable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, creationflags=FLAGS)
        line = r.stdout.strip().splitlines()[0]
        used, total, util = (p.strip() for p in line.split(","))
        return float(used) / 1024, float(total) / 1024, float(util)
    except Exception:
        return 0.0, 0.0, 0.0


class SysMon:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.vram_used = 0.0
        self.vram_total = 0.0
        self.gpu_util = 0.0
        self.ram_used = 0.0
        self.ram_total = 0.0
        self.has_gpu = False
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        # one immediate sample so the first frame has numbers
        self._sample()
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self):
        u, tot, util = _gpu()
        if tot > 0:
            self.vram_used, self.vram_total, self.gpu_util, self.has_gpu = u, tot, util, True
        self.ram_used, self.ram_total = _ram()

    @property
    def vram_frac(self) -> float:
        return self.vram_used / self.vram_total if self.vram_total else 0.0

    @property
    def vram_tight(self) -> bool:
        """VRAM within ~400 MB of full — the point where the KV cache starts
        spilling to shared memory and generation slows to a crawl."""
        return self.vram_total > 0 and (self.vram_total - self.vram_used) < 0.4

    def stop(self):
        self._stop.set()
