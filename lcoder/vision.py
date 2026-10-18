"""Images for the model: load and shrink a file, capture a window.

Images go to llama-server as data URLs inside a user message. They are
scaled down first: a 4K screenshot costs thousands of image tokens and the
model sees no more detail than at ~1280 px.
"""

from __future__ import annotations

import base64
import ctypes
import io
import time
from ctypes import wintypes
from pathlib import Path

MAX_SIDE = 1280

user32 = ctypes.windll.user32
try:
    # physical pixels, so window rects match what ImageGrab captures
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass


def data_url(path: str | Path) -> tuple[str, tuple[int, int]]:
    from PIL import Image
    im = Image.open(path)
    im.load()
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    w, h = im.size
    k = min(1.0, MAX_SIDE / max(w, h))
    if k < 1.0:
        im = im.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), im.size


def image_message(paths, text: str) -> dict:
    content = [{"type": "text", "text": text}]
    for p in paths:
        url, _ = data_url(p)
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


def _title(hwnd) -> str:
    # InternalGetWindowText reads the stored title without sending
    # WM_GETTEXT, so asking about our own window cannot block on our own
    # render thread
    buf = ctypes.create_unicode_buffer(512)
    user32.InternalGetWindowText(hwnd, buf, 512)
    return buf.value


def windows() -> list[tuple[int, str]]:
    out = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            t = _title(hwnd)
            if t.strip():
                out.append((hwnd, t))
        return True

    user32.EnumWindows(proc(cb), 0)
    return out


def find_window(title_part: str):
    want = title_part.lower()
    hits = [(h, t) for h, t in windows() if want in t.lower()]
    # exact title first, then the shortest match (least likely a browser tab)
    hits.sort(key=lambda ht: (ht[1].lower() != want, len(ht[1])))
    return hits[0] if hits else (None, "")


def screenshot(out: Path, title_part: str | None = None) -> str:
    from PIL import ImageGrab
    if title_part:
        hwnd, title = find_window(title_part)
        if not hwnd:
            names = ", ".join(sorted({t for _, t in windows()})[:25])
            raise RuntimeError(f"nenhuma janela com '{title_part}' no título. Abertas: {names}")
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)          # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.4)
        r = wintypes.RECT()
        # DWM extended frame bounds exclude the invisible resize border
        try:
            ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(r), ctypes.sizeof(r))
        except Exception:
            user32.GetWindowRect(hwnd, ctypes.byref(r))
        img = ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom), all_screens=True)
        label = title
    else:
        img = ImageGrab.grab(all_screens=True)
        label = "tela inteira"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return f"{label} ({img.size[0]}x{img.size[1]})"
