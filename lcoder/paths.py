"""Where the app's data lives.

Running from source, that is the project folder. Frozen into an .exe by
PyInstaller, `__file__` points inside the unpacked bundle, so the models and
the llama.cpp build would be looked for in a temp dir. When frozen, resolve
everything next to the executable instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent.parent
