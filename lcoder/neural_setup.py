"""Provision the neural CUDA venv (ai-venv) used for audio/image generation.

This is a SEPARATE environment from the app: it carries the heavy torch stack
so the main app's Python stays light and torch-free. neural_server.py runs
inside it and the app drives it over HTTP.

    python -m lcoder.neural_setup            # create + install
    python -m lcoder.neural_setup --check     # just report what's there

Torch is installed first from the CUDA wheel index so the build matches the
local driver; the rest comes from requirements-neural.txt.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import ROOT

VENV = ROOT / "ai-venv"
VENV_PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
REQS = ROOT / "requirements-neural.txt"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
# packages neural_server.py imports — used by --check and by ensure()
NEEDED = ["torch", "diffusers", "transformers", "accelerate", "soundfile", "torchsde"]


def present() -> bool:
    """True if the venv exists and imports the full neural stack."""
    if not VENV_PY.exists():
        return False
    code = "import " + ", ".join(NEEDED)
    r = subprocess.run([str(VENV_PY), "-c", code], capture_output=True)
    return r.returncode == 0


def _run(args, **kw):
    print("  $", " ".join(str(a) for a in args), flush=True)
    return subprocess.run(args, **kw).returncode


def provision(log=print) -> bool:
    """Create ai-venv (if missing) and install the neural stack. Idempotent;
    safe to re-run. Returns True on success."""
    if present():
        log("ai-venv already complete.")
        return True
    if not VENV_PY.exists():
        log("Creating ai-venv ...")
        if _run([sys.executable, "-m", "venv", str(VENV)]):
            log("venv creation failed."); return False
    log("Installing torch (CUDA wheel) — this is a large download ...")
    if _run([str(VENV_PY), "-m", "pip", "install", "torch", "--index-url", TORCH_INDEX]):
        log("torch install failed."); return False
    log("Installing neural requirements ...")
    if _run([str(VENV_PY), "-m", "pip", "install", "-r", str(REQS)]):
        log("requirements install failed."); return False
    ok = present()
    log("Done — neural stack ready." if ok else "Install finished but imports still fail.")
    return ok


def main():
    if "--check" in sys.argv:
        print("ai-venv:", "ready" if present() else "missing/incomplete", f"({VENV})")
        return
    sys.exit(0 if provision() else 1)


if __name__ == "__main__":
    main()
