"""Launch Local Coder.

    python run.py                       the window
    python run.py --cli --workspace DIR "task"   one unattended task

VertexUI is used from the sibling repo checkout, put ahead of any pip-installed
copy on sys.path — an older installed vertexui (1.2) would otherwise shadow the
repo (1.13+) and the new modules would be "missing".
"""

from __future__ import annotations

import sys
from pathlib import Path

# repo first, so the current VertexUI wins over any stale pip install
for cand in (Path(r"C:\claude\vertexui-repo"),
             Path(__file__).resolve().parent.parent / "vertexui-repo"):
    if (cand / "vertexui").is_dir():
        sys.path.insert(0, str(cand))
        break

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from lcoder.cli import main as cli_main
        cli_main()
        return
    from lcoder.app import run
    run()


if __name__ == "__main__":
    main()
