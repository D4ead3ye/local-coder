"""Launch Local Coder.

    python run.py                       the window
    python run.py --cli --workspace DIR "task"   one unattended task

VertexUI is imported from the sibling repo checkout if it is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import vertexui  # noqa: F401
except ImportError:
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
