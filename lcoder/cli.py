"""Headless runner: one task, fully autonomous, progress on stdout.

    python -m lcoder.cli --workspace C:\\proj "build X and test it"

Starts the server if none is running, runs the task in "auto" mode and
prints every tool call as it happens. Used for unattended jobs and for
benchmarking the agent.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import server as srv
from . import prompt
from .agent import Agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--model", default="normal", choices=[m.key for m in srv.MODELS])
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--image", action="append", default=[], help="imagem de referência (pode repetir)")
    ap.add_argument("--profile", default="redhand", choices=list(prompt.PROFILES))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    s = srv.Server(a.port)
    if not s.adopt_existing():
        print(f"[iniciando servidor: {a.model}]", flush=True)
        s.start(a.model, a.ctx)
        while s.state == "starting":
            time.sleep(0.5)
        if not s.ready:
            lines, _ = s.log_since(0)
            print("\n".join(lines[-30:]))
            sys.exit(f"servidor falhou: {s.error}")

    ag = Agent(s.base, a.workspace, ctx=s.ctx or a.ctx, mode="auto", profile=a.profile)
    ag.max_steps = a.max_steps
    ag.vision = s.vision
    t0 = time.time()

    def ev(kind, it):
        el = time.time() - t0
        if kind == "tool_start":
            print(f"[{el:6.0f}s] > {it.name}: {it.summary()[:160]}", flush=True)
        elif kind == "tool":
            first = it.output.splitlines()[0] if it.output else ""
            print(f"[{el:6.0f}s]   {it.status}: {first[:160]}", flush=True)
        elif kind == "assistant" and it.text.strip():
            print(f"[{el:6.0f}s] {it.text.strip()[:600]}", flush=True)
        elif kind == "error":
            print(f"[{el:6.0f}s] ERRO {it.text}", flush=True)
        elif kind == "progress":
            print(f"[{el:6.0f}s]   ... {it}", flush=True)

    ag.on_event = ev
    ag.send(a.task, blocking=True, images=a.image)
    final = next((i.text for i in reversed(ag.items) if i.kind == "assistant" and i.text.strip()), "")
    tools_n = sum(1 for i in ag.items if i.kind == "tool")
    print("\n===== RESPOSTA FINAL =====\n" + final)
    print(f"\n[{time.time() - t0:.0f}s, {ag.steps} passos, {tools_n} ferramentas, "
          f"{ag.last_tps:.1f} tok/s, contexto {ag.ctx_used}/{ag.ctx}; sessão {ag.session_file}]")
    if s.proc:
        s.stop()


if __name__ == "__main__":
    main()
