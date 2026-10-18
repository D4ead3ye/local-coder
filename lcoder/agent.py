"""The agent loop: ask the model, run the tools it calls, feed results back,
until it answers without calling a tool.

It is UI-agnostic. A front-end reads `items` (the transcript as drawn) and
answers approvals through `resolve`; the headless CLI just sets mode "auto".
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import llm, prompt, tools, vision

MODES = ["auto", "edits", "ask"]
MODE_LABELS = ["Autônomo total", "Pergunta antes de comandos", "Pergunta tudo"]
from .paths import ROOT
SESSIONS = ROOT / "sessions"


@dataclass
class Item:
    kind: str                       # user | assistant | tool | error | info
    text: str = ""
    think: str = ""
    name: str = ""
    args: dict = field(default_factory=dict)
    status: str = ""                # tool: approve | running | ok | err | denied
    output: str = ""
    images: list = field(default_factory=list)
    t0: float = field(default_factory=time.time)
    t1: float = 0.0
    _gate: threading.Event | None = None
    _answer: str = ""

    def summary(self) -> str:
        a = self.args
        if self.name in ("run", "bg_start"):
            return a.get("command", "")
        if self.name in ("read_file", "write_file", "edit_file", "list_dir"):
            return a.get("path", "") or "."
        if self.name == "search":
            return f"/{a.get('pattern', '')}/ " + (a.get("glob") or "")
        if self.name == "fetch_url":
            return a.get("url", "")
        if self.name in ("bg_output", "bg_stop"):
            return f"id {a.get('id', '?')}"
        return json.dumps(a, ensure_ascii=False)[:200]


class Agent:
    def __init__(self, base_url: str, workspace: str, ctx: int = 65536, mode: str = "auto",
                 profile: str = prompt.DEFAULT_PROFILE):
        self.base = base_url
        self.ctx = ctx
        self.mode = mode
        self.profile = profile
        self.allow = set()          # kinds approved "always" this session
        self.items: list[Item] = []
        self.busy = False
        self.cancel = threading.Event()
        self.last_tps = 0.0
        self.ctx_used = 0
        self.steps = 0
        self.max_steps = 200
        self.vision = True          # set from the server: was a projector loaded
        self.live = None            # what the model is producing right now, for progress
        self.on_event = None        # callback(kind, item) for headless output
        self._toolchains = prompt.probe()
        self.set_workspace(workspace)
        self.session_file = None

    # -- setup -----------------------------------------------------------

    def set_workspace(self, ws: str):
        Path(ws).mkdir(parents=True, exist_ok=True)
        self.workspace = ws
        self.box = tools.Toolbox(ws)
        self.box.cancel = self.cancel
        self.reset()

    def reset(self):
        if self.busy:
            return
        self.box.stop_all()
        self.messages = [{"role": "system",
                          "content": prompt.build(self.workspace, self._toolchains, self.profile)}]
        self.items = []
        self._tool_imgs = set()     # screenshots may be dropped; the user's reference never is
        self.ctx_used = 0
        self.session_file = None

    # -- driving ---------------------------------------------------------

    def send(self, text: str, blocking=False, images=()):
        if self.busy or not text.strip():
            return
        self.box.vision = self.vision
        images = [str(p) for p in images]
        self.items.append(Item("user", text=text, images=images))
        if images:
            try:
                self.messages.append(vision.image_message(images, text))
            except Exception as e:
                self.items.append(Item("error", text=f"não consegui abrir a imagem: {e}"))
                return
        else:
            self.messages.append({"role": "user", "content": text})
        self.busy = True
        self.cancel.clear()
        if blocking:
            self._run()
        else:
            threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.cancel.set()
        for it in self.items:
            if it._gate is not None and not it._gate.is_set():
                it._answer = "no"
                it._gate.set()

    def resolve(self, item: Item, answer: str):
        """answer: yes | always | no"""
        if answer == "always":
            self.allow.add(tools.KIND.get(item.name, "exec"))
        item._answer = answer
        if item._gate:
            item._gate.set()

    def live_text(self) -> str:
        lv = self.live
        if not lv:
            return ""
        what = lv["phase"]
        if what == "ferramenta":
            m = re.search(r'"(?:path|command|url)"\s*:\s*"((?:[^"\\]|\\.){0,90})', lv["args"])
            what = f"preparando {lv['tool']}" + (f" ({m.group(1)})" if m else "")
        n = lv["chars"]
        size = f"{n / 1000:.1f}k caracteres" if n >= 1000 else f"{n} caracteres"
        return f"{what} · {size} · {time.time() - lv['t0']:.0f}s"

    def _emit(self, kind, item):
        if self.on_event:
            try:
                self.on_event(kind, item)
            except Exception:
                pass

    def _needs_ok(self, name: str) -> bool:
        kind = tools.KIND.get(name, "exec")
        if kind == "read" or self.mode == "auto" or kind in self.allow:
            return False
        if self.mode == "edits":
            return kind == "exec"
        return True

    def _run(self):
        self.steps = 0
        self._nudges = 0
        try:
            while not self.cancel.is_set():
                self.steps += 1
                if self.steps > self.max_steps:
                    self.items.append(Item("error", text=f"parei após {self.max_steps} passos"))
                    break
                if not self._step():
                    break
        except llm.Cancelled:
            self.items.append(Item("info", text="interrompido"))
        except Exception as e:
            self.items.append(Item("error", text=f"{type(e).__name__}: {e}"))
            self._emit("error", self.items[-1])
        finally:
            self.busy = False
            self._save()
            self._emit("done", None)

    def _step(self) -> bool:
        """One model call plus the tools it asked for. False when finished."""
        it = Item("assistant")
        self.items.append(it)

        live = self.live = {"phase": "pensando", "chars": 0, "tool": "", "args": "",
                            "t0": time.time(), "beat": time.time()}

        def on_delta(kind, text):
            if kind == "text":
                it.text += text
                live["phase"] = "escrevendo"
            elif kind == "think":
                it.think += text
                live["phase"] = "pensando"
            elif kind == "tool_name":
                live["phase"], live["tool"], live["args"] = "ferramenta", live["tool"] + text, ""
            elif kind == "tool":
                live["args"] += text
            live["chars"] += len(text)
            # a long tool call (a whole file in write_file) prints nothing for
            # minutes; a heartbeat says it is still going
            if time.time() - live["beat"] > 20:
                live["beat"] = time.time()
                self._emit("progress", self.live_text())

        try:
            msg, stats = llm.stream_chat(self.base, self.messages, tools.SCHEMAS,
                                         on_delta, self.cancel)
        finally:
            self.live = None
        it.t1 = time.time()
        if stats.get("tps"):
            self.last_tps = stats["tps"]
        self.ctx_used = (stats.get("cache_n", 0) + stats.get("prompt_n", 0)
                         + stats.get("predicted_n", 0))
        if not it.text.strip() and not it.think.strip():
            self.items.remove(it)
        else:
            self._emit("assistant", it)
        self.messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            if stats.get("finish") == "length":
                # cut off mid-answer: let it continue rather than end the task
                self.messages.append({"role": "user", "content":
                                      "Your last message was cut off by the length limit. Continue."})
                return True
            # The model sometimes announces the next action in prose ("agora vou
            # escrever o X:") and stops without emitting the tool call, which
            # would otherwise read as "done" and leave the task half-finished.
            # Nudge it to actually act, a few times, before accepting the end.
            tail = it.text.strip()[-80:].lower()
            announces = (tail.endswith(":") or re.search(
                r"\b(vou|irei|agora vou|deixa eu|let me|i'?ll|i will|next[, ]|em seguida|"
                r"então vou|now i)\b.*\b(escrever|criar|fazer|implementar|gerar|rodar|"
                r"compilar|testar|write|create|run|build|implement|add|edit)\b", tail))
            if announces and self._nudges < 4:
                self._nudges += 1
                self.messages.append({"role": "user", "content":
                    "Você anunciou o próximo passo mas não chamou a ferramenta. Execute agora, "
                    "de fato, chamando a ferramenta; não apenas descreva. Se a tarefa já está "
                    "completa e verificada, responda com o relatório final."})
                return True
            return False

        for c in calls:
            if self.cancel.is_set():
                raise llm.Cancelled()
            self.messages.append({"role": "tool", "tool_call_id": c["id"],
                                  "content": self._call(c)})
        self._nudges = 0        # it acted; the nudge budget is for consecutive stalls
        if self.box.images:
            # images cannot ride in a tool result, so they follow as a user turn
            shots, self.box.images = self.box.images, []
            try:
                m = vision.image_message(
                    shots, "Imagem(ns) pedida(s) pela ferramenta acima: " + ", ".join(map(str, shots)))
                self.messages.append(m)
                self._tool_imgs.add(id(m))
            except Exception as e:
                self.messages.append({"role": "user", "content": f"(falha ao anexar imagem: {e})"})
        self._compact()
        return True

    def _call(self, c) -> str:
        name = c["function"]["name"]
        raw = c["function"].get("arguments") or "{}"
        ti = Item("tool", name=name)
        self.items.append(ti)
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except Exception as e:
            ti.status, ti.output = "err", f"JSON inválido: {e}\n{raw[:400]}"
            self._emit("tool", ti)
            return f"erro: os argumentos não são JSON válido ({e}). Reenvie a chamada corretamente."
        ti.args = args

        if self._needs_ok(name):
            ti.status = "approve"
            ti._gate = threading.Event()
            ti._gate.wait()
            ti._gate = None
            if ti._answer == "no":
                ti.status, ti.t1 = "denied", time.time()
                return "O usuário NEGOU esta ação. Não tente de novo do mesmo jeito; ajuste o plano ou pergunte."

        ti.status = "running"
        self._emit("tool_start", ti)

        def live(line):
            ti.output += line + "\n"
            if len(ti.output) > 60000:
                ti.output = ti.output[-40000:]

        self.box.on_output = live if name == "run" else None
        result = self.box.call(name, args)
        self.box.on_output = None
        ti.output = result
        ti.status = "err" if result.startswith("erro") or (
            name == "run" and not result.startswith("[exit code 0]")) else "ok"
        ti.t1 = time.time()
        self._emit("tool", ti)
        return result

    # -- context ---------------------------------------------------------

    def _compact(self):
        """Keep the history inside the window: old tool output is the bulk
        and the least useful, so it goes first."""
        if self.ctx_used < self.ctx * 0.65:
            return
        keep_from = max(1, len(self.messages) - 10)
        for m in self.messages[1:keep_from]:
            if isinstance(m.get("content"), list) and id(m) in self._tool_imgs:
                # an image is ~1-2k tokens; keep only the text beside it
                m["content"] = " ".join(p.get("text", "") for p in m["content"]
                                        if p.get("type") == "text") + " [imagem antiga removida]"
            if m["role"] == "tool" and len(m["content"]) > 400:
                m["content"] = m["content"][:300] + "\n[... saída antiga removida para liberar contexto]"
            if m["role"] == "assistant":
                m.pop("reasoning_content", None)
        if self.ctx_used > self.ctx * 0.85:
            for m in self.messages[1:keep_from]:
                if m["role"] == "assistant":
                    for tc in m.get("tool_calls") or []:
                        fn = tc["function"]
                        if len(fn.get("arguments", "")) > 600 and fn["name"] == "write_file":
                            try:
                                a = json.loads(fn["arguments"])
                                a["content"] = "[conteúdo escrito no arquivo; leia o arquivo se precisar]"
                                fn["arguments"] = json.dumps(a, ensure_ascii=False)
                            except Exception:
                                pass
        self.items.append(Item("info", text="contexto compactado (saídas antigas resumidas)"))

    # -- persistence -----------------------------------------------------

    def _save(self):
        try:
            SESSIONS.mkdir(exist_ok=True)
            if not self.session_file:
                self.session_file = SESSIONS / time.strftime("%Y%m%d-%H%M%S.json")
            msgs = [dict(m, content=[p if p.get("type") != "image_url" else {"type": "image_url", "image_url": "<omitida>"}
                                     for p in m["content"]]) if isinstance(m.get("content"), list) else m
                    for m in self.messages]
            self.session_file.write_text(json.dumps(
                {"workspace": self.workspace, "messages": msgs},
                ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass
