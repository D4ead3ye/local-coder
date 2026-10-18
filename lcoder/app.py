"""Local Coder's window, drawn with VertexUI.

Left: model, project folder and how much autonomy the agent has. Right: the
transcript - what the model says, its collapsed reasoning, and every tool it
ran with live output - and the prompt box. The agent runs on its own thread;
this module only draws `agent.items` and answers approvals.

No shader backdrop or particles: the GPU is busy holding the model, and a
frame spent on decoration is VRAM and time taken from generation.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from imgui_bundle import hello_imgui, imgui, ImVec2

import vertexui as vui
from vertexui import theme, widgets, cards, icons, logview, fonts, settings, anim

from . import server as srv
from . import prompt
from .agent import Agent, Item, MODES, MODE_LABELS

TABS = ["Agente", "Servidor", "Ajustes"]
CTX_OPTIONS = [32768, 65536, 98304, 131072]
DEFAULT_WS = srv.ROOT / "workspace"
SIDE_W = 300.0


def C(c):
    return imgui.get_color_u32(c)


def note(text, tone=None):
    """A help line under a control: recessed ground and an accent rule."""
    t = theme.current()
    dl = imgui.get_window_draw_list()
    w = max(120.0, imgui.get_content_region_avail().x - 4.0)
    rule, pad = 3.0, 9.0
    inner = w - rule - pad * 2
    h = imgui.calc_text_size(text, wrap_width=inner).y + 12.0
    p = imgui.get_cursor_screen_pos()
    dl.add_rect_filled(p, ImVec2(p.x + w, p.y + h), C(theme.with_alpha(t.bg, 0.85)), t.rounding)
    dl.add_rect_filled(ImVec2(p.x, p.y + 2), ImVec2(p.x + rule, p.y + h - 2),
                       C(theme.with_alpha(tone or t.accent_bright, 0.85)), 1.0)
    imgui.set_cursor_screen_pos(ImVec2(p.x + rule + pad, p.y + 6))
    imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + inner)
    imgui.text_colored(t.text_dim, text)
    imgui.pop_text_wrap_pos()
    imgui.set_cursor_screen_pos(ImVec2(p.x, p.y + h + 5))
    imgui.dummy(ImVec2(w, 0))


def fmt_secs(s: float) -> str:
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s"


class App:
    def __init__(self, st: settings.Settings):
        self.st = st
        ex = st.extras
        self.model_key = ex.get("model", "normal")
        self.ctx = int(ex.get("ctx", 65536))
        self.mode = ex.get("mode", "auto")
        self.profile = ex.get("profile", prompt.DEFAULT_PROFILE)
        self.ws = ex.get("workspace", str(DEFAULT_WS))
        self.tab = 0
        self.prompt = ""
        self.focus_prompt = True
        self.open: dict[int, bool] = {}     # id(item) -> expanded
        self.stick = True                   # transcript follows the bottom
        self._last_items = 0
        self.attach: list[str] = []         # images going with the next prompt

        self.server = srv.Server(int(ex.get("port", 8089)))
        self.log = logview.LogView(speed="calm", timestamps=True, tags=True)
        self.st.apply(self.log)
        self._log_seq = 0

        self.agent = Agent(self.server.base, self.ws, ctx=self.ctx, mode=self.mode,
                           profile=self.profile)
        self.agent.on_event = self._agent_event
        self._apply_accent()

        if not self.server.adopt_existing() and srv.model(self.model_key).present:
            self.server.start(self.model_key, self.ctx)

    # -- plumbing --------------------------------------------------------

    def save(self):
        self.st.extras.update(model=self.model_key, ctx=self.ctx, mode=self.mode,
                              workspace=self.ws, profile=self.profile)
        try:
            self.st.save()
        except Exception:
            pass

    def _agent_event(self, kind, item):
        if kind == "done":
            vui.sound.play("ok")

    def pump_log(self):
        lines, self._log_seq = self.server.log_since(self._log_seq)
        for ln in lines:
            low = ln.lower()
            kind = ("error" if ("error" in low or "failed" in low) else
                    "warn" if "warn" in low else
                    "ok" if ln.startswith("[local coder]") else None)
            self.log.add(ln, kind=kind, tag="srv")

    # -- frame -----------------------------------------------------------

    def frame(self):
        self.pump_log()
        t = theme.current()
        imgui.push_style_var(imgui.StyleVar_.window_padding, ImVec2(0, 0))
        self._topbar(t)
        imgui.pop_style_var()

        pad = 16.0
        imgui.dummy(ImVec2(0, 2))
        imgui.indent(pad)
        self.tab = widgets.tabs("nav", TABS, self.tab)
        imgui.dummy(ImVec2(0, 6))
        [self._agent_tab, self._server_tab, self._settings_tab][self.tab](t, pad)
        imgui.unindent(pad)

        vui.end_frame()
        # tokens arrive from another thread with no input event to wake the
        # frame loop; idling would redraw the stream at the idle rate
        if self.agent.busy or self.server.state == "starting":
            try:
                hello_imgui.get_runner_params().fps_idling.enable_idling = False
            except Exception:
                pass

    # -- profile ---------------------------------------------------------

    def _apply_accent(self):
        """Give the live theme the active profile's accent, keeping the user's
        chosen shape, fonts and palette from Settings."""
        base = self.st.build_theme()
        accent = theme.rgb(prompt.profile(self.profile)["accent"])
        theme.use(base.with_accent(accent))

    def _profile_card(self, t):
        """Two big profile buttons at the top of the sidebar. Switching changes
        the agent's whole focus (and the accent), so it starts a fresh chat."""
        keys = ["redhand", "general"]
        with cards.card("modo", icon=None) as c:
            bw = (c.width - 10) / 2
            for i, key in enumerate(keys):
                p = prompt.profile(key)
                active = self.profile == key
                dl = imgui.get_window_draw_list()
                pos = imgui.get_cursor_screen_pos()
                bh = 52.0
                imgui.invisible_button(f"##prof{key}", ImVec2(bw, bh))
                hov = imgui.is_item_hovered()
                if imgui.is_item_clicked():
                    self.set_profile(key)
                acc = theme.rgb(p["accent"])
                fill = theme.lerp(t.surface, acc, 0.22 if active else (0.08 if hov else 0.0))
                dl.add_rect_filled(pos, ImVec2(pos.x + bw, pos.y + bh), C(fill), t.rounding)
                dl.add_rect(pos, ImVec2(pos.x + bw, pos.y + bh),
                            C(acc if active else theme.with_alpha(t.border, 0.7)),
                            t.rounding, 1.6 if active else 1.0)  # p_min,p_max,col,rounding,thickness
                self._profile_glyph(dl, p["icon"], pos.x + 17, pos.y + bh / 2, 9, C(acc))
                with fonts.use("semi"):
                    dl.add_text(ImVec2(pos.x + 32, pos.y + 10),
                                C(t.text if active else t.text_dim), p["label"])
                with fonts.use("label"):
                    dl.add_text(ImVec2(pos.x + 32, pos.y + 30), C(t.text_mute), p["tag"])
                if i == 0:
                    imgui.same_line(0, 10)
            note(prompt.profile(self.profile)["blurb"],
                 tone=theme.rgb(prompt.profile(self.profile)["accent"]))
            if self.agent.busy:
                imgui.text_colored(t.text_mute, "troca liberada quando terminar")

    def set_profile(self, key: str):
        if key == self.profile or self.agent.busy:
            return
        self.profile = key
        self.agent.profile = key
        self.agent.reset()          # the system prompt changes, so start fresh
        self.open.clear()
        self._apply_accent()
        self.save()
        vui.sound.play("click")

    def _profile_glyph(self, dl, name, cx, cy, r, col):
        """A drawn mark per profile - no font, no asset. 'hand' is the Red Hand
        badge; 'cube' is the general one."""
        if name == "hand":
            # palm
            pw, ph = r * 1.25, r * 1.15
            dl.add_rect_filled(ImVec2(cx - pw / 2, cy - ph * 0.15),
                               ImVec2(cx + pw / 2, cy + ph * 0.75), col, r * 0.45)
            # four fingers
            fw = pw / 5.2
            for i in range(4):
                fx = cx - pw / 2 + fw * 0.7 + i * (fw * 1.18)
                fh = r * (1.15 if i == 1 or i == 2 else 0.92)
                dl.add_rect_filled(ImVec2(fx, cy - fh), ImVec2(fx + fw, cy + r * 0.2),
                                   col, fw * 0.5)
            # thumb
            dl.add_rect_filled(ImVec2(cx - pw * 0.72, cy + r * 0.05),
                               ImVec2(cx - pw * 0.2, cy + r * 0.3), col, r * 0.25)
        else:  # cube, isometric
            import math
            top = [ImVec2(cx, cy - r), ImVec2(cx + r * 0.86, cy - r * 0.5),
                   ImVec2(cx, cy), ImVec2(cx - r * 0.86, cy - r * 0.5)]
            dl.add_convex_poly_filled(top, col)
            left = [ImVec2(cx - r * 0.86, cy - r * 0.5), ImVec2(cx, cy),
                    ImVec2(cx, cy + r), ImVec2(cx - r * 0.86, cy + r * 0.5)]
            dl.add_convex_poly_filled(left, C(theme.with_alpha(theme.current().accent, 0.65)))
            right = [ImVec2(cx + r * 0.86, cy - r * 0.5), ImVec2(cx, cy),
                     ImVec2(cx, cy + r), ImVec2(cx + r * 0.86, cy + r * 0.5)]
            dl.add_convex_poly_filled(right, C(theme.with_alpha(theme.current().accent_bright, 0.9)))

    def _topbar(self, t):
        dl = imgui.get_window_draw_list()
        w = imgui.get_window_size().x
        h = 54.0
        o = imgui.get_cursor_screen_pos()
        prof = prompt.profile(self.profile)
        dl.add_rect_filled(o, ImVec2(o.x + w, o.y + h), C(theme.with_alpha(t.panel, 0.96)))
        dl.add_line(ImVec2(o.x, o.y + h), ImVec2(o.x + w, o.y + h),
                    C(theme.with_alpha(t.accent, 0.5)), 1.5)
        self._profile_glyph(dl, prof["icon"], o.x + 27, o.y + h / 2, 11, C(t.accent_bright))
        name = prof["label"].upper()
        widgets.caps(dl, o.x + 50, o.y + 13, name, C(t.text), 2.4)
        with fonts.use("label"):
            widgets.caps(dl, o.x + 50, o.y + 33, prof["tag"], C(t.text_dim), 2.6)

        s = self.server
        ready = s.ready
        if self.agent.busy:
            word, on = "TRABALHANDO", True
        elif s.state == "starting":
            word, on = "CARREGANDO", True
        else:
            word, on = ("PRONTO" if ready else "DESLIGADO"), False

        # right-hand readouts: speed and context fill
        x = o.x + w - 470
        with fonts.use("label"):
            widgets.caps(dl, x, o.y + 12, "VELOCIDADE", C(t.text_mute), 1.6)
            widgets.caps(dl, x + 120, o.y + 12, "CONTEXTO", C(t.text_mute), 1.6)
        tps = self.agent.last_tps
        dl.add_text(ImVec2(x, o.y + 27), C(t.text if tps else t.text_mute),
                    f"{tps:.1f} tok/s" if tps else "--")
        used, cap = self.agent.ctx_used, max(1, s.ctx or self.ctx)
        frac = min(1.0, used / cap)
        dl.add_text(ImVec2(x + 120, o.y + 27), C(t.text_dim), f"{used // 1000}k / {cap // 1000}k")
        cards.meter(dl, x + 120, x + 220, o.y + 46, frac,
                    t.warn if frac > 0.75 else t.accent, t, h=3.0)

        imgui.set_cursor_screen_pos(ImVec2(o.x + w - 220, o.y + 15))
        widgets.status_pill(word, word, on=on, key="srv")
        imgui.set_cursor_screen_pos(ImVec2(o.x, o.y + h))

    # -- agent tab -------------------------------------------------------

    def _agent_tab(self, t, pad):
        avail = imgui.get_content_region_avail()
        h = avail.y - 12
        imgui.begin_child("side", ImVec2(SIDE_W, h))
        try:
            self._side(t)
        finally:
            imgui.end_child()
        imgui.same_line(0, 14)
        imgui.begin_group()
        main_w = avail.x - SIDE_W - 14 - pad
        input_h = 96.0
        pending = next((i for i in self.agent.items if i.kind == "tool" and i.status == "approve"), None)
        banner = 64.0 if pending else 0.0
        imgui.begin_child("chat", ImVec2(main_w, h - input_h - banner - 10))
        try:
            self._transcript(t, main_w)
        finally:
            imgui.end_child()
        if pending:
            self._approval_banner(t, pending, main_w)
        self._input(t, main_w, input_h)
        imgui.end_group()

    def _side(self, t):
        s = self.server
        self._profile_card(t)
        imgui.dummy(ImVec2(0, 4))
        with cards.card("modelo", icon="bolt",
                        right={"ready": "pronto", "external": "externo", "starting": "carregando",
                               "error": "erro", "off": "desligado"}[s.state],
                        right_col=t.ok if s.ready else (t.danger if s.state == "error" else t.text_dim)) as c:
            labels = [m.label + ("" if m.present else "  (baixando)") for m in srv.MODELS]
            idx = next((i for i, m in enumerate(srv.MODELS) if m.key == self.model_key), 0)
            new = widgets.combo("", labels, idx, width=c.width, key="model")
            if new != idx:
                self.model_key = srv.MODELS[new].key
                self.save()
            note(srv.MODELS[new].note + ("  Enxerga imagens." if srv.MODELS[new].vision
                                          else "  Sem visão (mmproj ausente)."))
            ci = CTX_OPTIONS.index(self.ctx) if self.ctx in CTX_OPTIONS else 1
            nci = widgets.combo("contexto", [f"{v // 1024}k tokens" for v in CTX_OPTIONS], ci,
                                width=c.width - 80, key="ctx")
            if nci != ci:
                self.ctx = CTX_OPTIONS[nci]
                self.save()
            loaded = s.ready and s.model_key == self.model_key and s.ctx == self.ctx
            if s.state == "starting":
                if widgets.button("Cancelar", c.width, key="srv_cancel"):
                    s.stop()
            elif loaded or s.state == "external":
                if widgets.button("Desligar modelo", c.width, key="srv_stop",
                                  enabled=not self.agent.busy):
                    s.stop()
            else:
                lbl = "Trocar modelo" if s.ready else "Ligar modelo"
                if widgets.button(lbl, c.width, primary=True, key="srv_start",
                                  enabled=srv.model(self.model_key).present and not self.agent.busy):
                    s.start(self.model_key, self.ctx)
            if s.state == "error":
                note(s.error, tone=t.danger)
            elif s.state == "starting":
                note("Carregando ~22 GB do disco para a GPU e a RAM. Leva de 20 s a 1 min.")
            elif s.state == "external":
                note("Usando um llama-server que já estava rodando nesta porta.")

        imgui.dummy(ImVec2(0, 4))
        with cards.card("projeto", icon="folder") as c:
            imgui.set_next_item_width(c.width)
            changed, val = imgui.input_text("##ws", self.ws)
            if changed:
                self.ws = val
            if imgui.is_item_deactivated_after_edit():
                self._set_ws(self.ws)
            bw = (c.width - 10) / 2
            if widgets.button("Escolher", bw, key="ws_pick", enabled=not self.agent.busy):
                self._pick_folder()
            imgui.same_line(0, 10)
            if widgets.button("Abrir", bw, key="ws_open"):
                if Path(self.ws).is_dir():
                    os.startfile(self.ws)
            note("O agente lê, cria e roda tudo a partir desta pasta.")

        imgui.dummy(ImVec2(0, 4))
        with cards.card("autonomia", icon="gear") as c:
            mi = MODES.index(self.mode) if self.mode in MODES else 0
            nmi = widgets.combo("", MODE_LABELS, mi, width=c.width, key="mode")
            if nmi != mi:
                self.mode = MODES[nmi]
                self.agent.mode = self.mode
                self.agent.allow.clear()
                self.save()
            note(["Escreve, compila, instala e roda sem perguntar nada.",
                  "Edita arquivos sozinho; pede ok antes de rodar comandos.",
                  "Pede ok para cada edição e cada comando."][nmi],
                 tone=t.warn if nmi == 0 else None)

        imgui.dummy(ImVec2(0, 4))
        with cards.card("sessão", icon="refresh") as c:
            n_tools = sum(1 for i in self.agent.items if i.kind == "tool")
            imgui.text_colored(t.text_dim, f"{self.agent.steps} passos · {n_tools} ferramentas")
            if widgets.button("Nova conversa", c.width, key="new_chat", enabled=not self.agent.busy):
                self.agent.reset()
                self.open.clear()
                self.focus_prompt = True

    def _set_ws(self, path):
        if self.agent.busy:
            return
        try:
            self.agent.set_workspace(path)
        except OSError as e:
            self.agent.items.append(Item("error", text=f"pasta inválida: {e}"))
            return
        self.ws = path
        self.save()

    def _pick_folder(self):
        try:
            from imgui_bundle import portable_file_dialogs as pfd
            r = pfd.select_folder("Pasta do projeto", self.ws).result()
        except Exception:
            r = ""
        if r:
            self._set_ws(r)

    # -- transcript ------------------------------------------------------

    def _transcript(self, t, width):
        items = self.agent.items
        # follow the bottom only while the user has not scrolled up
        at_bottom = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 40
        imgui.dummy(ImVec2(0, 4))
        if not items:
            self._empty(t)
        inner = width - 28
        for n, it in enumerate(items):
            imgui.push_id(n)
            try:
                if it.kind == "user":
                    self._user(t, it, inner)
                elif it.kind == "assistant":
                    self._assistant(t, it, inner, live=self.agent.busy and n == len(items) - 1)
                elif it.kind == "tool":
                    self._tool(t, it, inner)
                elif it.kind == "error":
                    imgui.text_colored(t.danger, "erro: " + it.text)
                else:
                    imgui.text_colored(t.text_mute, "· " + it.text)
            finally:
                imgui.pop_id()
            imgui.dummy(ImVec2(0, 4))
        if self.agent.busy:
            k = int(time.time() * 3) % 4
            live = self.agent.live_text()
            imgui.text_colored(t.text_mute, (live or "trabalhando") + " " + "." * k)
        imgui.dummy(ImVec2(0, 10))
        if at_bottom or len(items) != self._last_items and self.agent.busy:
            imgui.set_scroll_here_y(1.0)
        self._last_items = len(items)

    def _empty(self, t):
        imgui.dummy(ImVec2(0, 40))
        with fonts.use("title"):
            imgui.text("O que vamos construir?")
        imgui.text_colored(t.text_dim, "Descreva a tarefa. O agente lê a pasta do projeto, escreve o código,")
        imgui.text_colored(t.text_dim, "compila, testa e corrige até funcionar.")
        imgui.dummy(ImVec2(0, 10))
        for ex in ("Crie um parser de JSON em Lua puro com testes e rode os testes.",
                   "Escreva um servidor HTTP em C++ (winsock) que serve a pasta atual e teste com curl.",
                   "Faça um programa NASM x64 para Windows que imprime os 20 primeiros primos."):
            if imgui.selectable("> " + ex, False)[0]:
                self.prompt = ex
                self.focus_prompt = True

    def _user(self, t, it, w):
        with cards.card(None, width=w, edge=t.accent, pad=ImVec2(14, 10)):
            with fonts.use("label"):
                imgui.text_colored(t.accent_bright, "VOCÊ")
            imgui.text_wrapped(it.text)
            for p in it.images:
                imgui.text_colored(t.text_dim, "[imagem] " + Path(p).name)

    def _assistant(self, t, it, w, live):
        if it.think.strip():
            key = id(it)
            opened = self.open.get(key, False)
            busy_think = live and not it.text
            label = ("pensando..." if busy_think else "raciocínio") + f"  ({len(it.think)} caracteres)"
            if imgui.selectable(("[-] " if opened else "[+] ") + label + "##th", False)[0]:
                self.open[key] = not opened
            if opened or busy_think:
                imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + w)
                txt = it.think if opened else it.think[-600:]
                imgui.text_colored(t.text_mute, txt.strip())
                imgui.pop_text_wrap_pos()
        if it.text.strip():
            self._markdown(t, it.text, w)
            if not live and imgui.small_button("copiar resposta"):
                imgui.set_clipboard_text(it.text)

    def _markdown(self, t, text, w):
        """Enough markdown to read code answers: fenced blocks in mono with a
        copy button, headings, bullets. Inline marks are just stripped."""
        parts = text.split("```")
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + w)
        for i, part in enumerate(parts):
            if i % 2 == 1:
                lang, _, code = part.partition("\n")
                self._code(t, lang.strip(), code.rstrip("\n"), w, i)
                continue
            for line in part.strip("\n").splitlines():
                s = line.strip()
                if not s:
                    imgui.dummy(ImVec2(0, 4))
                elif s.startswith("#"):
                    with fonts.use("semi"):
                        imgui.text_wrapped(s.lstrip("#").strip().replace("**", ""))
                elif s[:2] in ("- ", "* "):
                    imgui.bullet()
                    imgui.same_line()
                    imgui.text_wrapped(s[2:].replace("**", "").replace("`", ""))
                else:
                    imgui.text_wrapped(line.replace("**", "").replace("`", ""))
        imgui.pop_text_wrap_pos()

    def _code(self, t, lang, code, w, n):
        with fonts.use("mono"):
            sz = imgui.calc_text_size(code or " ")
            line_h = imgui.get_text_line_height_with_spacing()
        # label row + code + child padding + room for the horizontal scrollbar
        h = min(sz.y + 64, 520)
        imgui.push_style_color(imgui.Col_.child_bg, theme.with_alpha(t.bg, 0.9))
        imgui.push_style_var(imgui.StyleVar_.child_rounding, t.rounding)
        imgui.begin_child(f"code{n}", ImVec2(w, h), imgui.ChildFlags_.borders,
                          imgui.WindowFlags_.horizontal_scrollbar)
        try:
            with fonts.use("label"):
                imgui.text_colored(t.text_mute, (lang or "código").upper())
            imgui.same_line(w - 80)
            if imgui.small_button("copiar"):
                imgui.set_clipboard_text(code)
            with fonts.use("mono"):
                imgui.text_unformatted(code)
        finally:
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()

    def _tool(self, t, it, w):
        col = {"ok": t.ok, "err": t.danger, "running": t.accent_bright,
               "approve": t.warn, "denied": t.text_mute}.get(it.status, t.text_dim)
        key = id(it)
        opened = self.open.get(key, it.status == "running" and it.name == "run")
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        row_h = 30.0
        clicked = imgui.invisible_button("##tool", ImVec2(w, row_h))
        hov = imgui.is_item_hovered()
        dl.add_rect_filled(p, ImVec2(p.x + w, p.y + row_h),
                           C(t.surface_hover if hov else t.surface), t.rounding)
        dl.add_rect_filled(ImVec2(p.x, p.y + 3), ImVec2(p.x + 3, p.y + row_h - 3), C(col), 1.0)
        if it.status == "running":
            pulse = 0.5 + 0.5 * abs(((time.time() * 1.6) % 2) - 1)
            icons.draw("dot", dl, p.x + 18, p.y + row_h / 2, 4 + 2 * pulse, C(col))
        else:
            icons.draw({"ok": "check", "err": "cross", "denied": "cross",
                        "approve": "warning"}.get(it.status, "dot"),
                       dl, p.x + 18, p.y + row_h / 2, 6, C(col))
        with fonts.use("label"):
            widgets.caps(dl, p.x + 34, p.y + 9, it.name.upper(), C(t.text_dim), 1.4)
            name_w = widgets.caps_width(it.name.upper(), 1.4)
        el = (it.t1 or time.time()) - it.t0
        right = fmt_secs(el)
        rw = imgui.calc_text_size(right).x
        summ = it.summary().replace("\n", " ; ")
        with fonts.use("mono"):
            room = w - (34 + name_w + 14) - rw - 26
            cw = max(1.0, imgui.calc_text_size("M").x)
            maxc = max(8, int(room / cw))
            if len(summ) > maxc:
                summ = summ[:maxc - 3] + "..."
            dl.add_text(ImVec2(p.x + 34 + name_w + 14, p.y + 7), C(t.text), summ)
        dl.add_text(ImVec2(p.x + w - rw - 12, p.y + 7), C(t.text_mute), right)
        if clicked:
            self.open[key] = not opened
            opened = not opened
        if opened:
            body = it.output
            if it.name == "write_file" and "content" in it.args:
                body = it.args["content"] + "\n\n— " + (it.output or "")
            elif it.name == "edit_file":
                body = ("- " + it.args.get("old_text", "").replace("\n", "\n- ") + "\n+ "
                        + it.args.get("new_text", "").replace("\n", "\n+ ")
                        + "\n\n— " + (it.output or ""))
            self._code(t, it.name, (body or "(sem saída ainda)").rstrip(), w, key)

    def _approval_banner(self, t, it, w):
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        h = 56.0
        dl.add_rect_filled(p, ImVec2(p.x + w, p.y + h), C(theme.with_alpha(t.warn, 0.12)), t.rounding)
        dl.add_rect(p, ImVec2(p.x + w, p.y + h), C(theme.with_alpha(t.warn, 0.6)), t.rounding)
        imgui.set_cursor_screen_pos(ImVec2(p.x + 12, p.y + 6))
        imgui.text_colored(t.warn, f"O agente quer usar {it.name}:")
        imgui.set_cursor_screen_pos(ImVec2(p.x + 12, p.y + 28))
        with fonts.use("mono"):
            s = it.summary().replace("\n", " ; ")
            imgui.text(s[:110] + ("..." if len(s) > 110 else ""))
        bx = p.x + w - 372
        imgui.set_cursor_screen_pos(ImVec2(bx, p.y + 10))
        if widgets.button("Permitir", 100, primary=True, key="ap_yes"):
            self.agent.resolve(it, "yes")
        imgui.same_line(0, 8)
        if widgets.button("Sempre nesta sessão", 160, key="ap_always"):
            self.agent.resolve(it, "always")
        imgui.same_line(0, 8)
        if widgets.button("Negar", 90, key="ap_no"):
            self.agent.resolve(it, "no")
        imgui.set_cursor_screen_pos(ImVec2(p.x, p.y + h + 6))
        imgui.dummy(ImVec2(w, 0))

    def _input(self, t, w, h):
        bw = 112.0
        if self.focus_prompt:
            imgui.set_keyboard_focus_here()
            self.focus_prompt = False
        flags = (imgui.InputTextFlags_.enter_returns_true
                 | imgui.InputTextFlags_.ctrl_enter_for_new_line)
        imgui.push_style_color(imgui.Col_.frame_bg, t.surface)
        sent, self.prompt = imgui.input_text_multiline(
            "##prompt", self.prompt, ImVec2(w - bw - 10, h - 22), flags)
        imgui.pop_style_color()
        io = imgui.get_io()
        if imgui.is_item_focused() and io.key_ctrl and imgui.is_key_pressed(imgui.Key.v, False):
            self._paste_image()
        imgui.same_line(0, 10)
        imgui.begin_group()
        ready = self.server.ready
        if self.agent.busy:
            if widgets.button("Parar", bw, height=h - 22, key="stop"):
                self.agent.stop()
        else:
            go = widgets.button("Enviar", bw, height=h - 22, primary=True, key="send",
                                enabled=ready and bool(self.prompt.strip()))
            if (go or sent) and ready and self.prompt.strip():
                self.agent.ctx = self.server.ctx or self.ctx
                self.agent.vision = self.server.vision
                self.agent.send(self.prompt.strip(), images=self.attach)
                self.prompt = ""
                self.attach = []
                self.focus_prompt = True
        imgui.end_group()
        self._attach_row(t)

    def _attach_row(self, t):
        if imgui.small_button("+ imagem"):
            self._pick_image()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Anexar uma imagem (ou cole com Ctrl+V na caixa de texto)")
        for i, p in enumerate(list(self.attach)):
            imgui.same_line(0, 8)
            imgui.text_colored(t.accent_bright, Path(p).name)
            imgui.same_line(0, 4)
            if imgui.small_button(f"x##att{i}"):
                self.attach.remove(p)
        imgui.same_line(0, 14)
        if not self.server.ready:
            hint = "Ligue o modelo na coluna da esquerda para começar."
        elif self.attach and not self.server.vision:
            hint = "Este modelo foi carregado sem visão: baixe o mmproj e religue."
        else:
            hint = "Enter envia · Ctrl+Enter quebra linha"
        imgui.text_colored(t.text_mute, hint)

    def _pick_image(self):
        try:
            from imgui_bundle import portable_file_dialogs as pfd
            r = pfd.open_file("Imagem de referência", self.ws,
                              ["Imagens", "*.png *.jpg *.jpeg *.bmp *.webp"]).result()
        except Exception:
            r = []
        self.attach += [p for p in r if p not in self.attach]

    def _paste_image(self):
        try:
            from PIL import ImageGrab
            clip = ImageGrab.grabclipboard()
        except Exception:
            return
        if isinstance(clip, list):            # files copied in Explorer
            self.attach += [p for p in clip if Path(p).suffix.lower()
                            in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]
        elif clip is not None:
            d = srv.ROOT / "attachments"
            d.mkdir(exist_ok=True)
            out = d / time.strftime("colado-%Y%m%d-%H%M%S.png")
            clip.save(out)
            self.attach.append(str(out))

    # -- other tabs ------------------------------------------------------

    def _server_tab(self, t, pad):
        s = self.server
        imgui.text_colored(t.text_dim, f"{s.base}  ·  estado: {s.state}  ·  modelo: {s.model_key or '-'}"
                           f"  ·  contexto: {s.ctx}")
        avail = imgui.get_content_region_avail()
        self.log.draw(ImVec2(avail.x - pad, avail.y - 12))

    def _settings_tab(self, t, pad):
        imgui.begin_child("settings", ImVec2(0, imgui.get_content_region_avail().y - 12))
        try:
            secs = [x for x in settings.SECTIONS if x not in ("backdrop", "effects")]
            settings.panel(self.st, self.log, sections=secs,
                           on_change=lambda _s: (self.save(), self._apply_accent()))
        finally:
            imgui.end_child()

    def shutdown(self):
        self.agent.stop()
        self.agent.box.stop_all()
        self.server.stop()
        self.save()


def run():
    st = settings.Settings.load(app="localcoder")
    if not st.extras.get("inited"):
        st.theme_name = "noir-blue"
        st.extras["inited"] = True
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Local Coder"
    params.app_window_params.window_geometry.size = (1320, 880)
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_window)
    params.imgui_window_params.show_menu_bar = False
    app = App(st)
    params.callbacks.show_gui = app.frame
    params.callbacks.before_exit = app.shutdown
    vui.install(params, theme_=st.build_theme(), faces=st.build_faces())
    hello_imgui.run(params)
