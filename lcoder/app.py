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
import re
import subprocess
import time
from pathlib import Path

from imgui_bundle import hello_imgui, imgui, ImVec2, ImVec4

import vertexui as vui
from vertexui import (theme, widgets, cards, icons, logview, fonts, settings, anim,
                      effects as fx_mod, backdrop as bg_mod, compose, profiles)

from . import server as srv
from . import prompt
from . import sysmon as sysmon_mod
from . import audio as audio_mod
from . import image as image_mod
from . import rp as rp_mod
from .agent import Agent, Item, MODES, MODE_LABELS

TABS = ["Agente", "Áudio", "Servidor", "Ajustes"]
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


def _fit(text: str, max_w: float) -> str:
    """Trim text (with an ellipsis) to fit max_w in the current font, so a
    wider themed face never spills out of a fixed box."""
    if imgui.calc_text_size(text).x <= max_w:
        return text
    for n in range(len(text) - 1, 0, -1):
        s = text[:n].rstrip() + "…"
        if imgui.calc_text_size(s).x <= max_w:
            return s
    return "…"


def _scrim_fn(dl, x, y, w, h, t, amount, rounding=0.0):
    if amount <= 0.004:
        return
    dl.add_rect_filled(ImVec2(x, y), ImVec2(x + w, y + h),
                       imgui.get_color_u32(theme.with_alpha(t.bg, min(1.0, amount))), rounding)


def _seed_themes(app_name: str):
    """Make VertexUI 1.13's characterful themes selectable in Settings by
    registering them as presets. They fit a coding/tooling app: PHOSPHOR is a
    green CRT console, SLATE an amber instrument, QUIET a calm cyan, PAPER a
    light reading ground. Accent is still overridden per profile."""
    try:
        from vertexui import profiles
    except Exception:
        return
    existing = set(theme.list_presets(app_name))
    for attr in ("PHOSPHOR", "SLATE", "QUIET", "PAPER_WHITE"):
        th = getattr(profiles, attr, None)
        if th is not None and th.name not in existing:
            try:
                theme.save_preset(th, th.name, app=app_name)
            except Exception:
                pass


class App:
    def __init__(self, st: settings.Settings):
        _seed_themes(st.app)
        self.st = st
        ex = st.extras
        self.model_key = ex.get("model", "normal")
        self.ctx = int(ex.get("ctx", 65536))
        self.mode = ex.get("mode", "auto")
        self.profile = ex.get("profile", prompt.DEFAULT_PROFILE)
        self.look = ex.get("look", "noir")   # VertexUI visual profile (identity)
        self.capp = None                     # the compose.App, set by Shell.setup
        self.look_plain = True
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

        # RP side (hidden): persona chat, no tools
        self.rp_mode = False
        self.rp_dir = ex.get("rp_dir", str(rp_mod.CARDS_DIR))
        self.rp_user = ex.get("rp_user", "Você")
        self.rp_user_desc = ex.get("rp_user_desc", "")      # your own character's lore
        self.rp_personas: list = ex.get("rp_personas", [])  # saved {"name", "desc"}
        self._rp_pidx = 0
        self.rp = rp_mod.RPChat(self.server.base, self.rp_user)
        self.rp.user_desc = self.rp_user_desc
        self.rp.on_update = lambda: None
        self.rp_cards: list = []
        self.rp_prompt = ""
        self.rp_focus = False
        self._rp_hint = 0.0                  # the corner dot brightens on hover
        self._rp_edit = -1                   # turn index being edited, or -1
        self._rp_edit_buf = ""
        self._rp_ctrl_hot = -1               # turn whose controls are shown

        # Style layer: a GPU shader backdrop and drifting particles, like the
        # OctoView/Bedside apps. First run seeds a look worth showing.
        if not ex.get("styled"):
            st.backdrop, st.backdrop_intensity, st.backdrop_speed = "nebula", 0.6, 0.5
            st.effect, st.effect_over = "embers", 0.3
            st.card_style = "raised"
            ex["styled"] = True
        self.fx = fx_mod.Effects(st.effect)
        self.bg = bg_mod.Backdrop(st.backdrop)
        self.st.apply(self.log, backdrop=self.bg, effects=self.fx)
        self._screen_was = ("rp", self.tab)
        self._fade = 0.0
        self._tex: dict = {}                 # avatar path -> (texid, w, h) or None
        self.sysmon = sysmon_mod.SysMon()    # background VRAM/RAM sampler
        self.audio = audio_mod.AudioTool()
        self.audio_focus = False
        self.image = image_mod.ImageTool()
        self.image_focus = False

        if not self.server.adopt_existing() and srv.model(self.model_key).present:
            self.server.start(self.model_key, self.ctx)

    # -- plumbing --------------------------------------------------------

    def save(self):
        self.st.extras.update(model=self.model_key, ctx=self.ctx, mode=self.mode,
                              workspace=self.ws, profile=self.profile, look=self.look,
                              port=self.server.port,
                              rp_dir=self.rp_dir, rp_user=self.rp_user,
                              rp_user_desc=self.rp_user_desc, rp_personas=self.rp_personas)
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
        io = imgui.get_io()
        now = imgui.get_time()
        w, h = io.display_size.x, io.display_size.y
        dlw = imgui.get_window_draw_list()

        # 1) GPU shader backdrop first, painting over the window's own fill.
        # a characterful look owns its own surface; the wallpaper is only for
        # the plain NOIR look.
        if self.look_plain:
            self.bg.draw()
        moving = self.look_plain and self.fx.kind != "none" and self.fx.step(now, io.delta_time)
        if moving:
            self.fx.paint(dlw, 0, 0, w, h, now, 1.0)

        imgui.push_style_var(imgui.StyleVar_.window_padding, ImVec2(0, 0))
        self._topbar(t)
        imgui.pop_style_var()

        pad = 16.0
        imgui.dummy(ImVec2(0, 2))
        imgui.indent(pad)
        # a wash behind bare text so the moving backdrop never eats legibility
        if (self.rp_mode or self.tab in (1, 2)):
            _scrim_fn(dlw, 0, 56, w, h - 56, t, 0.5 if self.tab in (1, 2) else 0.32)
        if self.rp_mode:
            self._rp_frame(t, pad)
        else:
            self.tab = widgets.tabs("nav", TABS, self.tab)
            imgui.dummy(ImVec2(0, 6))
            [self._agent_tab, self._audio_tab, self._server_tab,
             self._settings_tab][self.tab](t, pad)
        imgui.unindent(pad)

        # 3) particles again over the top, drifting across the panels.
        if moving and self.st.effect_over > 0.004:
            self.fx.paint(dlw, 0, 0, w, h, now, self.st.effect_over)
        # 4) a short fade over any screen change reads as navigation.
        screen = ("rp", self.rp.card.name if self.rp.card else "") if self.rp_mode else ("code", self.tab)
        if screen != self._screen_was:
            self._screen_was = screen
            self._fade = 1.0
        if self._fade > 0.004:
            self._fade = max(0.0, self._fade - io.delta_time * 4.5)
            dlw.add_rect_filled(ImVec2(0, 0), ImVec2(w, h),
                                C(theme.with_alpha(t.bg, self._fade * 0.6)))
            anim.mark_busy()

        vui.end_frame()
        # tokens arrive from another thread with no input event to wake the
        # frame loop; idling would redraw the stream at the idle rate
        if self.rp.busy or self.agent.busy or self.server.state == "starting":
            try:
                hello_imgui.get_runner_params().fps_idling.enable_idling = False
            except Exception:
                pass

    # -- profile ---------------------------------------------------------

    def _coding_model(self):
        """The right coding model for the active profile: abliterated (no
        refusals) for the game-tooling profile, else the best-quality one."""
        want = "abliterated" if self.profile == "redhand" else "normal"
        m = srv.model(want)
        if m.present:
            return m
        return next((x for x in srv.MODELS if not x.rp and x.present), srv.MODELS[0])

    def _apply_accent(self):
        """Apply the chosen VertexUI look (a whole identity: palette, shape,
        skin, spinner, fonts) and, on the plain NOIR look only, tint it with
        the agent mode's accent. Characterful looks (ena-world etc.) own their
        whole palette and get no wallpaper to fight."""
        from vertexui import profiles
        prof = profiles.get(self.look)
        prof.apply(self.st.app)
        self.look_plain = (self.look == "noir")
        if self.look_plain:
            base = self.st.build_theme()
            accent = theme.rgb(prompt.profile(self.profile)["accent"])
            theme.use(base.with_accent(accent))
            try:
                theme.apply_style(theme.current())   # needs an ImGui context
            except Exception:
                pass

    def set_look(self, name):
        if name == self.look:
            return
        self.look = name
        if self.capp is not None:
            # swap the whole identity AND the layout live, so the app reshapes
            self.capp.profile = profiles.get(name)
        self._apply_accent()
        self.save()
        vui.sound.play("click")

    # the VertexUI looks worth offering, retro/characterful first
    LOOKS = [("ena-world", "ENA World (Win9x retrô)"), ("frutiger-aero", "Frutiger Aero"),
             ("ymo-technodelic", "YMO Technodelic"), ("y2k-cyber", "Y2K Cyber"),
             ("steel-ball-run", "Steel Ball Run"), ("console", "Console (fósforo)"),
             ("instrument", "Instrument (âmbar)"), ("noir", "Noir (padrão)")]

    def _look_card(self, t):
        with cards.card("visual", icon=None) as c:
            keys = [k for k, _ in self.LOOKS]
            labels = [lbl for _, lbl in self.LOOKS]
            cur = keys.index(self.look) if self.look in keys else len(keys) - 1
            new = widgets.combo("", labels, cur, width=c.width, key="look")
            if new != cur:
                self.set_look(keys[new])
            note("Troca a cara inteira do app (paleta, formato, spinner, fonte). "
                 "O ENA World é um Windows 9x retrô.")

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
                tx, avail = pos.x + 32, bw - 32 - 6
                dl.push_clip_rect(ImVec2(pos.x + 2, pos.y), ImVec2(pos.x + bw - 2, pos.y + bh), True)
                with fonts.use("semi"):
                    dl.add_text(ImVec2(tx, pos.y + 10),
                                C(t.text if active else t.text_dim), _fit(p["label"], avail))
                with fonts.use("label"):
                    dl.add_text(ImVec2(tx, pos.y + 30), C(t.text_mute), _fit(p["tag"], avail))
                dl.pop_clip_rect()
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

    # -- hub home --------------------------------------------------------

    TILES = [("agent", "AGENTE", "cube", "Escreve, compila e testa código. Mods e engenharia reversa.", False),
             ("audio", "ÁUDIO", "play", "Gera SFX e sons a partir de prompt. Música em breve.", False),
             ("rp", "ROLEPLAY", "hand", "Conversa com personagens. Cards, lorebooks, personas.", False),
             ("image", "IMAGEM", "dot", "Geração de imagem local na GPU. SDXL e mais.", False)]

    def _home_page(self, t, pad):
        avail = imgui.get_content_region_avail()
        with fonts.use("title"):
            imgui.text("LOCAL CODER")
        imgui.same_line(0, 12)
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 6)
        imgui.text_colored(t.text_dim, "HUB DE IA LOCAL")
        imgui.dummy(ImVec2(0, 10))
        # tool tiles, two per row
        cols = 2
        gap = 14.0
        tw = (avail.x - pad - gap * (cols - 1)) / cols
        for i, (key, title, icon, desc, soon) in enumerate(self.TILES):
            if i % cols:
                imgui.same_line(0, gap)
            self._tile(t, key, title, icon, desc, soon, tw)
        imgui.dummy(ImVec2(0, 6))
        # runtime strip
        with cards.card("runtime", icon="bolt") as c:
            self._runtime_row(t, c.width)

    def _tile(self, t, key, title, icon, desc, soon, w):
        dl = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_pos()
        bh = 118.0
        imgui.begin_group()
        cards.plate(dl, pos.x, pos.y, w, bh, t,
                    edge=(t.text_mute if soon else t.accent))
        hov = imgui.is_mouse_hovering_rect(pos, ImVec2(pos.x + w, pos.y + bh)) and not soon
        self._profile_glyph(dl, icon, pos.x + 34, pos.y + 36, 16,
                            C(t.text_mute if soon else t.accent_bright)) \
            if icon in ("hand", "cube") else \
            icons.draw(icon, dl, pos.x + 34, pos.y + 36, 13,
                       C(t.text_mute if soon else t.accent_bright))
        with fonts.use("semi"):
            dl.add_text(ImVec2(pos.x + 62, pos.y + 22), C(t.text_mute if soon else t.text), title)
        if soon:
            with fonts.use("label"):
                dl.add_text(ImVec2(pos.x + 62, pos.y + 44), C(t.warn), "EM BREVE")
        imgui.set_cursor_screen_pos(ImVec2(pos.x + 18, pos.y + 66))
        imgui.push_text_wrap_pos(pos.x + w - 18)
        imgui.text_colored(t.text_dim, desc)
        imgui.pop_text_wrap_pos()
        # clickable surface
        imgui.set_cursor_screen_pos(pos)
        if imgui.invisible_button(f"##tile{key}", ImVec2(w, bh)) and not soon:
            self._open_tool(key)
        if hov:
            dl.add_rect(pos, ImVec2(pos.x + w, pos.y + bh),
                        C(theme.with_alpha(t.accent, 0.8)), t.rounding, 0, 2.0)
        imgui.end_group()

    def _open_tool(self, key):
        if key == "rp":
            self.enter_rp()
        elif self.capp is not None:
            self.capp.go(key)
        vui.sound.play("click")

    def _runtime_row(self, t, w):
        s = self.server
        labels = [m.label + ("" if m.present else "  (baixando)") for m in srv.MODELS]
        idx = next((i for i, m in enumerate(srv.MODELS) if m.key == self.model_key), 0)
        new = widgets.combo("", labels, idx, width=w - 150, key="rt_model")
        if new != idx:
            self.model_key = srv.MODELS[new].key
            self.save()
        imgui.same_line(0, 10)
        if s.state == "starting":
            if widgets.button("Cancelar", 130, key="rt_cancel"):
                s.stop()
        elif s.ready:
            if widgets.button("Desligar", 130, key="rt_stop", enabled=not self.agent.busy):
                s.stop()
        else:
            if widgets.button("Ligar modelo", 130, primary=True, key="rt_start",
                              enabled=srv.model(self.model_key).present and not self.agent.busy):
                self.server.start(self.model_key, self.ctx)
        self._status()

    def _status(self):
        """The status strip the layout places (dense: top bar; tabbed/rail:
        header). Compact, drawn inline so any layout can host it."""
        t = theme.current()
        s = self.server
        sm = self.sysmon
        tps = self.agent.last_tps if not self.rp_mode else self.rp.last_tps
        state = ("trabalhando" if (self.agent.busy or self.rp.busy) else
                 "carregando" if s.state == "starting" else
                 "pronto" if s.ready else "desligado")
        scol = t.ok if s.ready else (t.warn if s.state == "starting" else t.text_mute)
        used, cap = self.agent.ctx_used, max(1, s.ctx or self.ctx)
        imgui.text_colored(scol, state)
        imgui.same_line(0, 16)
        imgui.text_colored(t.danger if (tps and tps < 10 and sm.vram_tight) else t.text_dim,
                           f"{tps:.0f} tok/s" if tps else "-- tok/s")
        if sm.has_gpu:
            imgui.same_line(0, 16)
            imgui.text_colored(t.danger if sm.vram_tight else t.text_dim,
                               f"VRAM {sm.vram_used:.1f}/{sm.vram_total:.0f}G")
        imgui.same_line(0, 16)
        imgui.text_colored(t.text_dim, f"ctx {used // 1000}k/{cap // 1000}k")
        # discreet entrance to the RP side
        imgui.same_line(0, 16)
        if imgui.small_button("::"):
            self.enter_rp()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Roleplay")

    def _rp_overlay(self):
        """RP runs full-screen over the layout (the overlay slot draws last)."""
        if not self.rp_mode:
            return
        io = imgui.get_io()
        w, h = io.display_size.x, io.display_size.y
        t = theme.current()
        imgui.get_window_draw_list().add_rect_filled(ImVec2(0, 0), ImVec2(w, h), C(t.bg))
        imgui.set_cursor_screen_pos(ImVec2(14, 12))
        imgui.begin_child("rpoverlay", ImVec2(w - 28, h - 24))
        try:
            self._rp_frame(t, 14.0)
        finally:
            imgui.end_child()

    def _topbar(self, t):
        dl = imgui.get_window_draw_list()
        w = imgui.get_window_size().x
        h = 54.0
        o = imgui.get_cursor_screen_pos()
        prof = prompt.profile(self.profile)
        dl.add_rect_filled(o, ImVec2(o.x + w, o.y + h), C(theme.with_alpha(t.panel, 0.96)))
        dl.add_line(ImVec2(o.x, o.y + h), ImVec2(o.x + w, o.y + h),
                    C(theme.with_alpha(t.accent, 0.5)), 1.5)
        if self.rp_mode:
            cname = (self.rp.card.name if self.rp.card else "ROLEPLAY").upper()
            dl.add_circle_filled(ImVec2(o.x + 27, o.y + h / 2), 6, C(t.accent_bright))
            widgets.caps(dl, o.x + 50, o.y + 13, cname[:22], C(t.text), 2.0)
            with fonts.use("label"):
                widgets.caps(dl, o.x + 50, o.y + 33, "ROLEPLAY", C(t.text_dim), 2.6)
        else:
            self._profile_glyph(dl, prof["icon"], o.x + 27, o.y + h / 2, 11, C(t.accent_bright))
            widgets.caps(dl, o.x + 50, o.y + 13, prof["label"].upper(), C(t.text), 2.4)
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

        # right-hand readouts: speed, context fill, VRAM, RAM
        sm = self.sysmon
        tps = self.agent.last_tps if not self.rp_mode else self.rp.last_tps
        used, cap = self.agent.ctx_used, max(1, s.ctx or self.ctx)
        cfrac = min(1.0, used / cap)
        col = 115
        x = o.x + w - (col * 4 + 130)
        tight = sm.vram_tight and s.ready
        with fonts.use("label"):
            for i, lab in enumerate(("VELOCIDADE", "CONTEXTO", "VRAM", "RAM")):
                widgets.caps(dl, x + col * i, o.y + 10, lab, C(t.text_mute), 1.5)
        # speed: red when it has collapsed while VRAM is pinned
        slow = tps and tps < 10 and tight
        dl.add_text(ImVec2(x, o.y + 25), C(t.danger if slow else (t.text if tps else t.text_mute)),
                    f"{tps:.0f} tok/s" if tps else "--")
        dl.add_text(ImVec2(x + col, o.y + 25), C(t.warn if cfrac > 0.75 else t.text_dim),
                    f"{used // 1000}k/{cap // 1000}k")
        cards.meter(dl, x + col, x + col + 95, o.y + 42, cfrac,
                    t.warn if cfrac > 0.75 else t.accent, t, h=3.0)
        if sm.has_gpu:
            dl.add_text(ImVec2(x + col * 2, o.y + 25), C(t.danger if tight else t.text_dim),
                        f"{sm.vram_used:.1f}/{sm.vram_total:.0f}G")
            cards.meter(dl, x + col * 2, x + col * 2 + 95, o.y + 42, sm.vram_frac,
                        t.danger if tight else (t.warn if sm.vram_frac > 0.85 else t.accent), t, h=3.0)
            dl.add_text(ImVec2(x + col * 3, o.y + 25), C(t.text_dim),
                        f"{sm.ram_used:.0f}/{sm.ram_total:.0f}G")
            cards.meter(dl, x + col * 3, x + col * 3 + 95, o.y + 42,
                        sm.ram_used / max(1.0, sm.ram_total), t.accent, t, h=3.0)

        imgui.set_cursor_screen_pos(ImVec2(o.x + w - 200, o.y + 13))
        widgets.status_pill(word, word, on=on, key="srv")
        # the whole point: name the slowdown when VRAM is pinned
        if slow:
            with fonts.use("label"):
                msg = "VRAM CHEIA — CACHE NA RAM, POR ISSO LENTO. REDUZA O CONTEXTO OU USE O QWEN MoE."
                tw = widgets.caps_width(msg, 1.4)
                widgets.caps(dl, o.x + w - tw - 16, o.y + h - 12, msg, C(t.danger), 1.4)

        # Semi-hidden entrance to the RP side: a small unlabeled speech bubble
        # in the far corner. Dim but visible, it brightens and shows a tooltip
        # on hover - discreet, but findable now.
        bx, by, bw2, bh2 = o.x + w - 44, o.y + 13, 28.0, 26.0
        imgui.set_cursor_screen_pos(ImVec2(bx, by))
        imgui.invisible_button("##rpbubble", ImVec2(bw2, bh2))
        hov = imgui.is_item_hovered()
        self._rp_hint = anim.to("rpbubble", 1.0 if hov else 0.0, 12.0)
        if self._rp_hint > 0.01:
            dl.add_rect_filled(ImVec2(bx, by), ImVec2(bx + bw2, by + bh2),
                               C(theme.with_alpha(t.accent, 0.16 * self._rp_hint)), 6.0)
        col = theme.lerp(theme.with_alpha(t.text_mute, 0.7), t.accent_bright, self._rp_hint)
        self._bubble_glyph(dl, bx + bw2 / 2, by + bh2 / 2 - 1, 8.0, C(col))
        if hov:
            imgui.set_tooltip("Roleplay")
        if imgui.is_item_clicked():
            self.enter_rp()

        imgui.set_cursor_screen_pos(ImVec2(o.x, o.y + h))

    def _bubble_glyph(self, dl, cx, cy, r, col):
        """A little rounded speech bubble with a tail and two dots."""
        x0, y0, x1, y1 = cx - r, cy - r * 0.78, cx + r, cy + r * 0.5
        dl.add_rect_filled(ImVec2(x0, y0), ImVec2(x1, y1), col, r * 0.5)
        dl.add_triangle_filled(ImVec2(cx - r * 0.45, y1 - 1), ImVec2(cx - r * 0.05, y1 - 1),
                               ImVec2(cx - r * 0.75, y1 + r * 0.55), col)
        bg = theme.current().panel
        for dx in (-r * 0.4, 0.0, r * 0.4):
            dl.add_circle_filled(ImVec2(cx + dx, cy - r * 0.14), r * 0.12, C(bg))

    # -- agent tab -------------------------------------------------------

    def _win_begin(self, t, title, w, h, key, right=None, right_col=None):
        """Frame a region as one of the theme's windows (plate + title bar) and
        open a transparent scroll child inside it, so the main column reads as
        panels of the same kind as the sidebar instead of loose content."""
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        cards.plate(dl, p.x, p.y, w, h, t)
        if title:
            cards.card_head(dl, p.x, p.y, w, t, None, title, right, right_col)
        ip = 10.0
        top = (cards.HEAD_H if title else ip)
        imgui.set_cursor_screen_pos(ImVec2(p.x + ip, p.y + top))
        imgui.push_style_color(imgui.Col_.child_bg, ImVec4(0, 0, 0, 0))
        imgui.begin_child(key, ImVec2(w - ip * 2, h - top - ip))
        return p

    def _win_end(self, p, w, h):
        imgui.end_child()
        imgui.pop_style_color()
        # the child only spans the inset; anchor the cursor just below the full
        # plate so the next item starts there (0-height so no double spacing)
        imgui.set_cursor_screen_pos(ImVec2(p.x, p.y + h))
        imgui.dummy(ImVec2(w, 0.0))

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
        input_h = 118.0
        pending = next((i for i in self.agent.items if i.kind == "tool" and i.status == "approve"), None)
        banner = 64.0 if pending else 0.0
        chat_h = h - input_h - banner - 16
        n = sum(1 for i in self.agent.items if i.kind == "tool")
        right = f"{self.agent.steps} passos" if self.agent.items else None
        p = self._win_begin(t, "transcrição", main_w, chat_h, "chat", right=right,
                            right_col=t.text_dim)
        try:
            self._transcript(t, main_w - 24)
        finally:
            self._win_end(p, main_w, chat_h)
        if pending:
            imgui.dummy(ImVec2(0, 6))
            self._approval_banner(t, pending, main_w)
        imgui.dummy(ImVec2(0, 6))
        self._input(t, main_w, input_h)
        imgui.end_group()

    def _side(self, t):
        s = self.server
        self._profile_card(t)
        imgui.dummy(ImVec2(0, 4))
        self._look_card(t)
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
            # A loaded RP model writes weak code. Warn and offer a one-click
            # swap to the right coding model for this profile.
            if s.ready and srv.model(s.model_key or "").rp:
                cm = self._coding_model()
                note(f"Modelo de RP carregado — ele escreve código fraco. Para código, "
                     f"use o {cm.label}.", tone=t.danger)
                if widgets.button(f"Trocar p/ {cm.label}", c.width, primary=True,
                                  key="to_coding", enabled=cm.present and not self.agent.busy):
                    self.model_key, self.ctx = cm.key, cm.ctx
                    self.save()
                    self.server.start(cm.key, cm.ctx)
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
            widgets.spinner(20, key="work_spin", speed=1.2)
            imgui.same_line(0, 10)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 3)
            imgui.text_colored(t.text_mute, self.agent.live_text() or "trabalhando")
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

    def _input(self, t, outer_w, outer_h):
        # frame the input area as a theme window, like the panels around it
        dl = imgui.get_window_draw_list()
        wp = imgui.get_cursor_screen_pos()
        cards.plate(dl, wp.x, wp.y, outer_w, outer_h, t)
        ip = 10.0
        imgui.set_cursor_screen_pos(ImVec2(wp.x + ip, wp.y + ip))
        w = outer_w - ip * 2
        field_h = outer_h - ip - 40          # leave a clear row under it (attach + hint)
        bw = 112.0
        if self.focus_prompt:
            imgui.set_keyboard_focus_here()
            self.focus_prompt = False
        flags = (imgui.InputTextFlags_.enter_returns_true
                 | imgui.InputTextFlags_.ctrl_enter_for_new_line)
        imgui.push_style_color(imgui.Col_.frame_bg, t.surface)
        sent, self.prompt = imgui.input_text_multiline(
            "##prompt", self.prompt, ImVec2(w - bw - 10, field_h), flags)
        imgui.pop_style_color()
        h = field_h + 22    # for the button height below, matching the field
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
                self.agent.base = self.server.base   # port may have moved
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

    # -- RP (hidden) -----------------------------------------------------

    def enter_rp(self):
        self.rp_mode = True
        self.rp.base = self.server.base
        self.rp_cards = rp_mod.list_cards(self.rp_dir)
        self.rp_focus = True
        self.save()
        vui.sound.play("click")

    def exit_rp(self):
        self.rp.stop()
        self.rp_mode = False
        vui.sound.play("click")

    def _rp_frame(self, t, pad):
        avail = imgui.get_content_region_avail()
        h = avail.y - 12
        # left: characters
        imgui.begin_child("rpside", ImVec2(SIDE_W, h))
        try:
            self._rp_side(t)
        finally:
            imgui.end_child()
        imgui.same_line(0, 14)
        imgui.begin_group()
        main_w = avail.x - SIDE_W - 14 - pad
        input_h = 92.0
        chat_h = h - input_h - 16
        title = (self.rp.card.name if self.rp.card else "roleplay")
        p = self._win_begin(t, title, main_w, chat_h, "rpchat",
                            right=f"{len(self.rp.turns)} msgs" if self.rp.turns else None,
                            right_col=t.text_dim)
        try:
            self._rp_chat(t, main_w - 24)
        finally:
            self._win_end(p, main_w, chat_h)
        imgui.dummy(ImVec2(0, 6))
        self._rp_input(t, main_w, input_h)
        imgui.end_group()

    def _rp_side(self, t):
        with cards.card("roleplay", icon=None, right="sair", right_col=t.text_dim) as c:
            if widgets.button("<  Voltar ao código", c.width, key="rp_back"):
                self.exit_rp()
            note("Chat de personagem, sem ferramentas. Conteúdo gerado pelo modelo "
                 "local (use o abliterated). Ficção adulta.")

        imgui.dummy(ImVec2(0, 4))
        with cards.card("personagens", icon="folder") as c:
            imgui.set_next_item_width(c.width)
            changed, val = imgui.input_text("##rpdir", self.rp_dir)
            if changed:
                self.rp_dir = val
            if imgui.is_item_deactivated_after_edit():
                self.rp_cards = rp_mod.list_cards(self.rp_dir)
                self.save()
            bw = (c.width - 10) / 2
            if widgets.button("Pasta", bw, key="rp_pick"):
                self._rp_pick_folder()
            imgui.same_line(0, 10)
            if widgets.button("Recarregar", bw, key="rp_reload"):
                self.rp_cards = rp_mod.list_cards(self.rp_dir)
            imgui.dummy(ImVec2(0, 4))
            if not self.rp_cards:
                note("Nenhum card .json/.png aqui. Aponte para a pasta dos seus personagens.")
            for f in self.rp_cards[:200]:
                active = self.rp.card is not None and self.rp.card.path == str(f)
                if widgets.button(f.stem[:26], c.width, primary=active,
                                  key="rpc:" + str(f), enabled=not self.rp.busy):
                    self._rp_load(f)

        imgui.dummy(ImVec2(0, 4))
        imgui.dummy(ImVec2(0, 4))
        with cards.card("cena", icon=None) as c:
            with fonts.use("label"):
                imgui.text_colored(t.text_mute, "NOTA DE AUTOR (DIREÇÃO, SEM VIRAR FALA)")
            imgui.push_style_color(imgui.Col_.frame_bg, t.surface)
            chn, vn = imgui.input_text_multiline("##authnote", self.rp.author_note,
                                                 ImVec2(c.width, 56),
                                                 imgui.InputTextFlags_.word_wrap)
            imgui.pop_style_color()
            if chn:
                self.rp.author_note = vn
            self.rp.temperature = widgets.slider("criatividade", self.rp.temperature,
                                                 0.3, 1.3, width=c.width - 96, fmt="%.2f",
                                                 key="rp_temp")
            tl = {280: "curta", 420: "média", 640: "longa", 900: "muito longa"}
            opts = [280, 420, 640, 900]
            ci = min(range(len(opts)), key=lambda i: abs(opts[i] - self.rp.max_tokens))
            nci = widgets.combo("resposta", [tl[o] for o in opts], ci, width=c.width - 90, key="rp_len")
            self.rp.max_tokens = opts[nci]
            note("A nota de autor manda mais que a conversa. Criatividade alta varia "
                 "mais; baixa fica mais previsível.")

        imgui.dummy(ImVec2(0, 4))
        with cards.card("lorebook", icon=None) as c:
            if self.rp.lore and self.rp.lore.entries:
                n = len(self.rp.lore.entries)
                imgui.text_colored(t.text_dim, f"{n} entrada(s)" +
                                   (f" · {self.rp.lore_hits} ativa(s) agora" if self.rp.lore_hits else ""))
            else:
                imgui.text_colored(t.text_mute, "nenhuma (o card pode trazer a sua)")
            if widgets.button("Carregar lorebook (.json)", c.width, key="rp_lore"):
                self._rp_pick_lore()
            note("World Info: fatos do mundo injetados quando a conversa cita as "
                 "palavras-chave. O card v2 já traz o seu, se tiver.")

        imgui.dummy(ImVec2(0, 4))
        with cards.card("seu personagem", icon=None) as c:
            # saved personas
            if self.rp_personas:
                names = [p.get("name", "?") for p in self.rp_personas]
                self._rp_pidx = min(self._rp_pidx, len(names) - 1)
                ni = widgets.combo("", names, self._rp_pidx, width=c.width, key="rp_persona")
                if ni != self._rp_pidx:
                    self._rp_pidx = ni
                    self._rp_apply_persona(self.rp_personas[ni].get("name", "Você"),
                                           self.rp_personas[ni].get("desc", ""))
            with fonts.use("label"):
                imgui.text_colored(t.text_mute, "NOME")
            imgui.set_next_item_width(c.width)
            ch, val = imgui.input_text("##rpuser", self.rp_user)
            if ch:
                self.rp_user = val or "Você"
                self.rp.user_name = self.rp_user
            if imgui.is_item_deactivated_after_edit():
                self.save()
            with fonts.use("label"):
                imgui.text_colored(t.text_mute, "QUEM VOCÊ É (APARÊNCIA, HISTÓRIA, JEITO)")
            imgui.push_style_color(imgui.Col_.frame_bg, t.surface)
            chd, vd = imgui.input_text_multiline("##rpdesc", self.rp_user_desc,
                                                 ImVec2(c.width, 140),
                                                 imgui.InputTextFlags_.word_wrap)
            imgui.pop_style_color()
            if chd:
                self.rp_user_desc = vd
                self.rp.user_desc = vd
            if imgui.is_item_deactivated_after_edit():
                self.save()
            bw = (c.width - 10) / 2
            if widgets.button("Salvar", bw, key="rp_psave", enabled=bool(self.rp_user.strip())):
                self._rp_save_persona()
            imgui.same_line(0, 10)
            if widgets.button("Do card", bw, key="rp_pcard"):
                self._rp_persona_from_card()
            if self.rp_personas and widgets.button("Excluir esta", c.width, key="rp_pdel"):
                self.rp_personas = [p for p in self.rp_personas if p.get("name") != self.rp_user]
                self._rp_pidx = 0
                self.save()
            note("O personagem conhece você por isto e nunca fala ou age por você. "
                 "Vale na próxima resposta, sem reiniciar.")

        imgui.dummy(ImVec2(0, 4))
        with cards.card("modelo", icon=None) as c:
            cur = self.server.model_key or ""
            is_rp = srv.model(cur).rp if cur else False
            tone = t.ok if is_rp else t.warn
            m = srv.model(cur)
            imgui.text_colored(tone, "modelo: " + (m.label if cur else "-"))
            rpm = srv.rp_model()
            if not is_rp and not self.agent.busy:
                lbl = f"Ligar {rpm.label}" if rpm.present else f"{rpm.label} (baixando)"
                if widgets.button(lbl, c.width, primary=True, key="rp_switch",
                                  enabled=rpm.present):
                    self.model_key = rpm.key
                    self.ctx = rpm.ctx
                    self.save()
                    self.server.start(rpm.key, rpm.ctx)
                note("O Cydonia é tunado para roleplay; o Qwen escreve mais seco.")
            if self.rp.card is not None:
                if widgets.button("Reiniciar conversa", c.width, key="rp_restart",
                                  enabled=not self.rp.busy):
                    self.rp.load(self.rp.card)

    def _rp_apply_persona(self, name, desc):
        self.rp_user = name or "Você"
        self.rp_user_desc = desc or ""
        self.rp.user_name = self.rp_user
        self.rp.user_desc = self.rp_user_desc
        self.save()

    def _rp_save_persona(self):
        """Upsert by name, so saving an edited persona replaces the old one."""
        entry = {"name": self.rp_user.strip(), "desc": self.rp_user_desc}
        others = [p for p in self.rp_personas if p.get("name") != entry["name"]]
        self.rp_personas = others + [entry]
        self._rp_pidx = len(self.rp_personas) - 1
        self.save()
        vui.sound.play("ok")

    def _rp_persona_from_card(self):
        try:
            from imgui_bundle import portable_file_dialogs as pfd
            r = pfd.open_file("Seu personagem (card .json/.png)", self.rp_dir,
                              ["Cards", "*.json *.png"]).result()
        except Exception:
            r = []
        if r:
            try:
                name, desc = rp_mod.load_persona(r[0])
                self._rp_apply_persona(name, desc)
            except Exception as e:
                self.rp.turns.append({"role": "assistant", "text": f"[card inválido: {e}]", "think": ""})

    def _rp_pick_lore(self):
        try:
            from imgui_bundle import portable_file_dialogs as pfd
            r = pfd.open_file("Lorebook / World Info", self.rp_dir,
                              ["JSON", "*.json"]).result()
        except Exception:
            r = []
        if r:
            try:
                self.rp.attach_lorebook(rp_mod.load_lorebook(r[0]))
            except Exception as e:
                self.rp.turns.append({"role": "assistant", "text": f"[lorebook inválido: {e}]", "think": ""})

    def _rp_pick_folder(self):
        try:
            from imgui_bundle import portable_file_dialogs as pfd
            r = pfd.select_folder("Pasta dos personagens", self.rp_dir).result()
        except Exception:
            r = ""
        if r:
            self.rp_dir = r
            self.rp_cards = rp_mod.list_cards(r)
            self.save()

    def _rp_load(self, path):
        try:
            card = rp_mod.load_card(path)
        except Exception as e:
            self.rp.turns = [{"role": "assistant", "text": f"[não consegui ler o card: {e}]", "think": ""}]
            self.rp.card = None
            return
        self.rp.user_name = self.rp_user
        self.rp.load(card)
        self.rp_focus = True

    def _avatar(self, dl, cx, cy, r, name, col, path=""):
        """A round avatar: the card image if present, else a monogram disc."""
        tex = self._avatar_tex(path) if path else None
        if tex:
            tid, iw, ih = tex
            # square-crop centred, drawn into a circle via a clip-ish ring
            dl.add_image_rounded(imgui.ImTextureRef(int(tid)), ImVec2(cx - r, cy - r),
                                 ImVec2(cx + r, cy + r), ImVec2(0, 0), ImVec2(1, 1),
                                 0xFFFFFFFF, r)
            dl.add_circle(ImVec2(cx, cy), r, C(theme.with_alpha(col, 0.9)), 32, 2.0)
        else:
            dl.add_circle_filled(ImVec2(cx, cy), r, C(theme.with_alpha(col, 0.22)))
            dl.add_circle(ImVec2(cx, cy), r, C(theme.with_alpha(col, 0.8)), 32, 1.8)
            ch = (name or "?").strip()[:1].upper()
            with fonts.use("semi"):
                sz = imgui.calc_text_size(ch)
                dl.add_text(ImVec2(cx - sz.x / 2, cy - sz.y / 2), C(col), ch)

    def _avatar_tex(self, path):
        if path in self._tex:
            return self._tex[path]
        out = None
        try:
            data = Path(path).read_bytes()
            r = hello_imgui.image_and_size_from_encoded_data(data)
            # returns (TextureGpu-ish, size) or (texid, w, h) depending on build
            if isinstance(r, (tuple, list)) and len(r) >= 2:
                tid = r[0]
                sz = r[-1]
                out = (int(tid) if not hasattr(tid, "texture_id") else int(tid.texture_id),
                       getattr(sz, "x", 0), getattr(sz, "y", 0))
        except Exception:
            out = None
        self._tex[path] = out
        return out

    def _rp_chat(self, t, w):
        if self.rp.card is None:
            imgui.dummy(ImVec2(0, 40))
            with fonts.use("title"):
                imgui.text("Escolha um personagem")
            imgui.text_colored(t.text_dim, "Selecione um card na coluna da esquerda para começar.")
            return
        at_bottom = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 40
        av = 38.0                       # avatar diameter
        gap = 12.0
        inner = w - 28 - av - gap
        last = len(self.rp.turns) - 1
        for n, turn in enumerate(self.rp.turns):
            imgui.push_id(n)
            try:
                self._rp_bubble(t, n, turn, inner, av, gap, n == last)
            finally:
                imgui.pop_id()
            imgui.dummy(ImVec2(0, 6))
        if self.rp.busy and (not self.rp.turns or self.rp.turns[-1]["role"] == "user"):
            widgets.spinner(18, key="rp_spin", speed=1.0)
            imgui.same_line(0, 10)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 2)
            imgui.text_colored(t.text_mute, f"{self.rp.card.name} está digitando")
        imgui.dummy(ImVec2(0, 8))
        if at_bottom or self.rp.busy:
            imgui.set_scroll_here_y(1.0)

    def _rp_bubble(self, t, n, turn, inner, av, gap, is_last):
        is_user = turn["role"] == "user"
        who = self.rp_user if is_user else self.rp.card.name
        col = t.accent_bright if is_user else t.ok
        edge = t.accent if is_user else t.ok
        path = "" if is_user else (self.rp.card.avatar or "")
        text = self.rp.text_of(turn) or ("..." if self.rp.busy and is_last else "")

        row_x = imgui.get_cursor_screen_pos().x
        dl = imgui.get_window_draw_list()
        # avatar gutter on the left
        self._avatar(dl, row_x + av / 2, imgui.get_cursor_screen_pos().y + av / 2 + 6,
                     av / 2, who, col, path)
        imgui.indent(av + gap)
        with cards.card(None, width=inner, edge=edge, pad=ImVec2(14, 10)):
            with fonts.use("label"):
                imgui.text_colored(col, who.upper()[:28])
            editing = self._rp_edit == n
            if editing:
                imgui.set_next_item_width(inner - 28)
                chg, val = imgui.input_text_multiline("##edit", self._rp_edit_buf,
                                                      ImVec2(inner - 28, 120),
                                                      imgui.InputTextFlags_.word_wrap)
                if chg:
                    self._rp_edit_buf = val
                if widgets.button("Salvar", 90, key="ed_ok"):
                    self.rp.set_text(n, self._rp_edit_buf)
                    self._rp_edit = -1
                imgui.same_line(0, 8)
                if widgets.button("Cancelar", 100, key="ed_no"):
                    self._rp_edit = -1
            else:
                self._rp_text(t, text, inner - 28)
                self._rp_controls(t, n, turn, is_user, is_last, inner - 28)
        imgui.unindent(av + gap)

    def _rp_controls(self, t, n, turn, is_user, is_last, w):
        """A thin row of per-message actions that only shows on hover (or on the
        last message), so the transcript stays clean."""
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_blocked_by_active_item) \
            or imgui.is_mouse_hovering_rect(imgui.get_item_rect_min(), imgui.get_item_rect_max())
        show = is_last or self._rp_ctrl_hot == n
        if hovered:
            self._rp_ctrl_hot = n
        if not show:
            return
        imgui.dummy(ImVec2(0, 2))
        busy = self.rp.busy
        # swipe counter on the last assistant message
        if not is_user and turn.get("alts") and (len(turn["alts"]) > 1 or is_last):
            cur, tot = turn.get("alt", 0) + 1, len(turn["alts"])
            if imgui.small_button("<") and not busy:
                self.rp.swipe(n, -1)
            imgui.same_line(0, 4)
            imgui.text_colored(t.text_dim, f"{cur}/{tot}")
            imgui.same_line(0, 4)
            if imgui.small_button(">") and not busy:
                self.rp.base = self.server.base
                self.rp.swipe(n, +1)
            imgui.same_line(0, 12)
        if imgui.small_button("editar"):
            self._rp_edit, self._rp_edit_buf = n, self.rp.text_of(turn)
        imgui.same_line(0, 6)
        if imgui.small_button("copiar"):
            imgui.set_clipboard_text(self.rp.text_of(turn))
        imgui.same_line(0, 6)
        if imgui.small_button("apagar") and not busy:
            self.rp.delete_turn(n)
        if not is_user and is_last and not busy:
            imgui.same_line(0, 6)
            if imgui.small_button("continuar"):
                self.rp.base = self.server.base
                self.rp.continue_last()

    def _rp_text(self, t, text, w):
        """RP prose: *action* lines dimmed/italic-ish, plain speech normal."""
        imgui.push_text_wrap_pos(imgui.get_cursor_pos().x + w)
        for seg, is_action in self._split_actions(text):
            if is_action:
                imgui.text_colored(t.text_dim, seg)
            else:
                imgui.text_unformatted(seg)
        imgui.pop_text_wrap_pos()

    @staticmethod
    def _split_actions(text):
        # split on *...* so narration reads apart from speech
        out, i = [], 0
        for m in re.finditer(r"\*([^*]+)\*", text):
            if m.start() > i:
                out.append((text[i:m.start()], False))
            out.append((m.group(1), True))
            i = m.end()
        if i < len(text):
            out.append((text[i:], False))
        return out or [(text, False)]

    def _rp_input(self, t, outer_w, outer_h):
        bw = 108.0
        ready = self.server.ready and self.rp.card is not None
        dl = imgui.get_window_draw_list()
        wp = imgui.get_cursor_screen_pos()
        cards.plate(dl, wp.x, wp.y, outer_w, outer_h, t)
        ip = 10.0
        w = outer_w - ip * 2
        h = outer_h - ip * 2 + 22
        imgui.set_cursor_screen_pos(ImVec2(wp.x + ip, wp.y + ip))
        p = imgui.get_cursor_screen_pos()
        ic = 40.0
        # a chat-bubble emblem beside the field
        self._bubble_glyph(dl, p.x + ic / 2, p.y + (h - 22) / 2, 10.0,
                           C(t.accent_bright if ready else t.text_mute))
        imgui.set_cursor_screen_pos(ImVec2(p.x + ic + 8, p.y))

        if self.rp_focus:
            imgui.set_keyboard_focus_here()
            self.rp_focus = False
        flags = (imgui.InputTextFlags_.enter_returns_true
                 | imgui.InputTextFlags_.ctrl_enter_for_new_line)
        imgui.push_style_color(imgui.Col_.frame_bg, t.surface)
        sent, self.rp_prompt = imgui.input_text_multiline(
            "##rpprompt", self.rp_prompt, ImVec2(w - bw - ic - 18, h - 22), flags)
        imgui.pop_style_color()
        imgui.same_line(0, 10)
        if self.rp.busy:
            if widgets.button("Parar", bw, height=h - 22, key="rp_stop"):
                self.rp.stop()
        else:
            go = widgets.button("Enviar", bw, height=h - 22, primary=True,
                                key="rp_send", enabled=ready and bool(self.rp_prompt.strip()))
            if (go or sent) and ready and self.rp_prompt.strip():
                self.rp.base = self.server.base
                self.rp.send(self.rp_prompt.strip())
                self.rp_prompt = ""
                self.rp_focus = True
        hint = "Enter envia · Ctrl+Enter quebra linha · passe o mouse numa fala para editar/swipe"
        if not self.server.ready:
            hint = "Ligue o modelo para conversar."
        elif self.rp.card is None:
            hint = "Escolha um personagem à esquerda."
        imgui.text_colored(t.text_mute, hint)

    def _audio_tab(self, t, pad):
        a = self.audio
        avail = imgui.get_content_region_avail()
        h = avail.y - 12
        # left: controls
        imgui.begin_child("audioside", ImVec2(SIDE_W, h))
        try:
            self._audio_side(t, a)
        finally:
            imgui.end_child()
        imgui.same_line(0, 14)
        imgui.begin_group()
        main_w = avail.x - SIDE_W - 14 - pad
        input_h = 104.0
        var_h = 96.0 if a.vars else 0.0
        wave_h = h - input_h - var_h - 22 - (10 if a.vars else 0)
        p = self._win_begin(t, "forma de onda", main_w, wave_h, "wave",
                            right=(f"{a.duration:.2f}s" if a.audio is not None else None),
                            right_col=t.text_dim)
        try:
            self._audio_wave(t, a, main_w - 24, wave_h - cards.HEAD_H - 20)
        finally:
            self._win_end(p, main_w, wave_h)
        if a.vars:
            imgui.dummy(ImVec2(0, 6))
            p = self._win_begin(t, "variações", main_w, var_h, "vars")
            try:
                self._audio_vars(t, a, main_w - 24)
            finally:
                self._win_end(p, main_w, var_h)
        imgui.dummy(ImVec2(0, 6))
        self._audio_input(t, a, main_w, input_h)
        imgui.end_group()

    BACKENDS = [("procedural", "Procedural (instantâneo)"),
                ("sfx", "SFX neural (Stable Audio)"),
                ("music", "Música (Stable Audio)")]

    def _audio_side(self, t, a):
        with cards.card("motor", icon=None) as c:
            keys = [k for k, _ in self.BACKENDS]
            bi = keys.index(a.backend) if a.backend in keys else 0
            nbi = widgets.combo("", [lbl for _, lbl in self.BACKENDS], bi,
                                width=c.width, key="aud_backend")
            a.backend = keys[nbi]
            if a.backend == "procedural":
                note("Síntese procedural: instantânea, sem GPU. Cues de UI e SFX curtos.")
            else:
                a.neural.refresh_state()
                if not a.neural.available:
                    note("Modelo neural indisponível: o venv de IA (torch) não está "
                         "instalado. Rode build/ai setup.", tone=t.danger)
                else:
                    stx = {"off": "desligado", "starting": "iniciando",
                           "loading": "carregando modelo (~1 min na 1ª vez)",
                           "ready": "pronto" + (f" · {a.neural.device}" if a.neural.device else ""),
                           "generating": "gerando...", "error": "erro",
                           "missing": "indisponível"}.get(a.neural.state, a.neural.state)
                    tone = (t.danger if a.neural.state in ("error", "missing") else
                            t.warn if a.neural.state in ("loading", "starting") else t.ok)
                    imgui.text_colored(tone, "neural: " + stx)
                    a.seconds = widgets.slider("segundos", a.seconds, 1.0,
                                               47.0 if a.backend == "music" else 11.0,
                                               width=c.width - 90, fmt="%.0f", key="aud_secs")
                    a.steps = int(widgets.slider("passos", float(a.steps), 20, 100,
                                                 width=c.width - 90, fmt="%.0f", key="aud_steps"))
        imgui.dummy(ImVec2(0, 4))
        with cards.card("gerar", icon=None) as c:
            if a.backend == "procedural":
                note("Descreva o som: \"soft premium click\", \"glass shatter\". "
                     "Cada palavra molda o timbre.")
                roles = audio_mod.ROLES
                ri = roles.index(a.role) if a.role in roles else 0
                nri = widgets.combo("papel", roles, ri, width=c.width - 70, key="aud_role")
                a.role = roles[nri]
            elif a.backend == "music":
                note("Descreva a música: \"lo-fi hip hop beat, mellow, rhodes\", "
                     "\"synthwave, driving bass, 120 bpm\".")
            else:
                note("Descreva o efeito: \"heavy metal door slam\", \"laser zap\", "
                     "\"thunder rumble\". O modelo entende o prompt.")
            a.seed = imgui.slider_int("seed", a.seed, 0, 999)[1]
            a.volume = widgets.slider("volume", a.volume, 0.0, 1.0, width=c.width - 70,
                                      fmt="%.2f", key="aud_vol")
        imgui.dummy(ImVec2(0, 4))
        with cards.card("exportar", icon=None) as c:
            if widgets.button("Salvar WAV", c.width, key="aud_wav",
                              enabled=a.audio is not None):
                self._audio_save()
            note("Salva o som atual como .wav. Geração de música/áudio avançado "
                 "entra aqui em seguida.")
        if a.error:
            imgui.dummy(ImVec2(0, 4))
            note(a.error, tone=t.danger)

    def _audio_wave(self, t, a, w, h):
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        mid = p.y + h / 2
        dl.add_line(ImVec2(p.x, mid), ImVec2(p.x + w, mid),
                    C(theme.with_alpha(t.border, 0.6)), 1.0)
        wf = a.waveform(int(max(2, w)))
        if wf is None:
            txt = "Gere um som para ver a onda."
            tw = imgui.calc_text_size(txt).x
            dl.add_text(ImVec2(p.x + max(8.0, (w - tw) / 2), mid - 9),
                        C(t.text_dim), txt)
            imgui.dummy(ImVec2(w, h))
            return
        amp = h / 2 * 0.92
        step = w / max(1, len(wf))
        for i, v in enumerate(wf):
            x = p.x + i * step
            dl.add_line(ImVec2(x, mid - v * amp), ImVec2(x, mid + v * amp),
                        C(theme.with_alpha(t.accent, 0.85)), max(1.0, step))
        imgui.dummy(ImVec2(w, h))

    def _audio_vars(self, t, a, w):
        cols = len(a.vars)
        bw = (w - (cols - 1) * 8) / max(1, cols)
        for i, r in enumerate(a.vars):
            if i:
                imgui.same_line(0, 8)
            active = a.recipe is r
            if widgets.button(getattr(r, "name", str(i + 1))[:14], bw, height=40,
                              primary=active, key=f"var{i}"):
                a.pick_variation(i)

    def _audio_input(self, t, a, outer_w, outer_h):
        dl = imgui.get_window_draw_list()
        wp = imgui.get_cursor_screen_pos()
        cards.plate(dl, wp.x, wp.y, outer_w, outer_h, t)
        ip = 10.0
        imgui.set_cursor_screen_pos(ImVec2(wp.x + ip, wp.y + ip))
        w = outer_w - ip * 2
        field_h = outer_h - ip * 2 - 22
        bw = 108.0
        if self.audio_focus:
            imgui.set_keyboard_focus_here()
            self.audio_focus = False
        flags = imgui.InputTextFlags_.enter_returns_true
        imgui.push_style_color(imgui.Col_.frame_bg, t.surface)
        sent, a.prompt = imgui.input_text_multiline("##audprompt", a.prompt,
                                                    ImVec2(w - bw - 10, field_h), flags)
        imgui.pop_style_color()
        imgui.same_line(0, 10)
        imgui.begin_group()
        if a.busy:
            widgets.spinner(20, key="aud_spin")
        else:
            proc = a.backend == "procedural"
            gh = (field_h / 2 - 3) if proc else field_h
            if (widgets.button("Gerar", bw, height=gh, primary=True,
                               key="aud_gen", enabled=bool(a.prompt.strip())) or sent) \
                    and a.prompt.strip():
                a.generate()
                self.audio_focus = True
            if proc and widgets.button("Variações", bw, height=field_h / 2 - 3, key="aud_var",
                                       enabled=bool(a.prompt.strip())):
                a.variations()
        imgui.end_group()
        imgui.text_colored(t.text_mute, "Enter gera · o som toca sozinho · "
                           "Variações dá 6 opções para escolher")

    def _audio_save(self):
        try:
            from imgui_bundle import portable_file_dialogs as pfd
            default = (self.audio.recipe.name if self.audio.recipe else "som") + ".wav"
            r = pfd.save_file("Salvar WAV", default, ["WAV", "*.wav"]).result()
        except Exception:
            r = ""
        if r:
            self.audio.error = self.audio.export_wav(r)

    # -- image tool ------------------------------------------------------

    def _image_tab(self, t, pad):
        im = self.image
        avail = imgui.get_content_region_avail()
        h = avail.y - 12
        imgui.begin_child("imgside", ImVec2(SIDE_W, h))
        try:
            self._image_side(t, im)
        finally:
            imgui.end_child()
        imgui.same_line(0, 14)
        imgui.begin_group()
        main_w = avail.x - SIDE_W - 14 - pad
        input_h = 104.0
        prev_h = h - input_h - 16
        right = None
        if im.last_path is not None and im.image is not None:
            right = f"{im.image.width}x{im.image.height}"
        p = self._win_begin(t, "imagem", main_w, prev_h, "wave",
                            right=right, right_col=t.text_dim)
        try:
            self._image_preview(t, im, main_w - 24, prev_h - cards.HEAD_H - 20)
        finally:
            self._win_end(p, main_w, prev_h)
        imgui.dummy(ImVec2(0, 6))
        self._image_input(t, im, main_w, input_h)
        imgui.end_group()

    def _image_side(self, t, im):
        with cards.card("modelo", icon=None) as c:
            keys = [k for k, _ in image_mod.MODELS]
            mi = keys.index(im.model) if im.model in keys else 0
            nmi = widgets.combo("", [lbl for _, lbl in image_mod.MODELS], mi,
                                width=c.width, key="img_model")
            if keys[nmi] != im.model:
                im.set_model(keys[nmi])
            im.server.refresh_state()
            if not im.server.available:
                note("Modelo indisponível: o venv de IA (torch) não está instalado. "
                     "Rode: python -m lcoder.neural_setup", tone=t.danger)
            else:
                stx = {"off": "desligado", "starting": "iniciando",
                       "loading": "carregando modelo (1ª vez baixa os pesos)",
                       "ready": "pronto" + (f" · {im.server.device}" if im.server.device else ""),
                       "generating": "renderizando...", "error": "erro",
                       "missing": "indisponível"}.get(im.server.state, im.server.state)
                tone = (t.danger if im.server.state in ("error", "missing") else
                        t.warn if im.server.state in ("loading", "starting") else t.ok)
                imgui.text_colored(tone, "gpu: " + stx)
        imgui.dummy(ImVec2(0, 4))
        with cards.card("parâmetros", icon=None) as c:
            im.steps = int(widgets.slider("passos", float(im.steps), 1, 60,
                                          width=c.width - 90, fmt="%.0f", key="img_steps"))
            im.guidance = widgets.slider("guidance", im.guidance, 0.0, 15.0,
                                         width=c.width - 90, fmt="%.1f", key="img_guid")
            sizes = [512, 768, 1024]
            si = sizes.index(im.size) if im.size in sizes else len(sizes) - 1
            nsi = widgets.combo("tamanho", [f"{s}px" for s in sizes], si,
                                width=c.width - 90, key="img_size")
            im.size = sizes[nsi]
            im.seed = imgui.slider_int("seed", im.seed, 0, 999999)[1]
            note("seed 0 = aleatório a cada geração. SDXL rende em 1024; turbo "
                 "usa poucos passos e guidance 0.")
        imgui.dummy(ImVec2(0, 4))
        with cards.card("exportar", icon=None) as c:
            if widgets.button("Salvar PNG", c.width, key="img_png",
                              enabled=im.last_path is not None):
                self._image_save()
            note("As imagens também ficam em output/images/ automaticamente.")
        if im.error:
            imgui.dummy(ImVec2(0, 4))
            note(im.error, tone=t.danger)

    def _image_preview(self, t, im, w, h):
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        if im.last_path is None:
            txt = "Descreva uma imagem e gere."
            tw = imgui.calc_text_size(txt).x
            dl.add_text(ImVec2(p.x + max(8.0, (w - tw) / 2), p.y + h / 2 - 9),
                        C(t.text_dim), txt)
            imgui.dummy(ImVec2(w, h))
            return
        tex = self._avatar_tex(str(im.last_path))
        if not tex or not tex[1]:
            imgui.dummy(ImVec2(w, h))
            return
        tid, iw, ih = tex
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        ox, oy = p.x + (w - dw) / 2, p.y + (h - dh) / 2
        dl.add_image(tid, ImVec2(ox, oy), ImVec2(ox + dw, oy + dh))
        imgui.dummy(ImVec2(w, h))

    def _image_input(self, t, im, outer_w, outer_h):
        dl = imgui.get_window_draw_list()
        wp = imgui.get_cursor_screen_pos()
        cards.plate(dl, wp.x, wp.y, outer_w, outer_h, t)
        ip = 10.0
        imgui.set_cursor_screen_pos(ImVec2(wp.x + ip, wp.y + ip))
        w = outer_w - ip * 2
        field_h = outer_h - ip * 2 - 22
        bw = 108.0
        if self.image_focus:
            imgui.set_keyboard_focus_here()
            self.image_focus = False
        flags = imgui.InputTextFlags_.enter_returns_true
        imgui.push_style_color(imgui.Col_.frame_bg, t.surface)
        sent, im.prompt = imgui.input_text_multiline("##imgprompt", im.prompt,
                                                     ImVec2(w - bw - 10, field_h), flags)
        imgui.pop_style_color()
        imgui.same_line(0, 10)
        imgui.begin_group()
        if im.busy:
            widgets.spinner(20, key="img_spin")
        else:
            if (widgets.button("Gerar", bw, height=field_h, primary=True,
                               key="img_gen", enabled=bool(im.prompt.strip())) or sent) \
                    and im.prompt.strip():
                im.generate()
                self.image_focus = True
        imgui.end_group()
        imgui.text_colored(t.text_mute, "Enter gera · a 1ª geração baixa/carrega o "
                           "modelo (pode levar minutos) · salva em output/images/")

    def _image_save(self):
        if self.image.last_path is None:
            return
        try:
            from imgui_bundle import portable_file_dialogs as pfd
            r = pfd.save_file("Salvar PNG", "imagem.png", ["PNG", "*.png"]).result()
        except Exception:
            r = ""
        if r:
            try:
                import shutil
                shutil.copyfile(self.image.last_path, r)
                self.image.error = f"salvo: {r}"
            except Exception as e:
                self.image.error = f"salvar: {e}"

    def _server_tab(self, t, pad):
        s = self.server
        imgui.text_colored(t.text_dim, f"{s.base}  ·  estado: {s.state}  ·  modelo: {s.model_key or '-'}"
                           f"  ·  contexto: {s.ctx}")
        avail = imgui.get_content_region_avail()
        self.log.draw(ImVec2(avail.x - pad, avail.y - 12))

    def _settings_tab(self, t, pad):
        imgui.begin_child("settings", ImVec2(0, imgui.get_content_region_avail().y - 12))
        try:
            settings.panel(self.st, self.log, backdrop=self.bg, effects=self.fx,
                           on_change=lambda _s: (self.save(), self._apply_accent()))
        finally:
            imgui.end_child()

    def shutdown(self):
        self.agent.stop()
        self.agent.box.stop_all()
        self.server.stop()
        self.sysmon.stop()
        self.audio.neural.stop()
        self.image.stop()
        self.save()


class Shell(compose.Module):
    """Wires Local Coder's existing views into the compose architecture, so the
    profile's LAYOUT (tabs / rail / dense / windows) decides the app's shape -
    a genuinely different UI per theme, not a reskin. The state and all the
    drawing live on `App`; this just registers them as pages and slots."""
    name = "coder"

    def __init__(self, state: "App"):
        self.state = state

    def setup(self, app):
        self.state.capp = app
        s = self.state
        p = lambda fn: (lambda a: fn(theme.current(), 16.0))
        app.page("home", "Hub", icon="bolt", draw=p(s._home_page), order=-1)
        app.page("agent", "Agente", icon="bolt", draw=p(s._agent_tab), order=0)
        app.page("audio", "Áudio", icon="play", draw=p(s._audio_tab), order=1)
        app.page("image", "Imagem", icon="dot", draw=p(s._image_tab), order=2)
        app.page("server", "Servidor", icon="gear", draw=p(s._server_tab), order=3)
        app.page("settings", "Ajustes", icon="gear", draw=p(s._settings_tab), order=4)
        app.slot("status", lambda a: s._status())
        app.slot("overlay", lambda a: s._rp_overlay())

    def frame(self, app):
        self.state.pump_log()


def run():
    st = settings.Settings.load(app="localcoder")
    if not st.extras.get("inited"):
        st.theme_name = "noir-blue"
        st.extras["inited"] = True
    app = App(st)
    capp = compose.App(profile=profiles.get(app.look), title="LOCAL CODER",
                       app_id="localcoder", size=(1320, 880))
    capp.use(Shell(app))
    capp.run()
    app.shutdown()
