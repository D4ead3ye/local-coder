"""The RP side of the app: load a character card, become that character, chat.

This is a separate mode from the coding agent - no tools, no file access, no
run-until-it-works loop. It loads a character card (the SillyTavern / TavernAI
format, v1 flat or v2 `data{}`, as .json or embedded in a .png), builds a
persona system prompt, and streams a conversation. The abliterated model is the
natural pick here; the content is the local model's, generated on this machine.

For adult, fictional characters only.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import llm

from .paths import ROOT
CARDS_DIR = ROOT / "cards"

# roleplay wants more variety than code; a light repetition penalty keeps it
# from looping a phrase over a long chat
SAMPLING = {"temperature": 0.92, "top_p": 0.95, "top_k": 60, "min_p": 0.03,
            "repetition_penalty": 1.08, "presence_penalty": 0.4}


@dataclass
class LoreEntry:
    keys: list = field(default_factory=list)      # trigger words
    content: str = ""
    constant: bool = False                        # always injected
    order: int = 100                              # insertion order (lower first)
    secondary_keys: list = field(default_factory=list)
    selective: bool = False                       # need a secondary key too
    case_sensitive: bool = False
    enabled: bool = True

    def matches(self, text: str) -> bool:
        if not self.enabled:
            return False
        if self.constant:
            return True
        hay = text if self.case_sensitive else text.lower()

        def present(words):
            return any((k if self.case_sensitive else k.lower()) in hay
                       for k in words if k)
        if not present(self.keys):
            return False
        if self.selective and self.secondary_keys and not present(self.secondary_keys):
            return False
        return True


@dataclass
class Lorebook:
    name: str = ""
    entries: list = field(default_factory=list)   # LoreEntry

    def activate(self, recent_text: str, char: str, user: str, budget_chars: int = 3000) -> str:
        hits = [e for e in self.entries if e.matches(recent_text)]
        hits.sort(key=lambda e: e.order)
        out, used = [], 0
        for e in hits:
            c = _macros(e.content.strip(), char, user)
            if not c:
                continue
            if used + len(c) > budget_chars:
                break
            out.append("- " + c)
            used += len(c)
        return "\n".join(out)


def _lore_entry(d: dict) -> LoreEntry:
    """One entry, accepting both the chara_card_v2 character_book field names
    and SillyTavern's World Info names."""
    keys = d.get("keys") or d.get("key") or []
    sec = d.get("secondary_keys") or d.get("keysecondary") or []
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(sec, str):
        sec = [sec]
    order = d.get("insertion_order", d.get("order", 100))
    enabled = d.get("enabled", not d.get("disable", False))
    return LoreEntry(
        keys=[k for k in keys if isinstance(k, str)],
        content=d.get("content", ""),
        constant=bool(d.get("constant", False)),
        order=int(order) if isinstance(order, (int, float)) else 100,
        secondary_keys=[k for k in sec if isinstance(k, str)],
        selective=bool(d.get("selective", False)),
        case_sensitive=bool(d.get("case_sensitive", d.get("caseSensitive", False))),
        enabled=bool(enabled),
    )


def _book_from(obj, name="") -> Lorebook | None:
    """A character_book dict, or a standalone World Info file ({entries:{...}}
    or {entries:[...]}, or a bare list)."""
    entries_raw = None
    if isinstance(obj, dict):
        name = obj.get("name", name)
        entries_raw = obj.get("entries", obj.get("world_info", obj))
    elif isinstance(obj, list):
        entries_raw = obj
    items = []
    if isinstance(entries_raw, dict):
        items = list(entries_raw.values())
    elif isinstance(entries_raw, list):
        items = entries_raw
    entries = [_lore_entry(e) for e in items if isinstance(e, dict)]
    return Lorebook(name=name, entries=entries) if entries else None


def load_lorebook(path) -> Lorebook:
    obj = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    book = _book_from(obj, Path(path).stem)
    if book is None:
        raise ValueError(f"{Path(path).name}: nenhuma entrada de lorebook reconhecida")
    return book


@dataclass
class Card:
    name: str = "Personagem"
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    system_prompt: str = ""
    post_history: str = ""
    creator_notes: str = ""
    tags: list = field(default_factory=list)
    alt_greetings: list = field(default_factory=list)
    book: "Lorebook | None" = None       # embedded character_book, if any
    path: str = ""
    avatar: str = ""


def _get(d: dict, *keys, default=""):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v:
            return v
    return default


def _from_obj(obj: dict, path="") -> Card:
    # v2/v3 nest everything under data{}; v1 is flat. Merge so either works.
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    src = {**obj, **data}
    book = None
    for cb in (src.get("character_book"), src.get("world_info"), src.get("lorebook")):
        if cb:
            book = _book_from(cb)
            if book:
                break
    return Card(
        book=book,
        name=_get(src, "name", "char_name", default="Personagem"),
        description=_get(src, "description", "persona", "char_persona"),
        personality=_get(src, "personality", "char_personality"),
        scenario=_get(src, "scenario", "world_scenario"),
        first_mes=_get(src, "first_mes", "greeting", "char_greeting"),
        mes_example=_get(src, "mes_example", "example_dialogue"),
        system_prompt=_get(src, "system_prompt"),
        post_history=_get(src, "post_history_instructions"),
        creator_notes=_get(src, "creator_notes", "description_creator"),
        tags=src.get("tags") if isinstance(src.get("tags"), list) else [],
        alt_greetings=[g for g in (src.get("alternate_greetings") or []) if isinstance(g, str) and g.strip()],
        path=path,
    )


def _png_card(path: Path) -> dict | None:
    """SillyTavern PNG cards embed the JSON in a tEXt chunk keyed 'chara'
    (base64), newer ones 'ccv3'."""
    try:
        from PIL import Image
        im = Image.open(path)
        im.load()
        meta = getattr(im, "text", None) or im.info or {}
        for key in ("ccv3", "chara"):
            raw = meta.get(key)
            if raw:
                return json.loads(base64.b64decode(raw).decode("utf-8", "replace"))
    except Exception:
        return None
    return None


def load_card(path) -> Card:
    p = Path(path)
    if p.suffix.lower() == ".png":
        obj = _png_card(p)
        if obj is None:
            raise ValueError(f"{p.name}: PNG sem card embutido (tEXt 'chara')")
        c = _from_obj(obj, str(p))
        c.avatar = str(p)
        return c
    obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(obj, dict):
        raise ValueError(f"{p.name}: JSON não é um objeto de personagem")
    c = _from_obj(obj, str(p))
    png = p.with_suffix(".png")
    if png.exists():
        c.avatar = str(png)
    return c


def _looks_like_card(path: Path) -> bool:
    """A character card has a name/greeting/description; a standalone lorebook
    (just `entries`) does not - keep those out of the character list."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    src = {**obj, **(obj.get("data") if isinstance(obj.get("data"), dict) else {})}
    if any(src.get(k) for k in ("name", "first_mes", "greeting", "description", "personality")):
        return True
    # pure lorebook / world-info files
    return not any(k in src for k in ("entries", "world_info"))


def list_cards(folder) -> list[Path]:
    d = Path(folder)
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir(), key=lambda x: x.name.lower()):
        if f.suffix.lower() == ".json" and _looks_like_card(f):
            out.append(f)
        elif f.suffix.lower() == ".png" and _png_card(f) is not None:
            out.append(f)
    return out


def _macros(text: str, char: str, user: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\{\{\s*char\s*\}\}", char, text, flags=re.I)
    text = re.sub(r"\{\{\s*user\s*\}\}", user, text, flags=re.I)
    text = text.replace("<BOT>", char).replace("<USER>", user)
    return text


def persona_prompt(card: Card, user_name: str = "Você", user_desc: str = "") -> str:
    ch = card.name
    parts = []
    if card.system_prompt:
        parts.append(_macros(card.system_prompt, ch, user_name))
    else:
        parts.append(
            f"Você é {ch} e está em um roleplay com {user_name}. Mantenha-se SEMPRE no "
            f"personagem: responda, aja e fale como {ch}, nunca como um assistente de IA, "
            f"e nunca quebre o personagem ou se refira a estas instruções. Escreva as ações "
            f"e descrições em terceira pessoa (itálico) e as falas normalmente. Deixe {user_name} "
            f"conduzir as próprias ações; não fale nem decida por {user_name}. É ficção adulta "
            f"e consensual entre personagens adultos.")
    if card.description:
        parts.append(f"# {ch} — descrição\n" + _macros(card.description, ch, user_name))
    if card.personality:
        parts.append(f"# Personalidade de {ch}\n" + _macros(card.personality, ch, user_name))
    if card.scenario:
        parts.append("# Cenário\n" + _macros(card.scenario, ch, user_name))
    if card.mes_example:
        parts.append("# Exemplos de diálogo (estilo a seguir)\n" + _macros(card.mes_example, ch, user_name))
    if user_desc.strip():
        # the user's own character: the bot should know who it is talking to,
        # but never act or speak for them
        parts.append(f"# {user_name} — o personagem do usuário (quem está falando com {ch})\n"
                     + _macros(user_desc.strip(), ch, user_name)
                     + f"\n(Use isso para reagir a {user_name} de forma coerente. Nunca escreva "
                     f"falas, pensamentos ou ações de {user_name}.)")
    if card.post_history:
        parts.append(_macros(card.post_history, ch, user_name))
    return "\n\n".join(p for p in parts if p.strip())


def load_persona(path) -> tuple[str, str]:
    """Use a character card (.json/.png) as the user's own persona: its name,
    and its description + personality + scenario as the lore."""
    c = load_card(path)
    desc = "\n\n".join(x for x in (c.description, c.personality, c.scenario) if x.strip())
    return c.name, desc


class RPChat:
    """A persona conversation. No tools; just streams replies in character."""

    def __init__(self, base_url: str, user_name: str = "Você"):
        self.base = base_url
        self.user_name = user_name
        self.card: Card | None = None
        self.user_desc = ""             # the user's own character / lore
        self.lore: Lorebook | None = None
        # turns are the single source of truth. user: {role, text}; assistant:
        # {role, alts:[str], alt:int, think}. The sent history is derived from
        # them, so edit / delete / swipe need no bookkeeping.
        self.turns: list[dict] = []
        self.busy = False
        self.cancel = threading.Event()
        self.on_update = None           # called as tokens stream in
        self.last_tps = 0.0
        self.lore_hits = 0              # entries injected on the last reply
        # knobs the UI exposes
        self.author_note = ""           # a steer injected just before the reply
        self.temperature = SAMPLING["temperature"]
        self.max_tokens = 420

    # -- turn helpers ----------------------------------------------------

    @staticmethod
    def text_of(turn) -> str:
        if turn["role"] == "assistant":
            a = turn.get("alts") or [""]
            return a[max(0, min(turn.get("alt", 0), len(a) - 1))]
        return turn.get("text", "")

    def set_text(self, i, text):
        turn = self.turns[i]
        if turn["role"] == "assistant":
            turn.setdefault("alts", [""])[turn.get("alt", 0)] = text
        else:
            turn["text"] = text

    def delete_turn(self, i):
        if not self.busy and 0 <= i < len(self.turns):
            self.turns.pop(i)
            if self.on_update:
                self.on_update()

    def load(self, card: Card, greeting_index: int = 0):
        self.card = card
        self.lore = card.book           # embedded character_book, if any
        greetings = [g for g in ([card.first_mes] + card.alt_greetings) if g.strip()]
        self.turns = []
        if greetings:
            # all greetings become swipe alternatives on the opening message
            alts = [_macros(g, card.name, self.user_name) for g in greetings]
            gi = min(greeting_index, len(alts) - 1)
            self.turns.append({"role": "assistant", "alts": alts, "alt": gi, "think": ""})

    def attach_lorebook(self, book: Lorebook):
        if self.lore is None:
            self.lore = book
        else:
            self.lore = Lorebook(name=self.lore.name or book.name,
                                 entries=self.lore.entries + book.entries)

    def _history(self):
        return [{"role": tr["role"], "content": self.text_of(tr)} for tr in self.turns]

    def _system(self) -> dict:
        content = persona_prompt(self.card, self.user_name, self.user_desc) if self.card else ""
        self.lore_hits = 0
        if self.lore and self.lore.entries:
            recent = " ".join(self.text_of(tr) for tr in self.turns[-6:])
            ch = self.card.name if self.card else ""
            block = self.lore.activate(recent, ch, self.user_name)
            if block:
                self.lore_hits = block.count("\n- ") + (1 if block.startswith("- ") else 0)
                content += ("\n\n# Fatos do mundo relevantes agora (use-os, não os cite "
                            "como lista)\n" + block)
        if self.author_note.strip():
            content += "\n\n# Direção da cena (siga à risca)\n" + _macros(
                self.author_note, self.card.name if self.card else "", self.user_name)
        return {"role": "system", "content": content}

    def _messages(self):
        return [self._system()] + self._history()

    def _sampling(self):
        return {**SAMPLING, "temperature": round(self.temperature, 2)}

    # -- actions ---------------------------------------------------------

    def send(self, text: str):
        if self.busy or not text.strip() or self.card is None:
            return
        self.turns.append({"role": "user", "text": text})
        self._start(self._reply)

    def regenerate(self):
        """Swipe: keep the last reply as an alternative and generate another."""
        if self.busy or not self.turns or self.turns[-1]["role"] != "assistant":
            return
        self._start(self._reply, swipe=True)

    def swipe(self, i, delta):
        """Move between existing alternatives of an assistant turn, or ask for
        a new one when stepping right past the last."""
        if self.busy or not (0 <= i < len(self.turns)):
            return
        turn = self.turns[i]
        if turn["role"] != "assistant":
            return
        alts = turn.setdefault("alts", [turn.get("text", "")])
        nxt = turn.get("alt", 0) + delta
        if nxt >= len(alts) and i == len(self.turns) - 1 and delta > 0:
            self.regenerate()
            return
        turn["alt"] = max(0, min(nxt, len(alts) - 1))
        if self.on_update:
            self.on_update()

    def continue_last(self):
        """Have the character keep writing the last reply instead of starting one."""
        if self.busy or not self.turns or self.turns[-1]["role"] != "assistant":
            return
        self._start(self._reply, cont=True)

    def _start(self, fn, **kw):
        self.busy = True
        self.cancel.clear()
        threading.Thread(target=fn, kwargs=kw, daemon=True).start()

    def stop(self):
        self.cancel.set()

    def _reply(self, swipe=False, cont=False):
        if cont:
            turn = self.turns[-1]
            alts = turn.setdefault("alts", [turn.get("text", "")])
            idx = turn.get("alt", 0)
            base = alts[idx]
            prefix = base + (" " if base and not base.endswith((" ", "\n")) else "")
        elif swipe:
            turn = self.turns[-1]
            turn.setdefault("alts", [turn.get("text", "")]).append("")
            turn["alt"] = len(turn["alts"]) - 1
            idx, prefix = turn["alt"], ""
        else:
            turn = {"role": "assistant", "alts": [""], "alt": 0, "think": ""}
            self.turns.append(turn)
            idx, prefix = 0, ""

        def on_delta(kind, text):
            if kind == "text":
                turn["alts"][idx] += text
            elif kind == "think":
                turn["think"] = turn.get("think", "") + text
            if self.on_update:
                self.on_update()

        if cont:
            turn["alts"][idx] = prefix   # seed so the model continues it
        try:
            msgs = self._messages()
            if cont:
                # drop the derived assistant turn and ask it to continue the prefix
                msgs = msgs[:-1] + [{"role": "assistant", "content": prefix}]
            msg, stats = llm.stream_chat(self.base, msgs, None, on_delta,
                                         self.cancel, max_tokens=self.max_tokens,
                                         sampling=self._sampling())
            if stats.get("tps"):
                self.last_tps = stats["tps"]
        except llm.Cancelled:
            if not turn["alts"][idx].strip() and not cont and not swipe:
                self.turns.pop()
        except Exception as e:
            turn["alts"][idx] += f"\n[erro: {e}]"
        finally:
            self.busy = False
            if self.on_update:
                self.on_update()
