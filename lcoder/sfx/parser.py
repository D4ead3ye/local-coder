"""Prompt -> Recipe.

This file used to hold 58 hand-written recipes and pick one. That approach
cannot cover language - "thunderstorm" missed while "thunder" hit - so the
work now happens in two places and this is only the seam between them:

    character.read()   words  -> a sixteen-dimension character
    design.build()     character + role -> a recipe

Which means there is no list of supported sounds any more. There is a
vocabulary, and every combination of it is reachable.
"""

import re
from dataclasses import fields

import numpy as np

from . import character as C
from . import design
from .character import Reading           # re-exported: the app reads it
from .recipe import Recipe

SET_CUES = ("hover", "click", "toast", "toast_out", "ok", "warn", "error")
ROLES = design.ROLES


def slug(prompt, fallback="sound"):
    s = re.sub(r"[^a-z0-9]+", "_", (prompt or "").lower()).strip("_")
    return s[:34] or fallback


def analyse(prompt, role=None):
    """What the prompt was understood to mean. Never raises.

    A caller-supplied role wins outright. The role selector is a deliberate
    control that says what the cue is *for*, so "warm wooden confirm chime"
    asked for as a hover has to come back at hover length - otherwise
    picking a role in the UI silently does nothing whenever the description
    happens to name a different one.
    """
    r = C.read(prompt, default_role=role or "click")
    if role:
        r.role = role
    return r


def clamp(r: Recipe) -> Recipe:
    """Keep a recipe inside what the engine and the ear can use.

    The editor calls this after every drag, so it has to be cheap and it has
    to be total - a NaN here would take out the render thread.
    """
    for L in r.layers:
        L.f0 = float(np.clip(L.f0, 24.0, 16000.0))
        L.f1 = float(np.clip(L.f1, 24.0, 16000.0))
        L.dur = float(np.clip(L.dur, 0.006, 4.0))
        L.start = float(np.clip(L.start, 0.0, 3.5))
        L.cut0 = float(np.clip(L.cut0, 60.0, 20000.0))
        L.cut1 = float(np.clip(L.cut1, 60.0, 20000.0))
        L.gain = float(np.clip(L.gain, 0.0, 3.0))
        L.res = float(np.clip(L.res, 0.0, 3.0))
        L.decay = float(np.clip(L.decay, 0.15, 40.0))
        L.attack = float(np.clip(L.attack, 0.0002, max(0.0003, L.dur * 0.6)))
        L.damp = float(np.clip(L.damp, 0.9, 0.9999))
    r.drive = float(np.clip(r.drive, 0.0, 0.85))
    r.rev_mix = float(np.clip(r.rev_mix, 0.0, 0.65))
    r.echo_mix = float(np.clip(r.echo_mix, 0.0, 0.6))
    r.peak_db = float(np.clip(r.peak_db, -30.0, -0.5))
    r.width = float(np.clip(r.width, 0.0, 0.8))
    r.bits = int(np.clip(r.bits, 0, 16))
    if len(r.layers) > 8:
        del r.layers[8:]
    return r


_JITTER_SPREAD = {          # how far each feature may drift per variation
    "bright": .16, "pitch": .18, "metal": .14, "wood": .12, "body": .14,
    "grit": .12, "snap": .12, "ring": .16, "space": .12, "air": .12,
    "noise": .12, "motion": .14, "wet": .1, "warm": .12, "digital": .1,
    "grain": .12,
}
_ALT_GESTURE = {
    "hover": ("single", "single", "double"),
    "click": ("single", "single", "double"),
    # A toast is a panel arriving, so its variations stay swipes; a bell or
    # an arpeggio is a different idea, not a variation on the same one.
    "toast": ("slide", "slide", "slide", "bell"),
    "toast_out": ("slide_out", "slide_out", "slide_out", "fall2"),
    "ok": ("rise2", "arp", "bell"),
    "warn": ("fall2", "double", "single"),
    "error": ("fall2", "double", "single"),
}


def _drift(ch, rng, amount):
    """Move the character a little, in every dimension at once.

    Varying the recipe alone gives six versions of the same idea. Varying
    the character gives six ideas within the same description, which is what
    a row of variations is for.
    """
    out = ch
    kw = {}
    for f in fields(ch):
        spread = _JITTER_SPREAD.get(f.name)
        if not spread:
            continue
        v = getattr(ch, f.name) + float(rng.normal(0.0, spread * amount))
        kw[f.name] = float(np.clip(v, 0.0, 1.0))
    return out.nudged(**kw)


def parse(prompt, seed=0, jitter=0.0, role=None, reading=None):
    """The whole job: prompt in, ready-to-render Recipe out."""
    rd = reading or analyse(prompt, role=role)
    ch = rd.character
    gesture = rd.gesture
    if jitter > 0:
        rng = np.random.default_rng((int(seed) * 7919 + 13) & 0x7FFFFFFF)
        ch = _drift(ch, rng, jitter)
        if not rd.gesture and rng.random() < 0.45 * jitter:
            pool = _ALT_GESTURE.get(rd.role, ("single",))
            gesture = str(rng.choice(pool))
    r = design.build(ch, role=rd.role, gesture=gesture, seed=seed,
                     prompt=prompt, name=slug(prompt) or rd.role,
                     reverse=rd.reverse, materials=rd.materials,
                     exciter=rd.exciter)
    return clamp(r)


def variations(prompt, count=6, seed=0, role=None):
    """The first is the straight reading; the rest drift further as they go,
    so the row spans safe to adventurous."""
    rd = analyse(prompt, role=role)
    out = []
    for i in range(count):
        j = 0.0 if i == 0 else 0.3 + 0.75 * (i / max(count - 1, 1))
        r = parse(prompt, seed=seed + i, jitter=j, reading=rd)
        r.name = "%s_%d" % (slug(prompt) or rd.role, i + 1)
        out.append(r)
    return out


def make_set(prompt, seed=0, cues=SET_CUES):
    """A matched family: one character, every role.

    This is the payoff of separating the two. The description is read once
    and every cue is built from the same character, so the set sounds like
    one set rather than seven separate downloads that happen to share a
    word.
    """
    rd = analyse(prompt)
    out = {}
    for i, cue in enumerate(cues):
        sub = C.Reading(character=rd.character, role=cue, role_explicit=True,
                        gesture="" if rd.role_explicit else rd.gesture,
                        reverse=rd.reverse, known=rd.known,
                        unknown=rd.unknown, materials=rd.materials,
                        exciter=rd.exciter)
        r = parse(prompt, seed=seed + i, reading=sub)
        r.name = cue
        r.prompt = "%s -> %s" % (prompt, cue)
        out[cue] = r
    return out
