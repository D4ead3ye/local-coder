"""Character + material + role -> Recipe.

The previous version voiced everything with an oscillator and a pitch
glide, which is a beep however it is filtered, and left every prompt
sounding like a variation on the same beep. This one voices a cue as
**something happening to an object**: an excitation drives a resonant body
whose modes decay at material-specific rates.

Three rules hold it together:

  * The **role** owns time and level. A thunderstorm hover is still 55ms.
  * The **material** owns timbre, and is chosen outright rather than
    averaged - naming porcelain gives you porcelain.
  * The **exciter** owns what happened, independently of both. Tapping a
    glass and scraping a glass are obviously the same glass, and that only
    falls out for free if the two are never mixed together.

Oscillators still exist, for the one family of sounds that really is an
oscillator: chiptune, and anything else described as digital.
"""

import numpy as np

from . import engine as E
from . import material as M
from .character import Character
from .recipe import Layer, Recipe


def _lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def _log(a, b, t):
    return float(a) * (float(b) / float(a)) ** float(np.clip(t, 0.0, 1.0))


# Roles own duration, level, how much of the prompt's reverb they tolerate,
# and what is being done to the object when nothing says otherwise.
ROLES = {
    "hover": dict(dur=(0.025, 0.09), level=(-22.0, -15.0), space=0.30,
                  room=0.16, hard=-0.22, exciter="strike", gesture="single",
                  body="plastic"),
    "click": dict(dur=(0.035, 0.20), level=(-11.0, -6.0), space=0.45,
                  room=0.24, hard=0.10, exciter="strike", gesture="single",
                  body="plastic"),
    "toast": dict(dur=(0.28, 1.10), level=(-8.0, -4.0), space=1.0,
                  room=0.9, hard=-0.05, exciter="scrape", gesture="slide",
                  body="ceramic"),
    "toast_out": dict(dur=(0.22, 0.85), level=(-11.0, -7.0), space=0.9,
                      room=0.8, hard=-0.05, exciter="scrape",
                      gesture="slide_out", body="ceramic"),
    "ok": dict(dur=(0.26, 0.95), level=(-8.0, -4.0), space=0.95, room=0.8,
               hard=0.0, exciter="strike", gesture="rise2", body="ceramic"),
    "warn": dict(dur=(0.28, 1.0), level=(-8.0, -4.0), space=0.8, room=0.7,
                 hard=0.05, exciter="strike", gesture="fall2", body="ceramic"),
    "error": dict(dur=(0.30, 1.05), level=(-8.0, -4.0), space=0.75, room=0.7,
                  hard=0.05, exciter="strike", gesture="fall2", body="ceramic"),
}

# Notes are strikes on the *same object* at different pitches, so a
# two-note confirm is now an instrument being played rather than two beeps.
GESTURES = {
    "single": dict(notes=[(0.0, 1.0, 1.0)]),
    "double": dict(notes=[(0.0, 1.0, 1.0), (0.5, 1.0, 0.78)]),
    "rise2": dict(notes=[(0.0, 1.0, 1.0), (0.38, 1.4983, 0.92)]),
    "fall2": dict(notes=[(0.0, 1.0, 1.0), (0.38, 0.7492, 0.92)]),
    "arp": dict(notes=[(0.0, 1.0, 1.0), (0.24, 1.2599, 0.92),
                       (0.48, 1.4983, 0.88)]),
    "bell": dict(notes=[(0.0, 1.0, 1.0)], ring=1.8),
    "sweep": dict(notes=[], sustain=True, travel=1),
    "slide": dict(notes=[(0.62, 1.0, 0.7)], sustain=True, travel=1),
    "slide_out": dict(notes=[(0.0, 1.0, 0.7)], sustain=True, travel=-1),
}

SUSTAINED = ("scrape", "blow", "particles")


def pick_material(reading_materials, ch, fallback="plastic"):
    """Which object this is.

    Two named substances blend; one wins outright; none falls back to
    something the character implies rather than to a default beep.
    """
    names = [n for n in reading_materials if M.get(n)]
    if len(names) >= 2:
        return M.blend(M.get(names[0]), M.get(names[1]), 0.5)
    if names:
        return M.get(names[0])

    # Nothing named. Infer a body from the description, in the order a
    # listener would: ringing metal, glassy, plucked, woody, else plastic -
    # which is what most real interface hardware is anyway.
    if ch.metal > 0.62 and ch.ring > 0.55:
        return M.get("bell" if ch.ring > 0.75 else "metal")
    if ch.metal > 0.5:
        return M.get("glass" if ch.bright > 0.6 else "ceramic")
    if ch.wood > 0.45:
        return M.get("wood" if ch.ring > 0.4 else "woodblock")
    if ch.body > 0.7:
        return M.get("membrane")
    if ch.noise > 0.75:
        return M.get("paper")
    return M.get(fallback) or M.get("plastic")


def pick_exciter(named, ch, role_default):
    if named:
        return named
    if ch.grain > 0.45:
        return "particles"
    if ch.wet > 0.55:
        return "bubble"
    if ch.air > 0.6 and ch.noise > 0.6:
        return "blow"
    if ch.noise > 0.72:
        return "scrape"
    return role_default


def from_clip(clip, audio, role="click", seed=0, prompt="", name="sound",
              reverse=False):
    """Wrap a generated clip as a cue.

    The model decides what the sound is; the role still decides how long it
    is allowed to be and how loud it sits, because those are properties of
    the interface rather than of the description. The window is looser than
    for synthesis - the model's own timing is part of what was asked for -
    but a hover cue is still not allowed to run for three seconds.

    The envelope is deliberately almost flat. Reshaping a finished
    recording with a steep decay is how a good generation gets turned back
    into a synthesiser patch.
    """
    R = ROLES[role if role in ROLES else "click"]
    audible = len(E.trim_tail(np.asarray(audio))) / float(E.RATE)
    dur = float(np.clip(audible, R["dur"][0], R["dur"][1] * 1.6))
    layer = Layer(kind="sample", clip=clip, dur=dur, gain=1.0,
                  attack=0.0015, decay=0.35, sustain=0.75,
                  release=min(0.05, dur * 0.25), filt="none")
    r = Recipe(name=name, prompt=prompt, seed=int(seed), layers=[layer],
               reverse=bool(reverse))
    # Only what an interface needs: a level, and the rumble taken off.
    # Everything else is left alone unless the user reaches for it.
    r.peak_db = _lerp(R["level"][0], R["level"][1], 0.55)
    r.highpass = 32.0
    r.rev_mix = 0.0
    r.width = 0.0
    return r


def build(ch: Character, role="click", gesture="", seed=0, prompt="",
          name="sound", reverse=False, materials=(), exciter=""):
    role = role if role in ROLES else "click"
    R = ROLES[role]
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    gesture = gesture if gesture in GESTURES else R["gesture"]
    g = GESTURES[gesture]

    mat = pick_material(materials, ch, R.get("body", "plastic"))
    # A role's default action only applies to a gesture that actually moves.
    # Otherwise "cathedral bell" inherits the toast role's scrape and comes
    # back as a bell being rubbed rather than rung.
    exc = pick_exciter(exciter, ch,
                       R["exciter"] if g.get("sustain") else "strike")
    if exc in SUSTAINED and not g.get("sustain") and gesture == "single":
        # A scrape has no single moment to strike on; give it room to move.
        gesture, g = "sweep", GESTURES["sweep"]

    # -- how hard, how big, how long ---------------------------------------
    # Hardness is contact time, so it stays inside a range where the object
    # still speaks: a fully soft strike is a thud with no material in it.
    hardness = float(np.clip(0.30 + 0.66 * ch.snap + R["hard"], 0.24, 0.98))
    f0 = mat.f_ref * 2.0 ** ((ch.pitch - 0.5) * 2.4) * \
        2.0 ** (-(ch.body - 0.35) * 1.5)
    f0 = float(np.clip(f0, 55.0, 7000.0))

    # `ring` scales the material's own decay rather than replacing it, so a
    # long wooden knock is still unmistakably wood.
    damp = _lerp(0.35, 2.4, ch.ring) * float(g.get("ring", 1.0))
    natural = mat.tau * damp * (2.2 if exc not in SUSTAINED else 1.0)
    total = float(np.clip(natural, R["dur"][0], R["dur"][1]))
    if exc in SUSTAINED:
        total = float(np.clip(_lerp(R["dur"][0], R["dur"][1],
                                    0.35 + 0.55 * ch.ring),
                              R["dur"][0], R["dur"][1]))

    spread = 0.004 + 0.03 * ch.grit + 0.02 * (1.0 - ch.metal)
    layers = []

    def voice(start, mult, gain, dur, exc_kind, hard):
        return Layer(
            kind="modal", material=mat.name, exciter=exc_kind,
            f0=float(np.clip(f0 * mult, 40.0, 9000.0)),
            hardness=hard, mat_damp=damp, mat_spread=spread,
            gain=gain, start=start, dur=dur,
            attack=0.0006 if exc_kind not in SUSTAINED else max(
                0.004, dur * float(g.get("swell", 0.25))),
            decay=_lerp(9.0, 1.6, ch.ring),
            density=_log(30.0, 1800.0, max(ch.grain, 0.35)),
            grain_ms=_lerp(24.0, 5.0, ch.bright),
            mod_rate=_lerp(1.5, 9.0, ch.motion),
            filt="none", f1=float(np.clip(f0 * mult * 2.6, 60.0, 15000.0)))

    travel = int(g.get("travel", 0))
    if g.get("sustain"):
        # A sliding panel is movement plus an object. Which one carries the
        # sound depends on what the object is: dragging paper *is* the
        # sound, but a glass told to slide should still be struck and ring,
        # with the movement as air around it. Scraping a wine glass for a
        # notification is a squeak, not a notification.
        if exciter in SUSTAINED or mat.tau < 0.09:
            body = voice(0.0, 1.0, 0.95, total, exc, hardness)
            body.attack = max(0.004, total * (0.34 if travel > 0 else 0.05))
            body.decay = (_lerp(3.4, 1.5, ch.ring) if travel > 0
                          else _lerp(6.0, 2.4, ch.ring))
            body.mod_rate = _lerp(2.0, 12.0, ch.motion)
            layers.append(body)
        else:
            # Moving air, as a filter sweep: arriving opens, leaving closes.
            n_cut = _log(900.0, 9000.0, ch.bright)
            span = _lerp(2.0, 6.0, max(ch.motion, 0.45))
            layers.append(Layer(
                kind="noise", noise_kind="pink", dur=total * 0.95,
                gain=_lerp(0.4, 0.85, max(ch.air, ch.noise, 0.4)),
                attack=max(0.004, total * (0.34 if travel > 0 else 0.05)),
                decay=_lerp(3.4, 1.6, ch.ring), filt="bp", res=0.9,
                cut0=n_cut / span if travel > 0 else n_cut,
                cut1=n_cut if travel > 0 else n_cut / span))
        # ...and the moment it arrives, or lets go.
        for at, mult, gain in g["notes"]:
            layers.append(voice(at * total * 0.92, mult, gain * 0.95,
                                max(0.06, total * 0.55), "strike",
                                min(0.98, hardness + 0.12)))
    else:
        for i, (at, mult, gain) in enumerate(g["notes"]):
            start = at * total
            layers.append(voice(start, mult, gain,
                                max(0.03, total - start), exc, hardness))

    # -- digital is genuinely an oscillator, so keep one for it ------------
    if ch.digital > 0.5:
        for at, mult, gain in (g["notes"] or [(0.0, 1.0, 1.0)]):
            layers.append(Layer(
                kind="tone", wave="square", bright=0.55,
                f0=float(np.clip(f0 * mult, 60.0, 6000.0)),
                f1=float(np.clip(f0 * mult, 60.0, 6000.0)),
                dur=max(0.03, total * 0.55), start=at * total,
                gain=0.55 * ch.digital * gain, attack=0.0008,
                decay=_lerp(9.0, 3.0, ch.ring), filt="lp",
                cut0=_log(2500, 12000, ch.bright), cut1=_log(1500, 7000,
                                                             ch.bright)))

    # -- weight, breath, grit ---------------------------------------------
    if ch.body > 0.5:
        layers.append(Layer(
            kind="sub", f0=f0 * 0.35, f1=f0 * 0.2, curve="snap",
            dur=min(total, _lerp(0.05, 0.3, ch.body)),
            gain=_lerp(0.08, 0.5, ch.body), attack=0.0015,
            decay=_lerp(12.0, 5.0, ch.ring), filt="lp",
            cut0=_lerp(300, 800, ch.bright), cut1=_lerp(160, 380, ch.bright)))

    if ch.air > 0.42 and exc != "blow":
        layers.append(Layer(
            kind="noise", noise_kind="pink", dur=total * 0.95,
            gain=_lerp(0.0, 0.35, ch.air), attack=total * 0.25,
            decay=_lerp(6.0, 1.8, ch.ring), filt="bp",
            cut0=_lerp(1800, 5000, ch.bright),
            cut1=_lerp(3200, 9000, ch.bright), res=0.8))

    if ch.grain > 0.3 and exc != "particles":
        layers.append(Layer(
            kind="grain", dur=total * 0.9, gain=float(np.clip(ch.grain * 0.7,
                                                              0.08, 0.9)),
            f0=_log(600, 4000, ch.bright), f1=_log(2500, 13000, ch.bright),
            density=_log(45.0, 2000.0, ch.grain),
            grain_ms=_lerp(24.0, 5.0, ch.bright), attack=0.001,
            decay=_lerp(9.0, 1.6, ch.ring), filt="none"))

    if not layers:
        layers.append(voice(0.0, 1.0, 1.0, total, "strike", hardness))

    # -- the room and the wires -------------------------------------------
    r = Recipe(name=name, prompt=prompt, seed=int(seed), layers=layers,
               reverse=bool(reverse))
    r.drive = float(np.clip(ch.grit * 0.7, 0.0, 0.8))
    if ch.digital > 0.45:
        r.bits = int(round(_lerp(11, 4, ch.digital)))
        r.crush_rate = int(_log(22050, 6000, ch.digital))
    r.highpass = _lerp(20.0, 260.0, max(0.0, 0.7 - ch.body) * (0.4 + ch.bright))
    r.lowpass = 0.0 if ch.bright > 0.5 else _log(11000, 2200, 1.0 - ch.bright * 2)
    r.rev_mix = float(np.clip(ch.space * 0.6 * R["space"], 0.0, 0.6))
    r.rev_size = float(np.clip((0.1 + ch.space * 0.7) * R["room"], 0.05, 0.9))
    r.rev_damp = float(np.clip(1.0 - ch.bright * 0.7, 0.15, 0.95))
    if r.rev_mix < 0.02:
        r.rev_mix = 0.0
    if ch.motion > 0.6 and role in ("toast", "ok", "warn", "error"):
        r.echo_time = _lerp(0.05, 0.2, ch.ring)
        r.echo_fb = 0.3
        r.echo_mix = _lerp(0.0, 0.24, ch.motion - 0.6)
    r.width = float(np.clip(ch.space * 0.4 + ch.air * 0.3, 0.0, 0.7))
    r.peak_db = _lerp(R["level"][0], R["level"][1],
                      0.35 + 0.3 * ch.body + 0.35 * (1.0 - ch.space))
    return r
