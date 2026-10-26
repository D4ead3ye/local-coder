"""What things are made of, as modal data.

A material is a set of partial ratios plus - and this is the part that
actually identifies it - a rule for how much faster the high partials die
than the low ones. Two objects with identical partial frequencies sound
like different substances entirely if their damping slopes differ, and that
single number is most of what a listener uses to say "wood" or "glass".

    ratios   partial frequencies, as multiples of the fundamental
    tau      seconds for the fundamental to fall to 1/e
    slope    how much faster mode k decays:  tau_k = tau * ratio_k ** -slope
    tilt     how much quieter mode k starts: amp_k = ratio_k ** -tilt
    f_ref    a fundamental this material naturally sits at
    noise    broadband contact rattle mixed in with the ring
    exciter  what is normally done to it

Wood has a steep slope (1.3): its brightness collapses in milliseconds and
you hear a knock. Steel has a shallow one (0.35): the highs outlive
everything and you hear a ring. That contrast is free here and was
impossible in a filtered oscillator.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    ratios: tuple
    tau: float
    slope: float
    tilt: float
    f_ref: float
    noise: float = 0.15
    exciter: str = "strike"
    hardness: float = 0.5


def _harmonic(k):
    return tuple(float(i) for i in range(1, k + 1))


def _odd(k):
    return tuple(float(i) for i in range(1, 2 * k, 2))


MATERIALS = {
    # -- vitreous: few modes, shallow damping, they ring ------------------
    "glass": Material("glass", (1, 2.71, 5.15, 8.43, 12.4, 17.1),
                      1.5, 0.55, 0.95, 1400, 0.10),
    "crystal": Material("crystal", (1, 2.68, 5.08, 8.31, 12.2, 16.8, 22.0),
                        2.6, 0.42, 0.85, 1900, 0.06),
    "ice": Material("ice", (1, 2.6, 4.9, 7.9, 11.4),
                    0.65, 0.85, 1.1, 1700, 0.28),
    "ceramic": Material("ceramic", (1, 2.66, 4.98, 8.1, 11.9),
                        0.55, 0.8, 1.0, 1150, 0.16),
    "porcelain": Material("porcelain", (1, 2.7, 5.06, 8.2, 12.1, 16.5),
                          0.8, 0.62, 0.92, 1550, 0.12),

    # -- metals: the shallowest slopes there are --------------------------
    "metal": Material("metal", (1, 2.756, 5.404, 8.933, 13.34, 18.64, 24.8),
                      3.0, 0.35, 0.85, 780, 0.10),
    "steel": Material("steel", (1, 2.76, 5.4, 8.93, 13.3, 18.6, 24.8, 31.8),
                      4.2, 0.28, 0.8, 620, 0.08),
    "tin": Material("tin", (1, 2.8, 5.5, 9.0, 13.5),
                    0.9, 0.7, 1.05, 1800, 0.22),
    "bell": Material("bell", (0.5, 1.0, 1.183, 1.506, 2.0, 2.514, 2.662,
                              3.011, 4.166), 3.2, 0.45, 0.9, 520, 0.06),
    "gong": Material("gong", (1, 1.42, 1.87, 2.34, 2.9, 3.6, 4.5, 5.7, 7.1),
                     4.0, 0.5, 0.95, 180, 0.12),
    "plate": Material("plate", (1, 1.72, 2.41, 3.11, 4.06, 5.3, 6.9, 8.8),
                      2.2, 0.55, 1.0, 420, 0.14),

    # -- woods: steep slopes, the highs are gone at once ------------------
    "wood": Material("wood", (1, 3.98, 10.4, 19.6),
                     0.30, 1.35, 1.3, 440, 0.22),
    "woodblock": Material("woodblock", (1, 2.53, 4.32, 6.6),
                          0.085, 1.6, 1.4, 900, 0.32),
    "bamboo": Material("bamboo", (1, 2.9, 5.6, 9.1),
                       0.16, 1.25, 1.2, 780, 0.26),

    # -- soft and dead ----------------------------------------------------
    "plastic": Material("plastic", (1, 2.43, 4.16, 6.3),
                        0.048, 1.7, 1.5, 1300, 0.34),
    "rubber": Material("rubber", (1, 2.2, 3.6),
                       0.022, 2.0, 1.9, 320, 0.4),
    "felt": Material("felt", (1, 2.1, 3.3),
                     0.014, 2.3, 2.2, 260, 0.5, hardness=0.15),
    "cardboard": Material("cardboard", (1, 2.2, 3.6, 5.2),
                          0.03, 2.0, 1.8, 820, 0.62),
    "paper": Material("paper", (1, 2.1, 3.5),
                      0.011, 2.4, 2.1, 2600, 0.85, exciter="scrape"),
    "cloth": Material("cloth", (1, 2.0, 3.2),
                      0.02, 2.4, 2.3, 900, 0.9, exciter="scrape"),

    # -- dense and mineral -------------------------------------------------
    "stone": Material("stone", (1, 2.31, 3.94, 5.8),
                      0.13, 1.2, 1.2, 700, 0.3),
    "marble": Material("marble", (1, 2.4, 4.1, 6.2, 8.6),
                       0.28, 0.95, 1.05, 950, 0.2),
    "bone": Material("bone", (1, 2.6, 4.7, 7.2),
                     0.2, 1.1, 1.15, 1100, 0.25),

    # -- driven and hollow -------------------------------------------------
    "membrane": Material("membrane", (1, 1.593, 2.135, 2.295, 2.653, 2.917,
                                      3.155), 0.22, 1.1, 1.0, 170, 0.3),
    "string": Material("string", _harmonic(14), 1.8, 0.7, 1.0, 330, 0.08,
                       exciter="pluck"),
    "tube": Material("tube", _odd(7), 0.55, 0.8, 1.1, 400, 0.2,
                     exciter="blow"),
    "pipe": Material("pipe", _odd(6), 0.9, 0.65, 1.0, 300, 0.18,
                     exciter="blow"),
    "shell": Material("shell", (1, 2.05, 3.3, 4.8, 6.4),
                      0.5, 0.9, 1.05, 900, 0.35, exciter="blow"),
}

# Materials that are really a process, handled by their exciter rather than
# by a resonator: the ring is incidental, the texture is the sound.
PROCESS = {"water", "sand", "gravel", "fire", "wind", "static"}


def blend(a: Material, b: Material, t: float) -> Material:
    """Interpolate two materials. "Glassy wood" is a real request and the
    honest answer is a resonator part way between the two."""
    t = float(np.clip(t, 0.0, 1.0))
    k = min(len(a.ratios), len(b.ratios))
    ratios = tuple(a.ratios[i] * (1 - t) + b.ratios[i] * t for i in range(k))
    return Material(
        name="%s/%s" % (a.name, b.name), ratios=ratios,
        tau=float(np.exp(np.log(a.tau) * (1 - t) + np.log(b.tau) * t)),
        slope=a.slope * (1 - t) + b.slope * t,
        tilt=a.tilt * (1 - t) + b.tilt * t,
        f_ref=float(np.exp(np.log(a.f_ref) * (1 - t) + np.log(b.f_ref) * t)),
        noise=a.noise * (1 - t) + b.noise * t,
        exciter=a.exciter if t < 0.5 else b.exciter,
        hardness=a.hardness * (1 - t) + b.hardness * t)


def modes(mat: Material, f0=None, hardness=0.5, count=0, damp=1.0,
          spread=0.0, rng=None):
    """Material -> (freqs, amps, taus) ready for engine.modal.

    `hardness` is the contact, not a filter: a hard strike puts energy into
    the high modes, a soft one never wakes them. `spread` detunes the
    partials slightly, which is what stops a struck object from sounding
    like an additive synth patch - no two real objects have exactly the
    textbook ratios.
    """
    f0 = float(f0 or mat.f_ref)
    ratios = np.asarray(mat.ratios, dtype=np.float64)
    if count:
        ratios = ratios[:max(1, int(count))]
    if spread > 0 and rng is not None:
        ratios = ratios * (1.0 + rng.uniform(-spread, spread, len(ratios)))

    freqs = f0 * ratios
    # `tilt` above is a relative index, not an exponent to use raw. Applied
    # directly it puts mode 2 more than 20dB down and mode 4 near 50dB, so
    # every material collapses onto its fundamental - which is a decaying
    # sine, which is a beep. Real struck objects sit far flatter than that.
    #
    # Hardness is deliberately almost absent here: contact time in
    # engine.strike already decides which modes receive energy, and
    # spending it twice made soft strikes vanish rather than soften.
    tilt = mat.tilt * 0.45 + 0.3 * (1.0 - float(np.clip(hardness, 0.0, 1.0)))
    amps = ratios ** (-tilt)
    taus = mat.tau * float(damp) * ratios ** (-mat.slope)
    return freqs, amps / (amps.max() or 1.0), np.maximum(taus, 3e-4)


# ------------------------------------------------------------ word -> stuff
# Discrete lookup, not a blended feature. Naming a material should *select*
# it: the old design averaged "porcelain" toward a neutral default and
# landed it next to "camera shutter", which is exactly the complaint.

MATERIAL_WORDS = {
    "glass": "glass", "glassy": "glass", "tumbler": "glass",
    "window": "glass", "shard": "glass", "crystal": "crystal",
    "crystalline": "crystal", "diamond": "crystal", "ice": "ice",
    "icy": "ice", "frozen": "ice", "frost": "ice",
    "ceramic": "ceramic", "pottery": "ceramic", "clay": "ceramic",
    "tile": "ceramic", "porcelain": "porcelain", "china": "porcelain",
    "teacup": "porcelain", "cup": "porcelain", "mug": "porcelain",
    "plate-dish": "porcelain", "saucer": "porcelain", "tea": "porcelain",

    "typewriter": "metal", "mechanism": "metal", "coin": "tin",
    "cymbal-crash": "plate", "keycaps": "plastic", "mouse": "plastic",
    "trackpad": "plastic", "phone": "glass", "screen": "glass",
    "tumbler-glass": "glass", "wineglass": "crystal", "chandelier": "crystal",
    "metal": "metal", "metallic": "metal", "aluminium": "metal",
    "aluminum": "metal", "chrome": "metal", "alloy": "metal",
    "steel": "steel", "iron": "steel", "anvil": "steel", "blade": "steel",
    "knife": "steel", "sword": "steel", "tin": "tin", "can": "tin",
    "foil": "tin", "bell": "bell", "chime": "bell", "carillon": "bell",
    "gong": "gong", "cymbal": "plate", "plate": "plate", "sheet": "plate",
    "brass": "metal", "copper": "metal", "bronze": "bell",

    "wood": "wood", "wooden": "wood", "oak": "wood", "timber": "wood",
    "marimba": "wood", "xylophone": "wood", "log": "wood",
    "woodblock": "woodblock", "knock": "woodblock", "clave": "woodblock",
    "block": "woodblock", "door": "woodblock", "desk": "woodblock",
    "table": "woodblock", "bamboo": "bamboo",

    "plastic": "plastic", "acrylic": "plastic", "keyboard": "plastic",
    "button": "plastic", "switch": "plastic", "keycap": "plastic",
    "toy": "plastic", "lego": "plastic", "rubber": "rubber",
    "silicone": "rubber", "eraser": "rubber", "felt": "felt",
    "foam": "felt", "velvet": "felt", "cushion": "felt", "pillow": "felt",

    "paper": "paper", "page": "paper", "envelope": "paper", "note": "paper",
    "letter": "paper", "newspaper": "paper", "cardboard": "cardboard",
    "box": "cardboard", "carton": "cardboard", "cloth": "cloth",
    "fabric": "cloth", "silk": "cloth", "curtain": "cloth",
    "leather": "cloth", "denim": "cloth",

    "stone": "stone", "rock": "stone", "pebble": "stone",
    "concrete": "stone", "brick": "stone", "marble": "marble",
    "granite": "marble", "bone": "bone", "skull": "bone", "tooth": "bone",

    "drum": "membrane", "snare": "membrane", "tom": "membrane",
    "kick": "membrane", "membrane": "membrane", "balloon": "membrane",
    "string": "string", "guitar": "string", "harp": "string",
    "piano": "string", "violin": "string", "banjo": "string",
    "kalimba": "metal", "wire": "string",
    "tube": "tube", "pipe": "pipe", "organ": "pipe", "flute": "tube",
    "whistle": "tube", "straw": "tube", "shell": "shell",
    "bottle": "shell", "jar": "shell", "vase": "shell",
}

# What is being *done*. Independent of the material, which is the point:
# tapping a glass and scraping a glass are obviously the same glass.
EXCITER_WORDS = {
    "strike": "strike", "hit": "strike", "tap": "strike", "knock": "strike",
    "click": "strike", "press": "strike", "punch": "strike",
    "impact": "strike", "smack": "strike", "clack": "strike",
    "slam": "strike", "bang": "strike", "drop": "strike", "ping": "strike",
    "tick": "strike", "type": "strike", "typing": "strike",
    "keystroke": "strike", "stamp": "strike", "snap": "strike",

    "pluck": "pluck", "plucked": "pluck", "strum": "pluck",
    "twang": "pluck", "boing": "pluck", "spring": "pluck",

    "scrape": "scrape", "scratch": "scrape", "rub": "scrape",
    "brush": "scrape", "slide": "scrape", "swipe": "scrape",
    "drag": "scrape", "sweep": "scrape", "rustle": "scrape",
    "shuffle": "scrape", "crumple": "scrape", "tear": "scrape",
    "rip": "scrape", "zip": "scrape", "zipper": "scrape",
    "turn": "scrape", "flip": "scrape", "swoosh": "scrape",
    "whoosh": "scrape", "whisper": "scrape",

    "blow": "blow", "breath": "blow", "wind": "blow", "breeze": "blow",
    "air": "blow", "gust": "blow", "hiss": "blow", "steam": "blow",

    "rain": "particles", "gravel": "particles", "sand": "particles",
    "crackle": "particles", "fire": "particles", "sizzle": "particles",
    "applause": "particles", "footsteps": "particles", "hail": "particles",
    "static": "particles", "popcorn": "particles", "confetti": "particles",
    "coins": "particles", "shatter": "particles", "crumble": "particles",

    "drip": "bubble", "droplet": "bubble", "water": "bubble",
    "bubble": "bubble", "splash": "bubble", "plop": "bubble",
    "liquid": "bubble", "pour": "bubble", "boil": "bubble",
}

EXCITERS = ("strike", "pluck", "scrape", "blow", "particles", "bubble")


def get(name):
    return MATERIALS.get(name)


def default_for(exciter):
    """A sensible body for an excitation nobody gave a material for."""
    return MATERIALS[{"pluck": "string", "scrape": "paper", "blow": "tube",
                      "particles": "woodblock",
                      "bubble": "shell"}.get(exciter, "plastic")]
