"""What a prompt *sounds like*, as numbers.

The old parser matched a prompt to one of N hand-written recipes. That
cannot cover language: "thunderstorm" missed while "thunder" hit, and every
new phrasing needed a new entry. So nouns no longer select a recipe. Every
word contributes to a **character** - sixteen acoustic dimensions - and the
recipe is built from that character afterwards. Unknown words cost coverage
of one word rather than the whole prompt, and any combination of known
words produces a sound no preset had to anticipate.

Four passes turn a word into features, cheapest first:

    exact  ->  morphology ("breezy" -> breeze, "clicks" -> click)
           ->  compound split ("thunderstorm" -> thunder + storm)
           ->  fuzzy match ("cristal" -> crystal), stdlib difflib

Anything still unmatched is reported rather than ignored, because a silent
fallback to a beep is how you end up not knowing the tool missed you.
"""

import difflib
import re
from dataclasses import dataclass, fields, replace

from . import material as _material

# Sixteen features. All run 0..1 except `rise`, which is signed.
#
#   noise   tonal ......... noisy          bright  dark ........ bright
#   pitch   low ........... high           metal   pure ........ inharmonic
#   wood    ............... plucked        body    thin ........ thick
#   grit    clean ......... distorted      snap    soft ........ percussive
#   ring    dead .......... sustaining     space   dry ......... reverberant
#   air     ............... breathy        digital organic ..... bitcrushed
#   motion  static ........ gliding        wet     ............. liquid
#   warm    neutral ....... warm           rise    falling ..... rising


@dataclass
class Character:
    noise: float = 0.35
    bright: float = 0.5
    pitch: float = 0.5
    metal: float = 0.2
    wood: float = 0.2
    body: float = 0.35
    grit: float = 0.15
    snap: float = 0.6
    ring: float = 0.3
    space: float = 0.2
    air: float = 0.15
    digital: float = 0.2
    motion: float = 0.25
    wet: float = 0.1
    warm: float = 0.5
    grain: float = 0.0
    rise: float = 0.0

    def blend(self, other, t):
        return Character(**{f.name: getattr(self, f.name) * (1 - t) +
                            getattr(other, f.name) * t for f in fields(self)})

    def nudged(self, **kw):
        return replace(self, **kw)


NEUTRAL = Character()
KEYS = tuple(f.name for f in fields(Character))


def W(**kw):
    """One lexicon entry: the features this word asserts, nothing else.

    A word only speaks to what it knows about. "glass" says nothing about
    reverb, so a glassy sound stays as dry as the rest of the prompt asks.
    """
    bad = set(kw) - set(KEYS) - {"role", "gesture", "w"}
    if bad:
        raise ValueError("unknown feature(s): %s" % ", ".join(sorted(bad)))
    return kw


# --------------------------------------------------------------- the lexicon
# Grouped by how people actually describe a sound they want. Weights (`w`)
# above 1 mark words that should dominate a phrase - a material usually
# says more about the result than a mood does.

LEXICON = {
    # -- materials ---------------------------------------------------------
    "glass": W(bright=.92, metal=.8, ring=.65, snap=.85, pitch=.8, body=.15, w=1.3),
    "crystal": W(bright=.95, metal=.85, ring=.8, pitch=.85, space=.45, body=.12, w=1.3),
    "ice": W(bright=.88, metal=.6, ring=.5, pitch=.8, air=.3, body=.15),
    "porcelain": W(bright=.85, metal=.7, ring=.55, snap=.9, pitch=.72, body=.2, w=1.2),
    "ceramic": W(bright=.8, metal=.65, ring=.5, snap=.9, pitch=.68, body=.25),
    "china": W(bright=.85, metal=.7, ring=.55, snap=.88, pitch=.72),
    "metal": W(metal=.9, bright=.75, ring=.7, grit=.3, w=1.3),
    "steel": W(metal=.92, bright=.8, ring=.72, grit=.3, w=1.2),
    "iron": W(metal=.85, bright=.55, ring=.6, body=.6, grit=.35),
    "brass": W(metal=.8, bright=.7, ring=.7, warm=.7),
    "copper": W(metal=.78, bright=.65, ring=.65, warm=.7),
    "tin": W(metal=.85, bright=.85, ring=.4, body=.1, pitch=.8),
    "aluminium": W(metal=.85, bright=.8, ring=.5, body=.2),
    "aluminum": W(metal=.85, bright=.8, ring=.5, body=.2),
    "chrome": W(metal=.9, bright=.88, ring=.6),
    "wood": W(wood=.9, bright=.4, ring=.25, warm=.75, metal=.05, w=1.3),
    "oak": W(wood=.9, bright=.35, body=.5, warm=.8),
    "bamboo": W(wood=.85, bright=.6, ring=.35, body=.2),
    "plastic": W(bright=.6, metal=.25, ring=.15, snap=.9, body=.2, w=1.1),
    "acrylic": W(bright=.7, metal=.35, ring=.3, snap=.85),
    "rubber": W(bright=.25, ring=.1, snap=.7, body=.45, warm=.7),
    "silicone": W(bright=.3, ring=.1, snap=.65, body=.4),
    "paper": W(noise=.85, bright=.7, ring=.08, body=.08, air=.4, w=1.2),
    "cardboard": W(noise=.7, bright=.45, ring=.12, body=.35),
    "card": W(noise=.7, bright=.6, ring=.1, body=.15),
    "cloth": W(noise=.8, bright=.35, ring=.08, air=.45, warm=.65),
    "fabric": W(noise=.8, bright=.4, ring=.08, air=.45),
    "silk": W(noise=.7, bright=.55, ring=.1, air=.6, snap=.35),
    "velvet": W(noise=.7, bright=.25, ring=.1, air=.5, snap=.25, warm=.8),
    "felt": W(noise=.6, bright=.2, ring=.08, snap=.3, warm=.8),
    "foam": W(noise=.55, bright=.25, ring=.05, snap=.4, body=.3),
    "stone": W(bright=.4, metal=.3, ring=.2, body=.7, snap=.95, w=1.1),
    "marble": W(bright=.6, metal=.45, ring=.35, body=.55, snap=.95),
    "concrete": W(bright=.35, ring=.12, body=.75, snap=.95, grit=.25),
    "sand": W(noise=.95, bright=.6, ring=.1, air=.3, snap=.3),
    "gravel": W(noise=.9, bright=.45, ring=.12, grit=.4, body=.4),
    "leather": W(noise=.6, bright=.35, ring=.1, warm=.75, snap=.6),
    "wax": W(bright=.3, ring=.1, snap=.5, body=.35, warm=.7),
    "bone": W(bright=.6, metal=.35, ring=.3, body=.4, snap=.95),
    "clay": W(bright=.35, ring=.15, body=.5, snap=.8, warm=.7),

    # -- nature and weather ------------------------------------------------
    "breeze": W(noise=.95, bright=.55, air=.9, snap=.1, ring=.6, motion=.7,
                space=.4, body=.2, w=1.3),
    "wind": W(noise=.95, bright=.5, air=.9, snap=.1, ring=.65, motion=.75, w=1.3),
    "gust": W(noise=.95, bright=.6, air=.85, snap=.35, motion=.85, ring=.4),
    "gale": W(noise=.95, bright=.5, air=.9, motion=.9, ring=.7, grit=.2),
    "storm": W(noise=.9, bright=.35, air=.7, body=.7, motion=.6, ring=.7, grit=.3),
    "thunder": W(noise=.8, bright=.15, body=.95, grit=.4, ring=.7, space=.6,
                 pitch=.1, w=1.3),
    "lightning": W(noise=.7, bright=.95, snap=.98, grit=.45, ring=.3),
    "rain": W(noise=.95, bright=.65, air=.5, snap=.3, ring=.5, wet=.8, w=1.2),
    "drizzle": W(noise=.95, bright=.7, air=.4, snap=.25, wet=.7, body=.1),
    "water": W(noise=.6, bright=.6, wet=.95, motion=.5, ring=.35, w=1.3),
    "droplet": W(noise=.15, bright=.75, wet=.95, snap=.85, ring=.3, rise=.5),
    "drip": W(noise=.15, bright=.75, wet=.95, snap=.85, ring=.3, rise=.5),
    "splash": W(noise=.85, bright=.6, wet=.95, snap=.7, ring=.3),
    "bubble": W(noise=.3, bright=.65, wet=.9, snap=.6, ring=.25, rise=.7, w=1.1),
    "stream": W(noise=.9, bright=.6, wet=.85, ring=.6, motion=.4, air=.4),
    "river": W(noise=.9, bright=.5, wet=.85, ring=.65, body=.4),
    "ocean": W(noise=.9, bright=.4, wet=.8, air=.7, ring=.75, body=.6,
               motion=.55, space=.5),
    "wave": W(noise=.85, bright=.45, wet=.8, air=.6, ring=.6, motion=.7),
    "sea": W(noise=.9, bright=.42, wet=.8, air=.7, ring=.7, body=.55),
    "forest": W(noise=.75, bright=.55, air=.6, wood=.4, space=.55, ring=.5),
    "leaves": W(noise=.95, bright=.7, air=.6, snap=.25, ring=.25, body=.08),
    "leaf": W(noise=.95, bright=.7, air=.6, snap=.25, ring=.2, body=.08),
    "grass": W(noise=.92, bright=.65, air=.5, snap=.2, ring=.2),
    "outdoor": W(space=.55, air=.5, noise=.5),
    "outside": W(space=.55, air=.5),
    "meadow": W(noise=.8, bright=.6, air=.6, space=.5),
    "cave": W(space=.9, bright=.25, ring=.8, body=.6, warm=.4),
    "mountain": W(space=.8, body=.7, ring=.6, bright=.35),
    "desert": W(noise=.85, bright=.55, air=.7, space=.6, ring=.4),
    "snow": W(noise=.8, bright=.45, air=.6, snap=.3, ring=.15, warm=.4),
    "fire": W(noise=.9, bright=.5, grit=.35, ring=.5, warm=.85, motion=.4),
    "ember": W(noise=.85, bright=.55, grit=.3, warm=.85, ring=.3),
    "sizzle": W(noise=.95, bright=.85, grit=.3, ring=.4, snap=.4),
    "crackle": W(noise=.9, bright=.75, grit=.45, snap=.8, ring=.2),
    "earth": W(body=.8, bright=.25, ring=.3, warm=.7),
    "sky": W(space=.7, bright=.7, air=.7, pitch=.7),
    "cloud": W(noise=.6, air=.85, snap=.15, space=.7, bright=.55),
    "sun": W(bright=.85, warm=.9, ring=.5, space=.4),
    "moon": W(bright=.4, space=.7, ring=.6, pitch=.35, air=.4),
    "star": W(bright=.95, pitch=.9, ring=.6, space=.6, metal=.5),
    "night": W(bright=.25, space=.6, warm=.45, pitch=.3),
    "morning": W(bright=.75, warm=.7, space=.4, pitch=.65),

    # -- creatures ---------------------------------------------------------
    "bird": W(noise=.2, bright=.9, pitch=.92, motion=.7, ring=.25, rise=.5),
    "chirp": W(noise=.15, bright=.85, pitch=.9, motion=.6, ring=.2, rise=.6),
    "cricket": W(noise=.5, bright=.9, pitch=.9, snap=.8, ring=.15),
    "insect": W(noise=.5, bright=.85, pitch=.85, snap=.7, ring=.2),
    "bee": W(noise=.4, bright=.4, pitch=.4, grit=.4, ring=.7, motion=.3),
    "cat": W(noise=.4, bright=.6, pitch=.7, warm=.7, ring=.5),
    "purr": W(noise=.6, bright=.2, pitch=.15, body=.6, warm=.85, ring=.7),
    "frog": W(noise=.45, bright=.35, pitch=.3, body=.5, grit=.3),
    "owl": W(noise=.35, bright=.35, pitch=.45, air=.6, ring=.5, space=.5),
    "whale": W(noise=.25, bright=.2, pitch=.12, body=.85, ring=.9, space=.85),
    "heartbeat": W(noise=.35, bright=.12, pitch=.06, body=.95, snap=.7, ring=.2),

    # -- instruments and music ---------------------------------------------
    "bell": W(metal=.9, bright=.75, ring=.9, snap=.85, space=.45, noise=.05, w=1.4),
    "chime": W(metal=.85, bright=.85, ring=.85, pitch=.8, space=.5, noise=.05, w=1.3),
    "gong": W(metal=.95, bright=.4, ring=.95, body=.8, space=.7, pitch=.15),
    "cymbal": W(metal=.85, noise=.6, bright=.9, ring=.8, space=.4),
    "triangle": W(metal=.9, bright=.95, ring=.85, pitch=.92, body=.08),
    "marimba": W(wood=.8, bright=.55, ring=.5, warm=.75, noise=.05, w=1.2),
    "xylophone": W(wood=.6, bright=.8, ring=.4, pitch=.8, metal=.25),
    "kalimba": W(metal=.45, wood=.5, bright=.7, ring=.6, pitch=.7),
    "harp": W(wood=.55, bright=.7, ring=.75, space=.45, noise=.05, motion=.3),
    "piano": W(wood=.5, bright=.6, ring=.7, body=.4, noise=.08),
    "guitar": W(wood=.75, bright=.55, ring=.6, warm=.7),
    "string": W(wood=.6, bright=.6, ring=.7, motion=.2),
    "flute": W(noise=.25, air=.75, bright=.7, ring=.5, pitch=.75, snap=.2),
    "whistle": W(noise=.15, air=.5, bright=.85, pitch=.9, ring=.35, motion=.4),
    "organ": W(noise=.05, bright=.5, ring=.8, body=.6, space=.6, snap=.2),
    "choir": W(noise=.2, air=.6, bright=.5, ring=.85, space=.8, snap=.15),
    "voice": W(noise=.3, air=.5, bright=.5, ring=.5, warm=.7),
    "drum": W(noise=.55, bright=.3, body=.85, snap=.95, ring=.2, w=1.2),
    "snare": W(noise=.85, bright=.6, snap=.98, ring=.15, body=.4),
    "kick": W(noise=.25, bright=.1, pitch=.04, body=.98, snap=.9, ring=.15),
    "hihat": W(noise=.9, bright=.92, snap=.95, ring=.1, metal=.5),
    "woodblock": W(wood=.95, bright=.6, snap=.95, ring=.15, body=.2),
    "clave": W(wood=.9, bright=.7, snap=.95, ring=.15),
    "pluck": W(wood=.7, snap=.85, ring=.4, noise=.1),
    "strum": W(wood=.7, ring=.6, motion=.35, noise=.15),
    "note": W(noise=.05, ring=.4),
    "chord": W(noise=.05, ring=.6, body=.5, space=.35),
    "melody": W(noise=.05, ring=.5, motion=.3),
    "arpeggio": W(noise=.08, ring=.4, motion=.4, gesture="arp"),
    "synth": W(noise=.1, bright=.65, digital=.5, ring=.5, w=1.1),
    "pad": W(noise=.15, snap=.1, ring=.9, space=.75, bright=.45),
    "bass": W(pitch=.06, body=.95, bright=.2, warm=.75),

    # -- technology --------------------------------------------------------
    "digital": W(digital=.85, bright=.65, noise=.1, w=1.2),
    "analog": W(digital=.05, warm=.8, grit=.2),
    "analogue": W(digital=.05, warm=.8, grit=.2),
    "robot": W(digital=.75, metal=.5, bright=.6, motion=.5, grit=.3),
    "robotic": W(digital=.75, metal=.5, bright=.6, motion=.5),
    "computer": W(digital=.7, bright=.7, noise=.15, snap=.8),
    "circuit": W(digital=.8, bright=.8, snap=.9, ring=.15),
    "laser": W(noise=.1, bright=.85, motion=.95, rise=-.8, grit=.35, pitch=.8),
    "beam": W(noise=.15, bright=.75, motion=.8, ring=.5, space=.4),
    "plasma": W(noise=.55, bright=.75, grit=.45, motion=.7, space=.4),
    "hologram": W(noise=.25, bright=.75, digital=.4, motion=.6, space=.7, ring=.6),
    "radar": W(noise=.1, bright=.6, ring=.55, space=.8, motion=.3),
    "sonar": W(noise=.08, bright=.5, ring=.75, space=.9, pitch=.55),
    "radio": W(noise=.7, bright=.55, grit=.35, digital=.3, w=1.1),
    "static": W(noise=1.0, bright=.6, grit=.4, ring=.35, snap=.2, w=1.2),
    "signal": W(noise=.3, bright=.7, digital=.4, ring=.4),
    "modem": W(noise=.5, bright=.7, digital=.8, grit=.4, motion=.5),
    "engine": W(noise=.7, bright=.3, body=.85, grit=.5, ring=.6, pitch=.15),
    "motor": W(noise=.6, bright=.35, body=.7, grit=.45, ring=.6),
    "server": W(noise=.8, bright=.35, body=.5, ring=.7, grit=.2),
    "fan": W(noise=.85, bright=.4, ring=.7, air=.6, motion=.2),
    "hum": W(noise=.25, bright=.25, ring=.85, body=.6, snap=.1),
    "buzz": W(noise=.45, bright=.5, grit=.6, ring=.6, snap=.3),
    "beep": W(noise=.05, bright=.7, digital=.55, ring=.25, snap=.7),
    "boop": W(noise=.05, bright=.45, digital=.5, ring=.3, rise=-.3, warm=.7),
    "blip": W(noise=.05, bright=.7, digital=.5, ring=.15, snap=.8),
    "glitch": W(noise=.7, digital=.9, grit=.7, bright=.7, snap=.9, ring=.15, w=1.2),
    "error": W(bright=.3, grit=.45, pitch=.2, rise=-.6, body=.6, role="error"),
    "alarm": W(bright=.7, grit=.3, ring=.4, snap=.7, digital=.4),
    "siren": W(bright=.65, motion=.9, ring=.7, grit=.25),
    "morse": W(noise=.05, bright=.75, digital=.6, snap=.9, ring=.1, gesture="double"),
    "vinyl": W(noise=.75, bright=.4, grit=.4, digital=.15, warm=.75),
    "tape": W(noise=.65, bright=.35, grit=.35, warm=.8, digital=.1),
    "cassette": W(noise=.7, bright=.35, grit=.35, warm=.8),
    "arcade": W(digital=.9, bright=.75, noise=.1, snap=.85, w=1.2),
    "chiptune": W(digital=.95, bright=.7, noise=.05, snap=.85, w=1.2),
    "8bit": W(digital=1.0, bright=.65, noise=.05, snap=.85, w=1.3),
    "16bit": W(digital=.75, bright=.7, noise=.08, snap=.8, w=1.2),
    "nes": W(digital=1.0, bright=.7, noise=.05, snap=.85),
    "gameboy": W(digital=1.0, bright=.6, noise=.08, snap=.85),
    "pixel": W(digital=.9, bright=.7, snap=.85),
    "retro": W(digital=.7, bright=.55, warm=.7, w=1.1),
    "vintage": W(digital=.15, warm=.85, bright=.35, noise=.4, grit=.3),
    "modern": W(digital=.3, bright=.65, grit=.05, space=.3),
    "futuristic": W(digital=.45, bright=.75, space=.6, motion=.55, ring=.5, w=1.1),
    "cyberpunk": W(digital=.6, bright=.6, grit=.55, space=.6, motion=.5, w=1.1),
    "neon": W(bright=.85, digital=.5, ring=.5, space=.5),
    "space": W(space=.9, ring=.7, bright=.5, air=.4, w=1.1),
    "alien": W(metal=.5, motion=.7, space=.7, digital=.4, bright=.6),
    "spaceship": W(noise=.5, body=.7, ring=.8, space=.85, motion=.4, bright=.4),

    # -- places and objects ------------------------------------------------
    "library": W(space=.6, bright=.4, noise=.4, snap=.4, warm=.65),
    "cathedral": W(space=1.0, ring=.9, bright=.45, body=.5),
    "church": W(space=.9, ring=.85, bright=.5),
    "hall": W(space=.8, ring=.75),
    "room": W(space=.45, ring=.4),
    "studio": W(space=.15, bright=.6, grit=.02),
    "office": W(space=.3, bright=.55, noise=.35),
    "kitchen": W(space=.35, bright=.7, metal=.4),
    "teacup": W(bright=.85, metal=.7, ring=.6, snap=.9, pitch=.78, body=.15),
    "cup": W(bright=.75, metal=.55, ring=.45, snap=.85, pitch=.7),
    "bottle": W(bright=.7, metal=.5, ring=.5, body=.3, snap=.8),
    "coin": W(metal=.85, bright=.9, ring=.55, pitch=.85, snap=.9, body=.1, w=1.2),
    "key": W(metal=.8, bright=.8, ring=.4, snap=.9, body=.15),
    "lock": W(metal=.7, bright=.5, snap=.95, ring=.25, body=.45),
    "door": W(wood=.6, body=.65, snap=.8, ring=.3, space=.4),
    "hatch": W(metal=.75, body=.7, snap=.9, ring=.35, grit=.25),
    "switch": W(bright=.65, snap=.98, ring=.12, body=.25, w=1.1),
    "button": W(bright=.55, snap=.95, ring=.15, body=.3, w=1.1),
    "keyboard": W(bright=.6, snap=.95, ring=.12, body=.3, wood=.3),
    "typewriter": W(metal=.55, bright=.6, snap=.98, ring=.15, body=.4),
    "clock": W(wood=.4, metal=.4, bright=.6, snap=.95, ring=.2),
    "watch": W(metal=.55, bright=.75, snap=.95, ring=.15, body=.08),
    "camera": W(metal=.6, bright=.6, snap=.98, ring=.15, body=.3),
    "shutter": W(metal=.6, bright=.65, snap=.98, ring=.12),
    "page": W(noise=.9, bright=.65, air=.45, snap=.3, ring=.12, body=.08, w=1.1),
    "book": W(noise=.7, bright=.45, body=.35, snap=.6, ring=.15),
    "zipper": W(noise=.85, bright=.7, grit=.4, motion=.7, snap=.5),
    "match": W(noise=.9, bright=.7, grit=.4, snap=.7, motion=.4),
    "candle": W(noise=.7, bright=.4, warm=.85, ring=.4, snap=.2),
    "balloon": W(noise=.5, bright=.6, snap=.9, body=.35, ring=.2),
    "spring": W(metal=.75, bright=.65, ring=.6, motion=.75),
    "chain": W(metal=.85, bright=.75, ring=.5, snap=.85, noise=.35),
    "anvil": W(metal=.95, bright=.7, ring=.85, body=.7, snap=.95),

    # -- texture and quality ----------------------------------------------
    "soft": W(snap=.15, bright=.35, grit=.02, ring=.45, warm=.7, w=1.2),
    "gentle": W(snap=.15, bright=.4, grit=.02, warm=.7),
    "subtle": W(snap=.25, bright=.4, grit=.02),
    "quiet": W(snap=.25, bright=.4),
    "delicate": W(snap=.3, bright=.7, body=.12, ring=.5),
    "smooth": W(grit=.02, snap=.3, bright=.5, noise=.15),
    "silky": W(grit=.02, snap=.25, air=.5, bright=.6),
    "mellow": W(bright=.3, snap=.3, warm=.85, grit=.03),
    "warm": W(warm=.9, bright=.4, grit=.08, w=1.1),
    "cold": W(warm=.1, bright=.8, metal=.5),
    "cool": W(warm=.25, bright=.7),
    "harsh": W(grit=.7, bright=.85, snap=.9, w=1.1),
    "rough": W(grit=.7, noise=.6, bright=.6),
    "gritty": W(grit=.8, noise=.55),
    "dirty": W(grit=.7, noise=.5, digital=.25),
    "crunchy": W(grit=.75, noise=.65, snap=.9, bright=.7),
    "distorted": W(grit=.9, bright=.65, w=1.1),
    "aggressive": W(grit=.75, snap=.95, bright=.75, body=.55, w=1.1),
    "brutal": W(grit=.85, snap=.95, body=.8, bright=.6),
    "clean": W(grit=.0, noise=.1, digital=.05, w=1.1),
    "pure": W(grit=.0, noise=.05, digital=.0, metal=.05),
    "crisp": W(snap=.9, bright=.8, ring=.2, grit=.05, w=1.1),
    "punchy": W(snap=.95, body=.6, ring=.2, w=1.1),
    "snappy": W(snap=.95, ring=.15, bright=.7),
    "sharp": W(snap=.95, bright=.85, ring=.2),
    "tight": W(snap=.9, ring=.12, space=.05),
    "dull": W(bright=.2, snap=.4, ring=.3),
    "muffled": W(bright=.15, snap=.35, air=.25, space=.3),
    "muted": W(bright=.3, snap=.4, ring=.2),
    "dark": W(bright=.12, warm=.55, pitch=.3, w=1.1),
    "bright": W(bright=.92, pitch=.65, w=1.1),
    "brilliant": W(bright=.95, metal=.5, ring=.5),
    "shiny": W(bright=.9, metal=.6, ring=.5),
    "glossy": W(bright=.85, metal=.45, grit=.02),
    "deep": W(pitch=.1, body=.9, bright=.28, w=1.2),
    "low": W(pitch=.12, body=.75, bright=.3),
    "high": W(pitch=.9, bright=.8, body=.12),
    "tiny": W(pitch=.9, body=.06, ring=.2, snap=.85),
    "small": W(pitch=.78, body=.15, ring=.25),
    "big": W(pitch=.2, body=.85, ring=.6, space=.5),
    "huge": W(pitch=.1, body=.95, ring=.75, space=.7),
    "massive": W(pitch=.06, body=.98, ring=.8, space=.75),
    "thick": W(body=.85, warm=.7, noise=.3),
    "thin": W(body=.1, bright=.7),
    "hollow": W(body=.25, ring=.6, metal=.35, bright=.45),
    "resonant": W(ring=.85, metal=.4, body=.55),
    "ringing": W(ring=.9, metal=.6),
    "airy": W(air=.9, noise=.7, snap=.2, bright=.6, w=1.1),
    "breathy": W(air=.9, noise=.75, snap=.15),
    "windy": W(air=.9, noise=.9, motion=.6),
    "hissy": W(noise=.95, bright=.8, air=.5),
    "wet": W(space=.75, ring=.6, w=1.1),
    "dry": W(space=.02, ring=.25, w=1.1),
    "spacious": W(space=.85, ring=.7, air=.4),
    "roomy": W(space=.6, ring=.5),
    "distant": W(space=.8, bright=.3, snap=.2),
    "close": W(space=.05, bright=.6, snap=.8),
    "echoing": W(space=.9, ring=.8),
    "ambient": W(space=.85, snap=.15, ring=.8, air=.5),
    "liquid": W(wet=.9, motion=.5, bright=.6),
    "watery": W(wet=.9, noise=.5, bright=.6),
    "bouncy": W(snap=.85, ring=.3, rise=.4, gesture="double"),
    "elastic": W(motion=.7, snap=.7, ring=.3),
    "wobbly": W(motion=.85, ring=.4, grit=.2),
    "shimmering": W(metal=.7, bright=.9, ring=.8, space=.6, motion=.3),
    "sparkly": W(metal=.75, bright=.95, ring=.7, pitch=.85, space=.5),
    "sparkle": W(metal=.75, bright=.95, ring=.7, pitch=.85, space=.5),
    "magical": W(metal=.6, bright=.85, ring=.75, space=.7, motion=.4, rise=.5),
    "magic": W(metal=.6, bright=.85, ring=.75, space=.7, rise=.5),
    "dreamy": W(space=.85, snap=.15, ring=.85, bright=.5, air=.4),
    "ethereal": W(space=.9, air=.6, ring=.85, bright=.65, snap=.12),
    "organic": W(digital=.02, wood=.55, warm=.75, grit=.1, w=1.1),
    "natural": W(digital=.02, wood=.45, warm=.7, noise=.55, grit=.08, w=1.1),
    "mechanical": W(metal=.6, snap=.95, digital=.3, ring=.25, grit=.25),
    "industrial": W(metal=.7, grit=.55, body=.7, snap=.9),
    "lofi": W(digital=.55, grit=.35, bright=.3, noise=.45, warm=.75, w=1.1),
    "cozy": W(warm=.9, bright=.35, snap=.35, space=.35, grit=.05, wood=.4),
    "elegant": W(grit=.0, snap=.4, bright=.6, ring=.6, space=.4, metal=.3),
    "premium": W(grit=.0, snap=.5, bright=.55, ring=.5, space=.35, noise=.15, w=1.1),
    "luxury": W(grit=.0, snap=.45, bright=.55, ring=.6, space=.4),
    "minimal": W(grit=.0, noise=.1, ring=.25, space=.12, body=.25),
    "playful": W(bright=.75, rise=.5, snap=.8, ring=.3, digital=.35),
    "cute": W(pitch=.85, bright=.8, snap=.8, ring=.25, rise=.4, body=.1),
    "cartoon": W(pitch=.8, bright=.8, motion=.6, snap=.85, rise=.4),
    "happy": W(bright=.75, rise=.6, ring=.4),
    "sad": W(bright=.3, rise=-.6, ring=.6, warm=.6),
    "calm": W(snap=.2, bright=.45, ring=.5, grit=.02, space=.35),
    "urgent": W(snap=.95, bright=.8, grit=.25, ring=.2, gesture="double"),
    "serious": W(bright=.4, grit=.05, ring=.35, pitch=.35),
    "tense": W(grit=.3, bright=.6, ring=.5, rise=.3),
    "heavy": W(body=.9, pitch=.12, grit=.3, snap=.9),
    "light": W(body=.12, pitch=.75, bright=.75, snap=.6),
    "fast": W(snap=.9, ring=.15),
    "slow": W(snap=.25, ring=.7),
    "old": W(noise=.5, grit=.35, bright=.3, warm=.8, digital=.1),
    "new": W(grit=.03, bright=.7, digital=.3),
    "empty": W(space=.7, body=.2, ring=.6),
    "full": W(body=.8, ring=.5),

    # -- filling gaps found by testing real prompts ------------------------
    "machine": W(noise=.6, metal=.55, body=.7, grit=.45, ring=.5, snap=.85),
    "machinery": W(noise=.6, metal=.6, body=.75, grit=.5, snap=.85),
    "gear": W(metal=.7, snap=.9, ring=.3, grit=.3, body=.45),
    "cog": W(metal=.7, snap=.9, ring=.3, body=.4),
    "piston": W(metal=.5, body=.8, snap=.95, grit=.35, ring=.2),
    "valve": W(metal=.6, noise=.5, snap=.8, air=.4),
    "steam": W(noise=1.0, bright=.7, air=.9, ring=.6, motion=.4),
    "potion": W(wet=.9, bright=.6, ring=.4, space=.5, motion=.4, metal=.3),
    "spell": W(metal=.6, bright=.8, ring=.75, space=.7, motion=.5, rise=.5),
    "wand": W(metal=.7, bright=.9, ring=.7, space=.6, rise=.6, pitch=.85),
    "song": W(noise=.1, ring=.7, motion=.35, space=.45),
    "melodic": W(noise=.05, ring=.6, motion=.3),
    "hymn": W(noise=.1, ring=.85, space=.85, air=.4),
    "champagne": W(wet=.85, bright=.8, snap=.9, ring=.35, pitch=.7, metal=.3),
    "cork": W(wet=.6, bright=.6, snap=.95, ring=.2, body=.35),
    "bottle-pop": W(wet=.7, bright=.65, snap=.95, ring=.25),
    "wings": W(noise=.8, bright=.6, air=.85, motion=.7, snap=.3, ring=.3),
    "wing": W(noise=.8, bright=.6, air=.85, motion=.7, snap=.3),
    "flutter": W(noise=.7, air=.8, motion=.75, snap=.35, ring=.3),
    "curtain": W(noise=.85, bright=.4, air=.55, motion=.6, ring=.25),
    "code": W(digital=.7, bright=.7, snap=.9, ring=.15, noise=.1),
    "data": W(digital=.8, bright=.75, snap=.9, ring=.12),
    "byte": W(digital=.9, bright=.7, snap=.9, ring=.1),
    "terminal": W(digital=.75, bright=.65, snap=.9, ring=.15),
    "console": W(digital=.7, bright=.6, snap=.85, ring=.2),
    "keypad": W(digital=.55, bright=.65, snap=.95, ring=.12),
    "elevator": W(metal=.5, bright=.55, ring=.5, space=.5, body=.4),
    "train": W(noise=.7, body=.85, grit=.4, ring=.6, metal=.5),
    "car": W(noise=.6, body=.7, grit=.4, ring=.5),
    "clock-tick": W(wood=.5, snap=.98, ring=.12, bright=.6),
    "bubbling": W(wet=.9, noise=.4, bright=.6, motion=.5, ring=.4),
    "boiling": W(wet=.85, noise=.7, bright=.55, ring=.5),
    "pour": W(wet=.9, noise=.7, bright=.6, motion=.5, ring=.5),
    "pouring": W(wet=.9, noise=.7, bright=.6, motion=.5),
    "footsteps": W(wood=.6, body=.6, snap=.9, ring=.2, gesture="double"),
    "heartbeats": W(body=.95, pitch=.06, snap=.7, gesture="double"),
    "applause": W(noise=.95, bright=.6, snap=.4, ring=.5, space=.6),
    "crowd": W(noise=.9, bright=.45, ring=.7, space=.7, air=.4),
    "market": W(noise=.85, bright=.5, ring=.6, space=.6),
    "party": W(noise=.7, bright=.7, ring=.5, space=.5),
    "toy": W(pitch=.85, bright=.8, digital=.4, snap=.85, ring=.25, body=.1),
    "balloon-pop": W(noise=.6, snap=.98, bright=.7, ring=.15),
    "confetti": W(noise=.9, bright=.85, snap=.5, ring=.3, air=.4),
    "ribbon": W(noise=.8, bright=.7, air=.6, motion=.6, snap=.25),
    "feather": W(noise=.75, bright=.6, air=.95, snap=.15, ring=.25, body=.05),
    "dust": W(noise=.9, bright=.55, air=.7, snap=.2, ring=.2),
    "smoke": W(noise=.85, bright=.4, air=.85, snap=.12, ring=.5),
    "mist": W(noise=.85, bright=.55, air=.9, snap=.1, ring=.55, space=.6),
    "fog": W(noise=.8, bright=.3, air=.9, snap=.1, ring=.7, space=.7),
    "shadow": W(bright=.15, space=.6, ring=.55, warm=.4),
    "ghost": W(noise=.5, air=.85, bright=.45, ring=.85, space=.9, motion=.5),
    "dream": W(space=.85, ring=.85, snap=.15, bright=.5, air=.5),
    "memory": W(space=.7, ring=.7, bright=.4, warm=.7, snap=.25),
    "coffee": W(noise=.75, bright=.55, wet=.6, warm=.75, ring=.3),
    "tea": W(wet=.7, bright=.7, warm=.7, ring=.4, metal=.4),
    "kettle": W(noise=.85, bright=.75, air=.7, ring=.6, motion=.4),
    "bread": W(noise=.7, bright=.45, snap=.6, warm=.75, ring=.15),
    "chip": W(noise=.85, bright=.8, snap=.95, ring=.1, grit=.35),
    "candy": W(bright=.85, pitch=.8, snap=.85, ring=.25, digital=.3),
    "honey": W(wet=.7, warm=.9, bright=.35, snap=.2, ring=.5),
    "velcro": W(noise=.95, bright=.6, grit=.6, motion=.6, snap=.5),
    "scissors": W(metal=.75, bright=.8, snap=.95, ring=.2, gesture="double"),
    "stapler": W(metal=.6, body=.5, snap=.98, ring=.15),
    "envelope": W(noise=.88, bright=.6, air=.4, snap=.35, ring=.1),
    "stamp": W(wood=.5, body=.6, snap=.98, ring=.15),
    "pen": W(bright=.7, snap=.95, ring=.15, body=.15),

    # -- the cue names themselves ------------------------------------------
    # The role decides how long and how loud; these decide what a cue of
    # that kind sounds like when the prompt says nothing else. Without them
    # a bare "click" and a bare "hover" come out identically coloured.
    "click": W(snap=.93, ring=.12, bright=.58, body=.32, noise=.3),
    "tap": W(snap=.95, ring=.1, bright=.6, body=.25, noise=.3),
    "tick": W(snap=.97, ring=.08, bright=.78, body=.12, noise=.45),
    "press": W(snap=.9, ring=.15, bright=.5, body=.4),
    "hover": W(snap=.4, ring=.28, bright=.45, body=.12, noise=.12),
    "focus": W(snap=.45, ring=.3, bright=.5, body=.12),
    "toast": W(ring=.55, metal=.42, bright=.65, space=.35, noise=.08, snap=.75),
    "notification": W(ring=.55, metal=.45, bright=.68, space=.35, noise=.08,
                      snap=.78),
    "notify": W(ring=.55, metal=.45, bright=.68, space=.35, noise=.08, snap=.78),
    "popup": W(ring=.45, bright=.6, space=.3, snap=.7),
    "warning": W(bright=.55, ring=.35, grit=.15, pitch=.4),
    "caution": W(bright=.55, ring=.35, pitch=.42),
    "success": W(bright=.72, ring=.45, metal=.3, rise=.5, noise=.08),
    "confirm": W(bright=.7, ring=.42, metal=.28, rise=.45, noise=.08),
    "done": W(bright=.68, ring=.4, metal=.25, rise=.4),
    "fail": W(bright=.28, grit=.4, pitch=.2, rise=-.55, body=.6),
    "denied": W(bright=.3, grit=.4, pitch=.22, rise=-.5, body=.55),
    "pencil": W(noise=.6, wood=.5, bright=.6, snap=.7, ring=.12),
    "chalk": W(noise=.8, bright=.65, grit=.4, snap=.6, ring=.1),
    "eraser": W(noise=.7, bright=.4, grit=.3, snap=.4, ring=.1),

    # -- second gap-filling pass, from the stress sweep ---------------------
    "gate": W(metal=.7, body=.6, snap=.9, ring=.35, grit=.25),
    "hinge": W(metal=.65, grit=.5, motion=.7, ring=.4, bright=.6),
    "creak": W(metal=.4, wood=.5, grit=.55, motion=.8, ring=.4),
    "soap": W(wet=.85, bright=.55, snap=.4, ring=.25, grit=.02),
    "underwater": W(bright=.08, wet=.9, space=.7, ring=.6, air=.3, snap=.2),
    "explosion": W(noise=.85, body=.95, grit=.6, snap=.95, ring=.6, space=.6),
    "electric": W(bright=.8, digital=.45, grit=.45, metal=.4, ring=.35),
    "aeroplane": W(noise=.85, body=.7, motion=.5, ring=.7, air=.6, bright=.35),
    "boing": W(motion=.95, ring=.45, rise=.5, bright=.6, snap=.8, wood=.35),
    "melancholy": W(bright=.28, rise=-.55, ring=.65, warm=.65, snap=.3),
    "music": W(noise=.05, ring=.6, motion=.25),
    "box": W(wood=.7, body=.5, ring=.3, snap=.85, bright=.45),
    "fantasy": W(metal=.6, bright=.82, ring=.75, space=.7, rise=.45),
    "menu": W(snap=.7, bright=.55, ring=.3),
    "swipe": W(noise=.9, bright=.65, air=.7, motion=.85, snap=.3, ring=.3),
    "flick": W(noise=.7, bright=.75, motion=.7, snap=.9, ring=.15),
    "rattle": W(noise=.6, metal=.6, bright=.7, snap=.8, ring=.35, grain=.7),
    "squeak": W(bright=.85, pitch=.9, motion=.6, ring=.25, grit=.3),
    "slam": W(body=.9, snap=.98, grit=.4, ring=.3, noise=.5),
    "bang": W(body=.85, snap=.98, grit=.45, noise=.6, ring=.35),
}

# Granularity cuts across every category above - rain, fire, gravel and
# vinyl have nothing else in common - so it is tagged separately rather
# than repeated in thirty entries. A sound with grain is a cloud of small
# events, which filtered noise cannot imitate however it is shaped.
_GRAINY = {
    "crackle": .95, "fire": .8, "ember": .75, "sizzle": .85, "match": .7,
    "candle": .45, "rain": .9, "drizzle": .85, "storm": .5, "thunder": .2,
    "gravel": .95, "sand": .5, "snow": .6, "leaves": .5, "leaf": .45,
    "forest": .35, "grass": .4, "dust": .45, "static": .75, "radio": .5,
    "vinyl": .85, "tape": .45, "cassette": .5, "old": .35, "lofi": .4,
    "cricket": .6, "insect": .5, "applause": .9, "confetti": .65,
    "crowd": .55, "market": .5, "chip": .45, "chalk": .5, "velcro": .6,
    "boiling": .6, "bubbling": .5, "kettle": .4, "coffee": .5, "steam": .45,
    "gritty": .5, "rough": .4, "crunchy": .7, "grainy": .95, "dirty": .45,
    "sparkle": .35, "glitch": .5, "engine": .4, "machine": .35,
}
for _w, _g in _GRAINY.items():
    if _w in LEXICON:
        LEXICON[_w]["grain"] = _g

# Words that pick the *shape* of the gesture rather than its colour.
GESTURE_WORDS = {
    "double": "double", "twice": "double", "twin": "double", "pair": "double",
    "triple": "arp", "arpeggio": "arp", "fanfare": "arp", "flourish": "arp",
    "sweep": "sweep", "whoosh": "sweep", "swoosh": "sweep", "swipe": "sweep",
    "transition": "sweep", "slide": "sweep", "wipe": "sweep", "pass": "sweep",
    "rising": "rise2", "ascending": "rise2", "upward": "rise2", "up": "rise2",
    "falling": "fall2", "descending": "fall2", "downward": "fall2",
    "down": "fall2", "bell": "bell", "chime": "bell", "gong": "bell",
    "ring": "bell", "single": "single", "one": "single",
}

# Playing a cue backwards is how you get its partner: a toast that slides
# in has to slide back out, and the retract is the same sound reversed.
# Only the literal effect lives here. "Retract" used to, and that was the
# bug: a retracting panel is not its entrance played backwards, it is the
# same motion going the other way. Those words now pick the toast_out role,
# which builds that properly. This flag stays for the effect people
# actually mean by reverse - the sucked-in swell.
REVERSE_WORDS = {"reverse", "reversed", "reversing", "backwards", "backward",
                 "rewind", "rewound", "inhale", "inhaled", "suck", "sucked",
                 "unswoosh", "swell-in"}

# Words that say which UI cue this is for. The role owns duration and level;
# the character only ever owns timbre.
ROLE_WORDS = {
    "hover": "hover", "rollover": "hover", "focus": "hover",
    "highlight": "hover", "hovering": "hover",
    "click": "click", "clicking": "click", "press": "click", "tap": "click",
    "button": "click", "select": "click", "tick": "click", "toggle": "click",
    "switch": "click", "keypress": "click", "type": "click", "typing": "click",
    "toast": "toast", "notification": "toast", "notify": "toast",
    "alert": "toast", "banner": "toast", "popup": "toast", "message": "toast",
    "ping": "toast", "reminder": "toast",
    "retract": "toast_out", "retracting": "toast_out",
    "retracted": "toast_out", "dismiss": "toast_out",
    "dismissing": "toast_out", "withdraw": "toast_out",
    "withdrawing": "toast_out", "hide": "toast_out", "leaving": "toast_out",
    "exit": "toast_out", "outgoing": "toast_out",
    "ok": "ok", "success": "ok", "confirm": "ok", "confirmed": "ok",
    "done": "ok", "complete": "ok", "accept": "ok", "save": "ok",
    "saved": "ok", "yes": "ok", "positive": "ok",
    "warn": "warn", "warning": "warn", "caution": "warn",
    "error": "error", "fail": "error", "failure": "error", "denied": "error",
    "invalid": "error", "reject": "error", "rejected": "error",
    "wrong": "error", "negative": "error",
}

ROLES = ("hover", "click", "toast", "toast_out", "ok", "warn", "error")

# role -> the more general role it refines
_ROLE_BASE = {"toast_out": "toast"}

_INTENSIFY = {"very": 1.8, "super": 1.8, "extremely": 2.1, "really": 1.5,
              "ultra": 2.0, "mega": 1.9, "hyper": 1.9, "so": 1.4,
              "quite": 1.2, "totally": 1.5, "insanely": 2.0,
              "slightly": 0.45, "bit": 0.45, "barely": 0.3, "somewhat": 0.55,
              "little": 0.45, "faintly": 0.35, "mildly": 0.5, "kinda": 0.6,
              "sorta": 0.6, "vaguely": 0.4}

_NEGATE = {"no", "not", "without", "less", "never", "avoid", "minus",
           "non", "anti", "isnt", "arent", "dont"}

_STOP = {"a", "an", "the", "of", "for", "and", "with", "sound", "sounds",
         "effect", "effects", "sfx", "audio", "noise", "please", "make",
         "me", "some", "kind", "like", "that", "this", "is", "it", "its",
         "ui", "interface", "app", "style", "feel", "feels", "feeling",
         "vibe", "something", "want", "need", "give", "generate", "create",
         "when", "then", "from", "into", "onto", "at", "on", "in", "to",
         "as", "by", "or", "but", "my", "your", "i", "you", "we",
         # Context words people write around the description. They carry no
         # timbre, and reporting them as "not understood" is noise.
         "frequency", "game", "being", "used", "using", "should", "would",
         "sounds", "sound-effect", "away", "very-", "one", "two", "kinda",
         "type-of", "similar", "version", "please-make", "im", "ive"}

_PHRASES = (
    (r"\b8\s*[-]?\s*bit\b", "8bit"), (r"\b16\s*[-]?\s*bit\b", "16bit"),
    (r"\bsci\s*[-]?\s*fi\b", "futuristic"), (r"\blo\s*[-]?\s*fi\b", "lofi"),
    (r"\bhi\s*[-]?\s*fi\b", "clean"), (r"\bpower\s*[-]?\s*up\b", "powerup"),
    (r"\blevel\s*[-]?\s*up\b", "powerup"), (r"\bwind\s*chimes?\b", "chime"),
    # "page turn" is deliberately not collapsed: "page" names the material
    # and "turn" names the action, and folding them threw the verb away -
    # which is how a page turn came out as a struck sheet of paper.
    (r"\bwater\s*drop\b", "droplet"),
    (r"\bnew\s*message\b", "toast"), (r"\bpush\s*button\b", "button"),
    (r"\bmouse\s*over\b", "hover"), (r"\bpop\s*up\b", "popup"),
    (r"\bnotification\s*bell\b", "toast bell"),
)

# Extra names for a few concepts, kept out of the main table so the feature
# lists there stay readable.
_SYNONYM = {
    "clicky": "click", "clicks": "click", "beeps": "beep", "dings": "bell",
    "ding": "bell", "bells": "bell", "chimes": "chime", "glassy": "glass",
    "metallic": "metal", "wooden": "wood", "papery": "paper",
    "watery": "water", "breezy": "breeze", "windy": "wind", "rainy": "rain",
    "stormy": "storm", "sandy": "sand", "icy": "ice", "starry": "star",
    "fiery": "fire", "smoky": "fire", "dusty": "old", "rusty": "old",
    "crystalline": "crystal", "digitized": "digital", "8-bit": "8bit",
    "scifi": "futuristic", "cyber": "cyberpunk", "tech": "computer",
    "buzzer": "buzz", "buzzing": "buzz", "humming": "hum", "whoosh": "wind",
    "swoosh": "wind", "swipe": "wind", "powerup": "coin", "sparkles": "sparkle",
    "twinkle": "sparkle", "twinkly": "sparkle", "shimmer": "shimmering",
    "boom": "thunder", "thump": "kick", "thud": "kick", "impact": "kick",
    "knock": "woodblock", "clack": "woodblock", "clunk": "hatch",
    "clank": "metal", "clang": "metal", "pop": "bubble", "plop": "droplet",
    "blop": "bubble", "squish": "wet", "slime": "wet", "goo": "wet",
    "zap": "laser", "pew": "laser", "blast": "laser", "shot": "laser",
    "scan": "radar", "scanner": "radar", "beeping": "beep",
    "rumble": "thunder", "roar": "thunder", "crash": "cymbal",
    "shatter": "glass", "smash": "glass", "crack": "crackle",
    "whisper": "breathy", "sigh": "breathy", "breath": "breathy",
    "footstep": "woodblock", "step": "woodblock", "walk": "woodblock",
    "keys": "keyboard", "typewriting": "typewriter", "scroll": "paper",
    "coins": "coin", "money": "coin", "cash": "coin", "purchase": "coin",
    "reward": "coin", "achievement": "chime", "trophy": "chime",
    "unlock": "lock", "locked": "lock", "opening": "door", "closing": "door",
    "startup": "chime", "boot": "chime", "shutdown": "gong",
    "loading": "beep", "progress": "beep", "delete": "static",
    "trash": "static", "remove": "static", "send": "wind", "sent": "wind",
    "receive": "bell", "incoming": "bell", "download": "beep",
    "upload": "wind", "refresh": "beep", "sync": "beep",
    "teapot": "teacup", "mug": "cup", "glassware": "glass",
    "wind-chime": "chime", "raindrop": "droplet", "waterdrop": "droplet",
    "thunderclap": "thunder", "birdsong": "bird", "tweet": "bird",
    "airplane": "aeroplane", "plane": "aeroplane", "jet": "aeroplane",
    "far": "distant", "faraway": "distant", "musicbox": "box",
    "boop-boop": "beep", "squelch": "wet", "drone": "hum",
    "sad-piano": "melancholy", "gloomy": "melancholy", "moody": "melancholy",
    "rusty": "old", "creaky": "creak", "squeaky": "squeak",
    "electrical": "electric", "electricity": "electric", "spark": "electric",
}

_ALL_WORDS = None


def _vocabulary():
    global _ALL_WORDS
    if _ALL_WORDS is None:
        _ALL_WORDS = (set(LEXICON) | set(_SYNONYM) | set(GESTURE_WORDS)
                      | set(ROLE_WORDS) | set(_material.MATERIAL_WORDS)
                      | set(_material.EXCITER_WORDS))
    return _ALL_WORDS


def tokens(prompt):
    p = " " + (prompt or "").lower().strip() + " "
    for pat, sub in _PHRASES:
        p = re.sub(pat, " " + sub + " ", p)
    return [t for t in re.split(r"[^a-z0-9]+", p) if t and t not in _STOP]


_MORPH = (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("s", ""),
          ("es", ""), ("ies", "y"), ("y", ""), ("y", "e"), ("ly", ""),
          ("ish", ""), ("ish", "e"), ("like", ""), ("er", ""), ("est", ""))


def resolve_word(tok):
    """One token -> a vocabulary word, by four widening passes.

    Returns (word, how) or (None, None). `how` is kept so the UI can be
    honest about which words were guessed rather than known.
    """
    vocab = _vocabulary()
    if tok in vocab:
        return tok, "exact"

    for cut, add in _MORPH:
        if tok.endswith(cut) and len(tok) > len(cut) + 2:
            cand = tok[:-len(cut)] + add
            if cand in vocab:
                return cand, "morph"

    # "thunderstorm" -> thunder + storm. The longer half wins: it carries
    # more of the meaning, and both halves usually agree anyway.
    if len(tok) >= 7:
        best = None
        for i in range(3, len(tok) - 2):
            a, b = tok[:i], tok[i:]
            if a in vocab and b in vocab:
                cand = a if len(a) >= len(b) else b
                if best is None or len(cand) > len(best):
                    best = cand
        if best:
            return best, "compound"

    close = difflib.get_close_matches(tok, vocab, n=1, cutoff=0.84)
    if close:
        return close[0], "fuzzy"
    return None, None


@dataclass
class Reading:
    """Everything the prompt asked for, and everything it did not say."""
    character: Character
    role: str = "click"
    role_explicit: bool = False
    gesture: str = ""
    reverse: bool = False
    materials: tuple = ()      # named substances, in the order written
    exciter: str = ""          # what is being done to them
    known: tuple = ()          # (token, resolved word, how)
    unknown: tuple = ()

    def understood(self):
        return tuple(w for _, w, _ in self.known)


def read(prompt, default_role="click"):
    """Prompt -> Reading. Never raises, never returns nothing usable."""
    acc = {k: [0.0, 0.0] for k in KEYS}       # feature -> [sum(w*v), sum(w)]
    role, role_explicit, gesture = default_role, False, ""
    known, unknown = [], []
    boost, negate = 1.0, False
    reverse = False
    materials, exciter = [], ""

    for tok in tokens(prompt):
        if tok in _INTENSIFY:
            boost = _INTENSIFY[tok]
            continue
        if tok in _NEGATE:
            negate = True
            continue

        if tok in REVERSE_WORDS:
            reverse = not negate
            known.append((tok, tok, "exact"))
            boost, negate = 1.0, False
            continue

        word, how = resolve_word(tok)
        if word is None:
            unknown.append(tok)
            boost, negate = 1.0, False
            continue
        known.append((tok, word, how))
        # Role and gesture are read from the word as typed; features from
        # what it is a synonym of. "whoosh" is a sweep *and* is coloured
        # like wind, and folding it to "wind" first would lose the sweep.
        if word in ROLE_WORDS:
            new_role = ROLE_WORDS[word]
            # A refinement is not undone by its own base word. "dismiss
            # toast" names one cue, and reading it left to right would let
            # the plain "toast" overwrite the "dismiss" that qualified it.
            if role_explicit and _ROLE_BASE.get(role) == new_role:
                pass
            else:
                role = new_role
            role_explicit = True
        if word in GESTURE_WORDS:
            gesture = GESTURE_WORDS[word]

        # Substance and action are *selected*, not averaged. Naming
        # porcelain has to give you porcelain; blending it 60% toward a
        # neutral default is how "porcelain teacup" ended up sounding like
        # every other cue.
        if word in _material.MATERIAL_WORDS:
            m = _material.MATERIAL_WORDS[word]
            if m not in materials:
                materials.append(m)
        if word in _material.EXCITER_WORDS:
            exciter = _material.EXCITER_WORDS[word]

        spec = dict(LEXICON.get(_SYNONYM.get(word, word), {}))
        spec.pop("role", None)
        gesture = spec.pop("gesture", gesture)
        weight = float(spec.pop("w", 1.0)) * boost
        # A guessed word should not outvote one the user actually typed.
        if how in ("fuzzy", "compound"):
            weight *= 0.75
        for feat, val in spec.items():
            if negate:
                val = 1.0 - val if feat != "rise" else -val
            acc[feat][0] += val * weight
            acc[feat][1] += weight
        boost, negate = 1.0, False

    ch = Character()
    for feat, (total, wsum) in acc.items():
        if wsum > 0:
            mean = total / wsum
            # Pull toward the asserted value rather than snapping to it, so
            # one strong word cannot erase a whole prompt's worth of nuance.
            base = getattr(ch, feat)
            # One confident word should mostly get its way; the softening
            # only exists so a passing adjective cannot erase the rest.
            pull = min(1.0, wsum / (wsum + 0.32))
            setattr(ch, feat, base + (mean - base) * pull)

    return Reading(character=ch, role=role, role_explicit=role_explicit,
                   gesture=gesture, reverse=reverse, known=tuple(known),
                   unknown=tuple(unknown), materials=tuple(materials),
                   exciter=exciter)


def vocabulary_size():
    return len(_vocabulary())
