"""A sound is data: a small stack of layers plus a master chain.

Keeping the whole sound as a plain dataclass tree - never as a buffer with
a history - is what makes the rest of the app possible. The prompt parser
writes one, the editor mutates one, the library stores one as JSON, and the
exported pack ships one. Audio is only ever the output.
"""

from dataclasses import dataclass, field, asdict, replace
import numpy as np

from . import engine as E
from . import material

# Struck-plate partial ratios, from the classic tubular-bell measurements.
# The point is that they are not whole numbers - that is the whole sound.
BELL_RATIOS = (1.0, 1.83, 2.41, 3.14, 4.27, 5.12, 6.37)


@dataclass
class Layer:
    """One voice. `kind` picks which of the five generators runs."""
    kind: str = "tone"     # tone | noise | pluck | metal | sub | texture | grain
    gain: float = 1.0
    pan: float = 0.0              # -1 left .. +1 right
    start: float = 0.0            # seconds of silence before this layer
    dur: float = 0.12

    # pitch (tone, pluck, metal, sub)
    f0: float = 900.0
    f1: float = 380.0
    curve: str = "exp"            # exp | lin | snap | slow
    wave: str = "sine"            # sine | tri | square | saw
    bright: float = 1.0
    detune: float = 0.0           # cents, a second voice for thickness

    # frequency modulation (tone)
    fm_ratio: float = 0.0
    fm_index: float = 0.0
    fm_decay: float = 6.0

    # metal
    blend: float = 1.0            # 1 = fully inharmonic, 0 = harmonic
    ring: float = 4.0             # partial decay rate

    # pluck
    damp: float = 0.996
    tone: float = 0.5

    # noise
    noise_kind: str = "white"     # white | pink | brown | blue

    # modulation (texture): a wandering cutoff and gusting level are what
    # separate wind from a hiss, and they are the same two controls for
    # fire, rain, waves and engine noise.
    mod_rate: float = 0.5         # Hz
    mod_depth: float = 0.6        # 0 = steady, 1 = full gusts
    mod_shape: str = "random"     # random | sine | tri

    # modal: an object, and something happening to it. `material` names a
    # resonant fingerprint in material.py, `exciter` what is done to it.
    # This is the voice that sounds like a thing rather than like a tone.
    material: str = "plastic"
    exciter: str = "strike"
    hardness: float = 0.6         # contact time: felt mallet .. steel tip
    mat_damp: float = 1.0         # scales every mode's decay time
    mat_spread: float = 0.0       # detune the partials off their ideal ratios
    modes_n: int = 0              # 0 = every mode the material has

    # sample: audio the model generated, referenced by cache id. The clip
    # itself lives on disk - a recipe stays small, diffable JSON, and the
    # rest of the chain treats it like any other voice.
    clip: str = ""

    # granular (grain): rain, crackle, sizzle, gravel
    density: float = 400.0        # grains per second
    grain_ms: float = 22.0
    pitched: bool = False         # pings rather than ticks

    # amplitude envelope
    attack: float = 0.002
    hold: float = 0.0
    decay: float = 5.0
    sustain: float = 0.0
    release: float = 0.006

    # per-layer filter
    filt: str = "lp"              # none | lp | hp | bp
    cut0: float = 16000.0
    cut1: float = 16000.0
    res: float = 0.0
    poles: int = 2


@dataclass
class Recipe:
    """A complete sound. Duration is derived, never stored - a layer that
    outlives the nominal length would otherwise be silently truncated."""
    name: str = "sound"
    prompt: str = ""
    seed: int = 0
    layers: list = field(default_factory=lambda: [Layer()])

    drive: float = 0.0
    bits: int = 0                 # 0 = off
    crush_rate: int = 0           # 0 = off
    highpass: float = 0.0         # 0 = off
    lowpass: float = 0.0          # 0 = off
    echo_time: float = 0.0
    echo_fb: float = 0.35
    echo_mix: float = 0.0
    rev_mix: float = 0.0
    rev_size: float = 0.3
    rev_damp: float = 0.5
    width: float = 0.0
    gain: float = 1.0
    peak_db: float = -1.5         # normalise target
    loop: bool = False            # wrap the tail into the head, seamlessly
    reverse: bool = False         # play backwards - a cue's retract partner

    def duration(self):
        return max((L.start + L.dur for L in self.layers), default=0.1)

    def copy(self):
        return replace(self, layers=[replace(L) for L in self.layers])

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        layers = [Layer(**{k: v for k, v in L.items()
                           if k in Layer.__dataclass_fields__})
                  for L in d.pop("layers", [])]
        keep = {k: v for k, v in d.items() if k in Recipe.__dataclass_fields__}
        return Recipe(layers=layers or [Layer()], **keep)


# ------------------------------------------------------------------ voices

def _sample_voice(L: Layer, n: int):
    """A generated clip, cut to the length this cue is allowed.

    Imported lazily and failing to silence on purpose: a recipe that
    references a clip must still open on a machine where the model was
    never installed, or the library becomes unreadable the moment somebody
    copies it somewhere else.
    """
    try:
        from . import neural
        a = neural.cached(L.clip)
    except Exception:                                 # noqa: BLE001
        a = None
    if a is None:
        return np.zeros(n)
    m = a.mean(axis=1) if a.ndim > 1 else np.asarray(a)
    if len(m) < n:
        m = np.pad(m, (0, n - len(m)))
    return np.array(m[:n], dtype=np.float64)


def _sample_stereo(L: Layer):
    """(left, right) for a clip layer, or None if it is not usable.

    Same envelope and filter as any other voice - the clip is a voice, it
    just came from a model instead of an oscillator.
    """
    try:
        from . import neural
        a = neural.cached(L.clip)
    except Exception:                                 # noqa: BLE001
        a = None
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    n = max(8, int(L.dur * E.RATE))
    if len(a) < n:
        a = np.pad(a, ((0, n - len(a)), (0, 0)))
    a = a[:n]
    env = E.amp_env(n, L.attack, L.decay, L.hold, L.release, L.sustain)
    out = []
    for ch in (a[:, 0], a[:, 1]):
        x = ch * env
        if L.filt not in (None, "none"):
            x = E.sweep(x, L.filt, L.cut0, L.cut1, L.res, max(1, int(L.poles)))
            x = x * env
        out.append(x * L.gain)
    return out[0], out[1]


def _exciter(L: Layer, n: int, seed: int):
    """The event, before the object has coloured it.

    Kept separate from the resonator on purpose: tapping a glass and
    scraping a glass are obviously the same glass, and that only falls out
    for free if the two halves never know about each other.
    """
    kind = L.exciter
    if kind == "scrape":
        return E.scrape(n, roughness=1.0 - L.hardness * 0.6,
                        speed=float(np.clip(L.mod_rate / 8.0, 0.05, 1.0)),
                        seed=seed)
    if kind == "blow":
        air = E.noise(n, "pink", seed)
        wander = E.lfo(n, max(L.mod_rate, 0.05), "random", seed + 3)
        return air * (0.35 + 0.65 * wander)
    if kind == "particles":
        return E.grains(n, max(L.density, 4.0), L.grain_ms,
                        min(L.f0, L.f1), max(L.f0, L.f1) * 3.0,
                        decay=9.0, seed=seed)
    if kind == "bubble":
        return E.bubble(n, L.f0, rise=1.4 + 2.4 * L.hardness,
                        decay=max(6.0, L.decay))
    if kind == "pluck":
        # A pluck is a release, not an impact: longer contact, no rattle.
        return E.strike(n, hardness=L.hardness * 0.55, seed=seed, noisy=0.05)
    return E.strike(n, hardness=L.hardness, seed=seed,
                    noisy=0.1 + 0.5 * (1.0 - L.hardness))


def _modal_voice(L: Layer, n: int, seed: int):
    mat = material.get(L.material) or material.default_for(L.exciter)
    rng = np.random.default_rng(seed ^ 0x5EED)
    freqs, amps, taus = material.modes(
        mat, f0=L.f0, hardness=L.hardness, count=L.modes_n,
        damp=max(L.mat_damp, 0.02), spread=L.mat_spread, rng=rng)
    exc = _exciter(L, n, seed)
    if L.exciter == "bubble":
        # The bubble is already the whole sound; the body only tints it.
        return 0.75 * exc + 0.25 * E.modal(exc, freqs, amps, taus, n_out=n)
    x = E.modal(exc, freqs, amps, taus, n_out=n)
    # Contact noise: the scuff of the hit itself, which every real strike
    # has and no pure resonator produces.
    if mat.noise > 0.01:
        rub = E.noise(n, "pink", seed + 11)
        rub *= E.amp_env(n, 0.0004, 40.0, release=0.002)
        x = x + rub * mat.noise * (0.35 + 0.65 * L.hardness)
    return x / (float(np.max(np.abs(x))) or 1.0)


def _voice(L: Layer, seed: int):
    """Render one layer to mono, unpanned, at its own length."""
    n = max(8, int(L.dur * E.RATE))
    env = E.amp_env(n, L.attack, L.decay, L.hold, L.release, L.sustain)
    pre_filtered = False

    if L.kind == "sample":
        x = _sample_voice(L, n)
    elif L.kind == "modal":
        x = _modal_voice(L, n, seed)
    elif L.kind == "texture":
        # A noise bed whose filter wanders and whose level gusts. Both
        # modulators are deliberately at different rates and seeds: locking
        # them together makes the result pulse like a machine.
        x = E.noise(n, L.noise_kind, seed)
        lo_c, hi_c = sorted((max(L.cut0, 20.0), max(L.cut1, 21.0)))
        m = E.lfo(n, max(L.mod_rate, 0.01), L.mod_shape, seed + 3)
        x = E.bank_filter(x, L.filt if L.filt != "none" else "bp",
                          lo_c * (hi_c / lo_c) ** m, L.res,
                          max(1, int(L.poles)))
        gust = E.lfo(n, max(L.mod_rate * 0.63, 0.01), L.mod_shape, seed + 91)
        x = x * (1.0 - L.mod_depth + L.mod_depth * gust)
        pre_filtered = True
    elif L.kind == "grain":
        x = E.grains(n, L.density, L.grain_ms, min(L.f0, L.f1),
                     max(L.f0, L.f1), L.decay, seed=seed, pitched=L.pitched)
        if L.mod_depth > 0.01:
            # Rain and fire come in waves; a flat grain cloud sounds synthetic.
            swell = E.lfo(n, max(L.mod_rate, 0.01), L.mod_shape, seed + 17)
            x = x * (1.0 - L.mod_depth + L.mod_depth * swell)
    elif L.kind == "noise":
        x = E.noise(n, L.noise_kind, seed)
    elif L.kind == "pluck":
        x = E.karplus(n, L.f0, L.damp, L.tone, seed)
    else:
        f = E.freq_env(n, L.f0, L.f1, L.curve)
        p = E.phase_of(f)
        if L.kind == "metal":
            x = E.inharmonic(p, n, BELL_RATIOS,
                             [L.ring * (1.0 + i * 0.55)
                              for i in range(len(BELL_RATIOS))], L.blend)
        elif L.kind == "sub":
            x = np.sin(p)
        else:
            if L.fm_index > 0.01 and L.fm_ratio > 0.01:
                idx = L.fm_index * np.exp(-L.fm_decay * np.linspace(0, 1, n))
                x = np.sin(p + idx * np.sin(p * L.fm_ratio))
            else:
                x = E.osc(p, f, L.wave, L.bright)
            if abs(L.detune) > 0.5:
                # A second voice a few cents away. Detune is what separates
                # "a tone" from "a sound with body".
                p2 = E.phase_of(f * 2.0 ** (L.detune / 1200.0))
                x = 0.62 * x + 0.62 * E.osc(p2, f, L.wave, L.bright)

    x = x * env
    if L.filt not in (None, "none") and not pre_filtered:
        x = E.sweep(x, L.filt, L.cut0, L.cut1, L.res, max(1, int(L.poles)))
        x = x * env               # the filter rings past the tail; re-gate it
    return x * L.gain


# ------------------------------------------------------------------ render

def render(r: Recipe, stereo=True):
    """Recipe -> float32 array, shape (n,) mono or (n, 2) stereo."""
    total = max(8, int(r.duration() * E.RATE) + 8)
    left = np.zeros(total)
    right = np.zeros(total)

    for i, L in enumerate(r.layers):
        seed_i = r.seed * 977 + i * 31 + 1
        # A generated clip keeps its own stereo image. Summing it to mono
        # and re-panning would throw away width the model put there, and
        # can cancel outright if the two channels are decorrelated.
        pair = _sample_stereo(L) if L.kind == "sample" else None
        v = pair[0] if pair is not None else _voice(L, seed_i)
        at = min(int(L.start * E.RATE), total - 1)
        end = min(at + len(v), total)
        pan = float(np.clip(L.pan, -1.0, 1.0))
        # Equal power, so a centred layer is not quietly louder than a
        # panned one at the same gain.
        gl = np.cos((pan + 1.0) * np.pi / 4.0)
        gr = np.sin((pan + 1.0) * np.pi / 4.0)
        left[at:end] += v[:end - at] * gl * 1.4142
        right[at:end] += (pair[1] if pair is not None else v)[:end - at] \
            * gr * 1.4142

    def master(x, ir_seed):
        x = E.drive(x, r.drive)
        x = E.bitcrush(x, r.bits, r.crush_rate)
        if r.highpass > 20:
            x = E.filt(x, "hp", r.highpass, 0.0, 2)
        if r.lowpass > 20:
            x = E.filt(x, "lp", r.lowpass, 0.0, 2)
        x = E.echo(x, r.echo_time, r.echo_fb, r.echo_mix)
        if r.rev_mix > 0.001:
            ir = E._impulse(r.rev_size, r.rev_damp, ir_seed)
            n = 1 << (len(x) + len(ir)).bit_length()
            wet = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n),
                               n)[:len(x) + len(ir)]
            dry = np.concatenate([x, np.zeros(len(wet) - len(x))])
            wet = wet * ((float(np.max(np.abs(dry))) or 1.0) /
                         (float(np.max(np.abs(wet))) or 1.0))
            x = dry * (1.0 - r.rev_mix * 0.35) + wet * r.rev_mix
        return x

    # Decorrelated reverb tails are most of the perceived width; the rest is
    # a very short delay, kept under 1ms so it stays a widening and not an echo.
    left = master(left, 7)
    right = master(right, 8 if r.width > 0.01 or r.rev_mix > 0.001 else 7)
    n = max(len(left), len(right))
    left = np.pad(left, (0, n - len(left)))
    right = np.pad(right, (0, n - len(right)))

    if r.width > 0.01:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * (1.0 + r.width * 2.5)
        left, right = mid + side, mid - side

    out = np.stack([left, right], axis=1) * r.gain

    if r.loop and len(out) > 8192:
        # Fold the tail back over the head so the file joins to itself. An
        # ambience that has to fade out at both ends is not an ambience -
        # it is a sound effect of some wind happening once.
        f = min(int(0.3 * E.RATE), len(out) // 3)
        ramp = (0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, f))))[:, None]
        out = np.concatenate([out[:f] * ramp + out[-f:] * (1.0 - ramp),
                              out[f:-f]])
    else:
        # Guarantee silence at both ends. Any step away from zero at a
        # boundary is a click, and that click is what "cheap" sounds like.
        # The lead-in is deliberately tiny - the layer envelopes already
        # start at zero, and a 1.5ms ramp would eat a 12ms tick outright.
        fi = min(12, len(out) // 8)
        fo = min(96, len(out) // 4)
        if fi > 1:
            out[:fi] *= (0.5 * (1.0 - np.cos(
                np.pi * np.linspace(0, 1, fi))))[:, None]
        if fo > 1:
            out[-fo:] *= (0.5 * (1.0 - np.cos(
                np.pi * np.linspace(1, 0, fo))))[:, None]

    if r.reverse:
        # Trim before reversing, never after. The quiet part of a cue is its
        # decay, so reversing an untrimmed one puts a few hundred ms of dead
        # air in front of the sound - which reads as lag, not as a swell.
        # A loop has no silence at either end and must keep its full length
        # or it stops joining to itself.
        if not r.loop:
            out = E.trim_tail(out)
        out = np.ascontiguousarray(out[::-1])

    # Normalise after the fades, not before, or a transient whose peak sits
    # inside the lead-in ramp comes out 20dB quiet.
    peak = float(np.max(np.abs(out)))
    if peak > 1e-7:
        out = out * (10.0 ** (r.peak_db / 20.0) / peak)

    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    return out if stereo else out.mean(axis=1)
