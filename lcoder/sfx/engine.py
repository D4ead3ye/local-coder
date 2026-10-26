"""The synthesis engine - numpy in, mono float64 out.

Everything here is vectorised. Three choices are worth explaining, because
they are the difference between a UI sound and a fizzy approximation of one:

  * Filters are minimum-phase, not zero-phase. Applying a magnitude curve
    with rfft -> multiply -> irfft is one line, and it smears the transient
    symmetrically, so a click grows an audible "pre-click" whoosh in front
    of it. Deriving the phase through the real cepstrum costs ten lines and
    keeps the attack where it was.

  * A filter sweep is K statically-filtered copies crossfaded along the
    cutoff envelope, not a per-sample recursion. Time-varying one-poles are
    inherently sequential, and a Python loop over 20k samples per layer
    makes live parameter editing feel dead. The crossfade is close enough
    that nothing audible survives at K=9.

  * Oscillators above a sine are additive and band-limited, with every
    harmonic gated as it approaches Nyquist. A naive saw at 4kHz aliases
    into a metallic buzz that sounds like a bug, not a texture.
"""

import numpy as np

RATE = 44100
NYQ = RATE * 0.5


# ---------------------------------------------------------------- envelopes

def amp_env(n, attack=0.002, decay=4.0, hold=0.0, release=0.004, sustain=0.0):
    """Raised-cosine attack, optional hold, exponential decay, and a short
    release fade that guarantees the tail reaches exactly zero.

    A waveform that stops at non-zero amplitude clicks, and that click is
    most of what people call "harsh" - no amount of filtering removes it.
    """
    if n <= 1:
        return np.zeros(max(n, 0))
    env = np.ones(n)
    a = max(1, int(attack * RATE))
    if a > 1:
        a = min(a, n)
        env[:a] = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, a)))
    h = min(n - a, int(hold * RATE))
    tail = n - a - h
    if tail > 0:
        u = np.linspace(0, 1, tail)
        env[a + h:] = (1.0 - sustain) * np.exp(-decay * u) + sustain
    r = min(int(release * RATE), n // 2)
    if r > 1:
        env[-r:] *= 0.5 * (1.0 - np.cos(np.pi * np.linspace(1, 0, r)))
    return env


def freq_env(n, f0, f1, curve="exp"):
    """Instantaneous frequency per sample. A falling pitch reads as a
    physical knock; the same tone held flat reads as an alert."""
    u = np.linspace(0.0, 1.0, max(n, 1))
    f0 = max(float(f0), 1e-3)
    f1 = max(float(f1), 1e-3)
    if curve == "lin":
        return f0 + (f1 - f0) * u
    if curve == "snap":                   # most of the drop in the first 15%
        k = (1.0 - np.exp(-6.0 * u)) / (1.0 - np.exp(-6.0))
        return f0 + (f1 - f0) * k
    if curve == "slow":                   # holds, then falls away at the end
        return f0 + (f1 - f0) * u ** 3.0
    return f0 * (f1 / f0) ** u            # "exp" - a constant musical rate


def phase_of(f):
    """Integrate frequency into phase.

    Evaluating sin(2*pi*f*t) with a moving f warps the wave and buzzes as
    the pitch slides; only the running sum is right.
    """
    return np.cumsum(2.0 * np.pi * f / RATE)


# -------------------------------------------------------------- oscillators

_PROFILES = {                # (harmonic step, amplitude exponent)
    "sine":   (0, 0.0),
    "tri":    (2, 2.0),
    "square": (2, 1.0),
    "saw":    (1, 1.0),
}


def osc(phase, f, wave="sine", bright=1.0, harmonics=24):
    """Band-limited additive oscillator.

    `bright` scales how far up the harmonic series we go, which is a more
    musical brightness control than a filter alone.
    """
    if wave == "sine" or harmonics <= 1:
        return np.sin(phase)
    step, expo = _PROFILES.get(wave, _PROFILES["saw"])
    out = np.sin(phase)
    fmax = float(np.max(f))
    limit = int(min(harmonics, max(1, NYQ / max(fmax, 1.0))))
    k = 1 + step
    while k <= limit:
        amp = (1.0 / k ** expo) * float(np.clip(bright, 0.0, 4.0))
        if amp < 0.002:
            break
        # Gate each partial out smoothly as it nears Nyquist, rather than
        # letting it fold back down the spectrum as a buzz.
        gate = np.clip((0.92 * NYQ - k * f) / (0.08 * NYQ), 0.0, 1.0)
        if float(np.max(gate)) > 0.0:
            sign = -1.0 if (wave == "tri" and (k // 2) % 2 == 0) else 1.0
            out = out + sign * amp * gate * np.sin(k * phase)
        k += step
    return out / (float(np.max(np.abs(out))) or 1.0)


def inharmonic(phase, n, ratios, decays, blend=1.0):
    """A stack of partials at non-integer ratios - bells, glass, struck
    metal. `blend` pulls the ratios back toward whole numbers, which walks
    the sound from a struck plate to a plain musical tone.
    """
    out = np.zeros(n)
    u = np.linspace(0.0, 1.0, n)
    for r, d in zip(ratios, decays):
        rr = r * blend + round(r) * (1.0 - blend)
        out = out + np.exp(-d * u) * np.sin(phase * rr) / (1.0 + 0.7 * rr)
    return out / (float(np.max(np.abs(out))) or 1.0)


def noise(n, kind="white", seed=0):
    """White, pink, brown or blue noise shaped in the frequency domain, so
    the slope is exact rather than an approximation that drifts."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if kind == "white" or n < 8:
        return x / (float(np.max(np.abs(x))) or 1.0)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    f[0] = f[1]
    expo = {"pink": 0.5, "brown": 1.0, "blue": -0.5}.get(kind, 0.5)
    y = np.fft.irfft(spec * (f[1] / f) ** expo, n)
    return y / (float(np.max(np.abs(y))) or 1.0)


def karplus(n, freq, damp=0.996, tone=0.5, seed=0):
    """Plucked string. Processed one delay-line at a time: within a block of
    D samples every value it reads was written by the previous block, so the
    recursion vectorises into n/D numpy steps instead of n Python ones."""
    d = max(2, int(RATE / max(float(freq), 20.0)))
    rng = np.random.default_rng(seed)
    seed_buf = rng.uniform(-1.0, 1.0, d + 1)
    out = np.empty(n + d + 1)
    out[:d + 1] = seed_buf
    mix = float(np.clip(tone, 0.05, 0.95))
    i = d + 1
    while i < n + d + 1:
        blk = min(d, n + d + 1 - i)
        win = out[i - d - 1:i - d + blk]
        out[i:i + blk] = damp * (mix * win[1:blk + 1] + (1.0 - mix) * win[:blk])
        i += blk
    return out[:n]


# ------------------------------------------------------------------ filters

def _min_phase(mag):
    """Turn a magnitude response into a minimum-phase one via the real
    cepstrum. Without this, every filter smears the attack backwards."""
    n = (len(mag) - 1) * 2
    cep = np.fft.irfft(np.log(np.maximum(mag, 1e-9)), n)
    w = np.zeros(n)
    w[0] = 1.0
    w[1:n // 2] = 2.0
    w[n // 2] = 1.0
    return np.exp(np.fft.rfft(cep * w, n))


def _response(f, kind, cut, res, poles):
    cut = float(np.clip(cut, 20.0, NYQ * 0.98))
    r = np.maximum(f, 1e-6) / cut
    if kind == "hp":
        h = r ** poles / np.sqrt(1.0 + r ** (2 * poles))
    elif kind == "bp":
        lo = 1.0 / np.sqrt(1.0 + (r * 2.0) ** (2 * poles))
        hi = (r * 0.5) ** poles / np.sqrt(1.0 + (r * 0.5) ** (2 * poles))
        h = lo * hi
        h = h / (float(np.max(h)) or 1.0)
    else:
        h = 1.0 / np.sqrt(1.0 + r ** (2 * poles))
    if res > 0.01:
        h = h * (1.0 + res * np.exp(-(np.log(np.maximum(r, 1e-6)) ** 2) / 0.08))
    return h


def _fast_len(n):
    """Smallest 5-smooth integer >= n. numpy's FFT is fast on these, and
    padding a 6-second texture to the next power of two instead costs the
    best part of a second per layer."""
    if n <= 8:
        return 8
    best = 1 << (n - 1).bit_length()
    p3 = 1
    while p3 < best:
        p35 = p3
        while p35 < best:
            v = p35
            while v < n:
                v *= 2
            if v < best:
                best = v
            p35 *= 5
        p3 *= 3
    return best


def _pad_for(n, cut):
    """Room for the filter's own impulse response, so the tail does not wrap
    around onto the head. Low cutoffs ring for longer, so they need more."""
    return _fast_len(n + int(np.clip(6.0 * RATE / max(cut, 20.0),
                                     2048, 32768)))


_MP_SIZE = 8192
_MP_CACHE = {}


def _mp_response(freqs, kind, cut, res, poles):
    """Minimum-phase response sampled onto `freqs`.

    Derived once at a fixed 8192-point resolution and interpolated, rather
    than at the length of whatever is being filtered. The cepstrum costs two
    more transforms per filter, and at full length that doubled the price of
    every layer; these filters are smooth and their group delay is a handful
    of samples, so the interpolation is inaudible and the cost stops scaling
    with duration entirely.
    """
    key = (kind, round(float(cut), 2), round(float(res), 3), int(poles))
    h = _MP_CACHE.get(key)
    if h is None:
        base = np.fft.rfftfreq(_MP_SIZE, 1.0 / RATE)
        h = (base, _min_phase(_response(base, kind, cut, res, poles)))
        if len(_MP_CACHE) > 512:
            _MP_CACHE.pop(next(iter(_MP_CACHE)))
        _MP_CACHE[key] = h
    base, hv = h
    if len(freqs) == len(base):
        return hv
    return (np.interp(freqs, base, hv.real) +
            1j * np.interp(freqs, base, hv.imag))


def filt(x, kind="lp", cut=18000.0, res=0.0, poles=2):
    """Static minimum-phase filter."""
    if kind in (None, "none") or len(x) < 8:
        return x
    n = _pad_for(len(x), cut)
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    h = _mp_response(freqs, kind, cut, res, poles)
    return np.fft.irfft(np.fft.rfft(x, n) * h, n)[:len(x)]


def bank_filter(x, kind, cut_env, res=0.0, poles=2, steps=0):
    """Filter with an arbitrary per-sample cutoff, as a crossfade between
    statically-filtered copies.

    Taking the cutoff as an array rather than two endpoints is what lets a
    texture's filter wander under an LFO instead of only sliding one way.
    """
    if kind in (None, "none") or len(x) < 8:
        return x
    n = len(x)
    cut_env = np.clip(np.asarray(cut_env, dtype=float), 20.0, NYQ * 0.98)
    lo_c, hi_c = float(cut_env.min()), float(cut_env.max())
    if np.log2(hi_c / lo_c) < 0.12:
        return filt(x, kind, float(np.sqrt(lo_c * hi_c)), res, poles)
    if not steps:
        # Every step is a full FFT pair over the layer, so a one-second pad
        # would cost nine of them. Long sounds move slowly in absolute terms
        # and hide the coarser grid; short ones need the resolution.
        steps = 9 if n < 0.3 * RATE else (7 if n < 0.8 * RATE else 5)
        # A wandering cutoff crosses the grid repeatedly, so it needs more
        # rungs than a single slide over the same span.
        if np.log2(hi_c / lo_c) > 3.0:
            steps += 2
    grid = np.geomspace(lo_c, hi_c, steps)
    # One forward transform for the whole bank. Calling filt() per rung
    # would re-transform the same input `steps` times, which was most of the
    # cost of every swept layer.
    size = _pad_for(n, lo_c)
    spec = np.fft.rfft(x, size)
    freqs = np.fft.rfftfreq(size, 1.0 / RATE)
    bank = np.stack([
        np.fft.irfft(spec * _mp_response(freqs, kind, c, res, poles),
                     size)[:n] for c in grid])
    pos = np.interp(np.log(cut_env), np.log(grid), np.arange(steps))
    lo = np.clip(pos.astype(int), 0, steps - 2)
    frac = pos - lo
    idx = np.arange(n)
    return bank[lo, idx] * (1.0 - frac) + bank[lo + 1, idx] * frac


def sweep(x, kind, cut0, cut1, res=0.0, poles=2, steps=0):
    """Time-varying filter that slides once, from cut0 to cut1."""
    if kind in (None, "none") or len(x) < 8:
        return x
    cut0, cut1 = max(float(cut0), 20.0), max(float(cut1), 20.0)
    if abs(np.log2(cut1 / cut0)) < 0.12:
        return filt(x, kind, cut0, res, poles)
    return bank_filter(x, kind, np.geomspace(cut0, cut1, len(x)),
                       res, poles, steps)


# --------------------------------------------------------------- modulation

def lfo(n, rate_hz, shape="random", seed=0):
    """A 0..1 control signal.

    "random" is the one that matters: wind gusts, a flickering fire and
    rain intensity are all smoothly-wandering values, and a sine LFO
    applied to any of them sounds immediately like a machine. It is value
    noise - random points at `rate_hz`, joined with a smoothstep so there
    are no corners to click on.
    """
    if n < 2:
        return np.ones(max(n, 0))
    t = np.linspace(0.0, n / RATE, n)
    if shape == "sine":
        return 0.5 + 0.5 * np.sin(2.0 * np.pi * rate_hz * t)
    if shape == "tri":
        p = (rate_hz * t) % 1.0
        return 1.0 - 2.0 * np.abs(p - 0.5)
    rng = np.random.default_rng(seed)
    k = max(2, int(rate_hz * n / RATE) + 2)
    pts = rng.random(k)
    pos = np.linspace(0.0, k - 1.0, n)
    i = np.clip(pos.astype(int), 0, k - 2)
    f = pos - i
    f = f * f * (3.0 - 2.0 * f)                 # smoothstep, no corners
    return pts[i] * (1.0 - f) + pts[i + 1] * f


def grains(n, density, grain_ms, f_lo, f_hi, decay=7.0, bands=4, seed=0,
           amp_var=0.75, pitched=False):
    """A cloud of short bursts - rain, fire crackle, sizzle, gravel.

    Built as a sparse impulse train convolved with a few grain kernels
    rather than a loop over events: one FFT per band renders ten thousand
    grains in the same time as ten, which is what makes dense rain cheap.
    Several bands stop every drop from being audibly the same drop.
    """
    if n < 16:
        return np.zeros(max(n, 0))
    rng = np.random.default_rng(seed)
    count = max(1, int(density * n / RATE))
    out = np.zeros(n)
    glen = max(8, int(grain_ms * RATE / 1000.0))
    centres = np.geomspace(max(f_lo, 30.0), max(f_hi, f_lo + 1.0), max(bands, 1))

    for b, fc in enumerate(centres):
        share = count // max(bands, 1) + (1 if b < count % max(bands, 1) else 0)
        if share <= 0:
            continue
        impulses = np.zeros(n)
        at = rng.integers(0, n, share)
        amps = rng.uniform(1.0 - amp_var, 1.0, share) * rng.choice(
            [-1.0, 1.0], share)
        np.add.at(impulses, at, amps)
        env = np.exp(-decay * np.linspace(0, 1, glen))
        if pitched:
            # A ping instead of a tick - bubbles, chimes, dripping water.
            ph = np.cumsum(2.0 * np.pi * fc / RATE * np.ones(glen))
            kern = np.sin(ph) * env
        else:
            kern = rng.standard_normal(glen) * env
            kern = filt(kern, "bp", fc, 1.2, 2)
        kern[:4] *= np.linspace(0, 1, 4)
        size = 1 << (n + glen).bit_length()
        out += np.fft.irfft(np.fft.rfft(impulses, size) *
                            np.fft.rfft(kern, size), size)[:n]
    return out / (float(np.max(np.abs(out))) or 1.0)


# ------------------------------------------------------------------ effects

# ------------------------------------------------- excitation and resonance
# The part that makes a sound identifiable as a *thing* rather than a tone.
# An object has resonant modes; something happens to it; you hear the modes
# ringing down. Both halves matter and they are independent, which is why
# tapping a glass and scraping a glass are obviously the same glass.


def modal(exciter, freqs, amps, taus, n_out=None):
    """Ring an excitation signal through a bank of resonant modes.

    Each mode is a two-pole resonator at `freq` decaying to 1/e in `tau`
    seconds. They are summed as complex transfer functions and applied in
    one FFT multiply, so twenty modes cost about what one filter costs -
    and, being linear and time-invariant, the result is exact rather than
    an approximation of a filter bank.

    The decay times are the whole game. Modes at the same frequencies with
    the wrong taus sound like a synthesiser; get the taus right and the
    same frequencies sound like wood, or glass, or steel.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    amps = np.atleast_1d(np.asarray(amps, dtype=np.float64))
    taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
    n_out = int(n_out or len(exciter))
    keep = (freqs > 12.0) & (freqs < NYQ * 0.96) & (amps > 1e-4)
    if not np.any(keep):
        return np.zeros(n_out)
    freqs, amps, taus = freqs[keep], amps[keep], np.maximum(taus[keep], 2e-4)

    n = _fast_len(n_out + 64)
    x = np.zeros(n)
    x[:min(len(exciter), n)] = exciter[:n]
    spec = np.fft.rfft(x)
    w = 2.0 * np.pi * np.fft.rfftfreq(n, 1.0 / RATE) / RATE
    e1 = np.exp(-1j * w)
    e2 = e1 * e1

    r = np.exp(-1.0 / (taus * RATE))
    th = 2.0 * np.pi * freqs / RATE
    h = np.zeros(len(w), dtype=np.complex128)
    # In blocks: the full (modes x bins) matrix is large for a long ring,
    # and there is no reason to hold it all at once.
    for i in range(0, len(freqs), 8):
        rr = r[i:i + 8, None]
        cc = np.cos(th[i:i + 8])[:, None]
        hk = 1.0 / (1.0 - 2.0 * rr * cc * e1[None, :] + rr * rr * e2[None, :])
        peak = np.abs(hk).max(axis=1, keepdims=True)
        h += (amps[i:i + 8, None] * hk / np.maximum(peak, 1e-12)).sum(axis=0)

    y = np.fft.irfft(spec * h, n)[:n_out]
    return y / (float(np.max(np.abs(y))) or 1.0)


def strike(n, hardness=0.5, seed=0, noisy=0.25):
    """The hit itself: a short contact pulse.

    Contact time is what "hard" means physically - a steel tip touches for
    under a millisecond and excites every mode, a felt mallet lingers and
    excites only the low ones. Modelling that, rather than filtering the
    result afterwards, is why a soft strike here sounds soft rather than
    muffled.
    """
    hardness = float(np.clip(hardness, 0.0, 1.0))
    contact = max(2, int(RATE * (0.0002 + (1.0 - hardness) ** 2 * 0.011)))
    t = np.linspace(0.0, np.pi, contact)
    pulse = np.sin(t) ** 2
    if noisy > 0.01:
        rng = np.random.default_rng(seed)
        pulse = pulse * (1.0 - noisy + noisy * rng.uniform(0.4, 1.6, contact))
    out = np.zeros(max(n, contact))
    out[:contact] = pulse
    return out[:n] if n >= contact else out[:n]


def scrape(n, roughness=0.5, speed=0.5, seed=0):
    """Sustained contact: a rough surface dragged across another.

    Not white noise - a scrape is a dense stream of micro-impacts, so its
    energy comes in irregular clumps. Flat noise through a resonator sounds
    like a filter being opened, not like something being dragged.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    lumps = lfo(n, 6.0 + 40.0 * speed, "random", seed + 5)
    grit = lfo(n, 60.0 + 300.0 * speed, "random", seed + 9)
    return x * (0.25 + 0.75 * lumps) * (1.0 - roughness + roughness * 1.6 * grit)


def bubble(n, f0=800.0, rise=2.6, decay=28.0):
    """A water drop.

    A bubble in liquid rings at a frequency that climbs as it collapses -
    the one sound whose pitch envelope is genuinely exponential upward.
    Every "water drop" that sounds fake is a downward or flat one.
    """
    t = np.arange(n, dtype=np.float64) / RATE
    f = f0 * np.exp(rise * t * decay / 12.0)
    f = np.minimum(f, NYQ * 0.9)
    return np.sin(np.cumsum(2.0 * np.pi * f / RATE)) * np.exp(-decay * t)


def trim_tail(x, floor_db=-62.0, pad_ms=12.0):
    """Cut the inaudible end off a decay.

    A reverb approaches zero forever, so a 1.2s chime is often 400ms of
    sound and 800ms of air that still costs full price in a WAV. -62dB is
    below anything audible under a UI, and the short fade afterwards keeps
    the new last sample from being a step.

    Lives here rather than in the exporter because reversing a sound needs
    it too: the quiet half of a cue is its tail, and reversing an untrimmed
    one puts all that air in front of the sound.
    """
    a = np.asarray(x)
    mono = a.mean(axis=1) if a.ndim > 1 else a
    thresh = 10.0 ** (floor_db / 20.0) * max(float(np.max(np.abs(mono))), 1e-9)
    loud = np.flatnonzero(np.abs(mono) > thresh)
    if len(loud) == 0:
        return a
    end = min(len(mono), int(loud[-1]) + int(pad_ms * RATE / 1000.0))
    if end >= len(mono) - 2:
        return a
    out = np.array(a[:end])
    fade = min(160, len(out) // 4)
    if fade > 1:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(1, 0, fade)))
        out[-fade:] *= ramp[:, None] if out.ndim > 1 else ramp
    return out


def drive(x, amount):
    if amount <= 0.001:
        return x
    g = 1.0 + amount * 14.0
    return np.tanh(x * g) / np.tanh(g) * (1.0 - 0.35 * amount)


def bitcrush(x, bits=0, rate=0):
    y = x
    if rate and rate < RATE:
        step = max(1, int(RATE / rate))
        idx = np.clip((np.arange(len(y)) // step) * step, 0, len(y) - 1)
        y = y[idx]
    if bits and bits < 16:
        q = float(2 ** (max(int(bits), 1) - 1))
        y = np.round(y * q) / q
    return y


def echo(x, time_s, feedback, mix):
    if mix <= 0.001 or time_s <= 0.001:
        return x
    d = max(1, int(time_s * RATE))
    y = np.concatenate([x, np.zeros(d * 4)])
    fb = float(np.clip(feedback, 0.0, 0.85))
    i = d
    while i < len(y):
        blk = min(d, len(y) - i)
        y[i:i + blk] += fb * y[i - d:i - d + blk]
        i += blk
    dry = np.concatenate([x, np.zeros(len(y) - len(x))])
    return dry * (1.0 - mix * 0.4) + y * mix


_IR_CACHE = {}


def _impulse(size, damp, seed=7):
    key = (round(float(size), 3), round(float(damp), 3), seed)
    if key not in _IR_CACHE:
        n = max(512, int(RATE * (0.12 + size * 1.6)))
        rng = np.random.default_rng(seed)
        ir = rng.standard_normal(n) * np.exp(
            -np.linspace(0, 1, n) * (7.0 - size * 4.5))
        # A short pre-delay ramp keeps the tail from sounding like a noise
        # burst glued to the end of the sound.
        pre = max(2, int(RATE * 0.008 * (0.4 + size)))
        ir[:pre] *= np.linspace(0, 1, pre) ** 2
        ir = filt(ir, "lp", 1200.0 + (1.0 - damp) * 12000.0, 0.0, 1)
        ir = filt(ir, "hp", 180.0, 0.0, 1)
        ir = ir / (float(np.sqrt(np.sum(ir ** 2))) or 1.0)
        if len(_IR_CACHE) > 32:
            _IR_CACHE.pop(next(iter(_IR_CACHE)))
        _IR_CACHE[key] = ir
    return _IR_CACHE[key]


def reverb(x, mix, size=0.3, damp=0.5):
    if mix <= 0.001:
        return x
    ir = _impulse(size, damp)
    n = 1 << (len(x) + len(ir)).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[:len(x) + len(ir)]
    dry = np.concatenate([x, np.zeros(len(wet) - len(x))])
    wet = wet * ((float(np.max(np.abs(dry))) or 1.0) /
                 (float(np.max(np.abs(wet))) or 1.0))
    return dry * (1.0 - mix * 0.35) + wet * mix
