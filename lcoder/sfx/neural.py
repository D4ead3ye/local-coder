"""Text-to-audio generation with Stable Audio Open, running locally.

The model does the part the procedural engine could not: it understands the
prompt. What it does *not* do is produce a UI cue - it returns a few
seconds of 44.1kHz stereo that happens to contain the thing you asked for,
usually after some lead-in and with a tail that runs past the interesting
part. So generation is only the first half:

    prompt -> model -> onset-aligned clip -> the existing recipe pipeline

which is why a generated sound still arrives as a `sample` layer inside a
Recipe. Role levels, trimming, fades, reverse/retract, the master chain and
the zero-asset Python pack all keep working on top of model output, because
none of them ever cared where the samples came from.

Everything here is lazy. Importing this module must stay cheap and must not
fail on a machine with no torch, so the app can fall back to the procedural
engine without knowing anything about why.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from . import engine as E

MODEL_ID = "stabilityai/stable-audio-open-1.0"
CACHE = Path.home() / "Documents" / "SFX Forge" / "clips"

# Generated a little long, then cut down. The model needs room to state the
# sound; asking it directly for 150ms mostly returns silence with a hint of
# something at the end.
GEN_SECONDS = 3.0
STEPS = 50

# Batching is off, and that is a measured decision rather than caution.
# On an RTX 5070 (12GB), per-sound cost *rises* with batch size - 7.5s at
# batch 1, 8.7s at 2, 9.1s at 3 - because one sequence already saturates
# the GPU, while peak VRAM climbs 6.0 -> 9.3 -> 12.5GB and batch 6 walked
# straight into a CUDA *sticky* error, which is not a clean out-of-memory:
# it kills the context for the whole process, so every later generation
# fails until the app is restarted. Batching cost speed and bought risk.
#
# Length is nearly free by comparison (3.0s costs the same as 2.0s at the
# same step count); only steps scale the time, roughly linearly.
MAX_BATCH = 1

# The model has never been asked for an interface cue by anyone who wrote
# its training captions, so the prompt is framed for it.
FRAME = ("%s, single isolated sound effect, clean studio recording, "
         "close mic, no music, no speech, silence before and after")
NEGATIVE = ("music, melody, song, speech, voice, talking, noise floor, hum, "
            "reverb tail, low quality, distortion, clipping")

_lock = threading.Lock()
_pipe = None
_state = {"status": "cold", "detail": "", "device": ""}


def status():
    return dict(_state)


def torch_ready():
    """Is the environment capable of running the model at all?"""
    try:
        import torch                                  # noqa: F401
        import diffusers                              # noqa: F401
    except Exception as exc:                          # noqa: BLE001
        _state["detail"] = "%s" % exc
        return False
    return True


def weights_ready():
    """Have the weights already been fetched? Distinguishes 'not set up'
    from 'about to download five gigabytes without warning you'."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, local_files_only=True,
                          allow_patterns=["model_index.json"])
        return True
    except Exception:                                 # noqa: BLE001
        return False


def load(progress=None):
    """Bring the pipeline up. Slow, blocking, and safe to call twice."""
    global _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        _state["status"] = "loading"
        if progress:
            progress("importing torch")
        import torch
        from diffusers import StableAudioPipeline

        cuda = torch.cuda.is_available()
        _state["device"] = (torch.cuda.get_device_name(0) if cuda else "cpu")
        if progress:
            progress("loading weights on %s" % _state["device"])
        dt = torch.float16 if cuda else torch.float32
        try:
            pipe = StableAudioPipeline.from_pretrained(MODEL_ID, dtype=dt)
        except TypeError:
            # diffusers renamed this; older versions only know torch_dtype.
            pipe = StableAudioPipeline.from_pretrained(MODEL_ID,
                                                       torch_dtype=dt)
        pipe = pipe.to("cuda" if cuda else "cpu")
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe
        _state["status"] = "ready"
        return _pipe


def clip_id(prompt, seed, seconds=GEN_SECONDS, steps=STEPS):
    key = json.dumps([prompt, int(seed), round(float(seconds), 2), int(steps)],
                     sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def cached(cid):
    """Load a clip from disk, or None. Generations are expensive enough
    that re-hearing one should never cost a second time."""
    path = CACHE / (cid + ".npy")
    if path.exists():
        try:
            return np.load(path)
        except Exception:                             # noqa: BLE001
            return None
    return None


def _store(cid, audio):
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / (cid + ".npy"), audio.astype(np.float32))


def onset_align(audio, floor_db=-38.0, pre_ms=6.0):
    """Cut to where the sound actually starts.

    The model reliably puts a beat of silence in front of the event, and a
    UI cue that begins 400ms after the click reads as lag no matter how
    good it sounds. The threshold is deliberately well above the noise
    floor - it is looking for the event, not for the first non-zero sample.
    """
    a = np.asarray(audio, dtype=np.float32)
    mono = a.mean(axis=1) if a.ndim > 1 else a
    peak = float(np.max(np.abs(mono))) or 1.0
    thr = peak * 10.0 ** (floor_db / 20.0)
    loud = np.flatnonzero(np.abs(mono) > thr)
    if not len(loud):
        return a
    start = max(0, int(loud[0]) - int(pre_ms * E.RATE / 1000.0))
    out = a[start:]
    # A hard cut at the onset would clip the attack; a couple of ms of
    # raised cosine keeps the transient without a click in front of it.
    fade = min(96, len(out) // 8)
    if fade > 1:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, fade)))
        out = np.array(out)
        out[:fade] *= ramp[:, None] if out.ndim > 1 else ramp
    return out


def _run(pipe, texts, seconds, steps, seed):
    """One diffusion run over a small batch of prompts."""
    import torch
    gen = torch.Generator(device=pipe.device.type).manual_seed(int(seed))
    with torch.inference_mode():
        out = pipe(texts, negative_prompt=[NEGATIVE] * len(texts),
                   num_inference_steps=int(steps),
                   audio_end_in_s=float(seconds),
                   num_waveforms_per_prompt=1, generator=gen)
    clips = [_finish(a.float().cpu().numpy()) for a in out.audios]
    try:
        torch.cuda.empty_cache()
    except Exception:                                 # noqa: BLE001
        pass
    return clips


def _finish(raw):
    """One model output -> the (n, 2) float32 clip we keep."""
    audio = np.asarray(raw, dtype=np.float32)
    if audio.ndim == 2:                               # (channels, samples)
        audio = audio.T
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    peak = float(np.max(np.abs(audio))) or 1.0
    return onset_align((audio / peak * 0.98).astype(np.float32))


def generate_many(prompt, seed=0, count=1, seconds=GEN_SECONDS, steps=STEPS,
                  frame=True, progress=None, on_clip=None):
    """Several variations of one prompt.

    Anything already on disk is served from there and only the misses are
    generated. `on_clip(index, id, audio)` fires as each one lands, so the
    caller can show the first sound after seven seconds instead of after
    the whole row.
    """
    ids = [clip_id(prompt, seed + i, seconds, steps) for i in range(count)]
    have = {cid: cached(cid) for cid in ids}
    missing = [i for i, cid in enumerate(ids) if have[cid] is None]
    if on_clip:
        for i, cid in enumerate(ids):
            if have[cid] is not None:
                on_clip(i, cid, have[cid])
    if not missing:
        return list(zip(ids, (have[c] for c in ids)))

    pipe = load(progress)
    text = FRAME % prompt if frame else prompt
    done = 0
    for k in range(0, len(missing), MAX_BATCH):
        chunk = missing[k:k + MAX_BATCH]
        if progress:
            progress("generating %d/%d" % (done + 1, len(missing)))
        for slot, clip in zip(chunk, _run(pipe, [text] * len(chunk), seconds,
                                          steps, seed + chunk[0])):
            have[ids[slot]] = clip
            _store(ids[slot], clip)
            if on_clip:
                on_clip(slot, ids[slot], clip)
        done += len(chunk)
    return list(zip(ids, (have[c] for c in ids)))


# What each cue is, said in the model's language rather than the app's.
ROLE_HINT = {
    "hover": "very short quiet subtle tick, barely there",
    "click": "short crisp click, one hit",
    "toast": "notification appearing, gentle chime",
    "toast_out": "notification dismissing, soft descending sound",
    "ok": "positive confirmation chime, pleasant",
    "warn": "warning alert tone, attention",
    "error": "negative error sound, low and blunt",
}


def generate_prompts(prompts, seed=0, seconds=GEN_SECONDS, steps=STEPS,
                     frame=True, progress=None):
    """One clip per prompt, all in a single diffusion run.

    diffusers batches a list of prompts, so a whole matched set costs one
    run instead of seven. Cached prompts drop out of the batch first.
    """
    ids = [clip_id(p, seed, seconds, steps) for p in prompts]
    have = {cid: cached(cid) for cid in ids}
    todo = [(i, p) for i, p in enumerate(prompts) if have[ids[i]] is None]
    if not todo:
        return list(zip(ids, (have[c] for c in ids)))

    pipe = load(progress)
    done = 0
    for k in range(0, len(todo), MAX_BATCH):
        chunk = todo[k:k + MAX_BATCH]
        if progress:
            progress("generating cue %d/%d" % (done + 1, len(todo)))
        texts = [FRAME % p if frame else p for _, p in chunk]
        for (slot, _), clip in zip(chunk, _run(pipe, texts, seconds, steps,
                                               seed + k)):
            have[ids[slot]] = clip
            _store(ids[slot], clip)
        done += len(chunk)
    return list(zip(ids, (have[c] for c in ids)))


def generate(prompt, seed=0, seconds=GEN_SECONDS, steps=STEPS, frame=True,
             progress=None):
    """Prompt -> (id, (n, 2) float32) at 44.1kHz, onset-aligned, cached."""
    return generate_many(prompt, seed, 1, seconds, steps, frame, progress)[0]


def unload():
    """Give the VRAM back. The app is a UI tool, not a batch job, and
    holding 5GB while somebody edits a filter is rude."""
    global _pipe
    with _lock:
        _pipe = None
        _state["status"] = "cold"
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:                                 # noqa: BLE001
        pass
