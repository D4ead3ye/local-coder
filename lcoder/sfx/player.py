"""Audio out, and rendering that never happens on the UI thread.

winsound is the whole audio backend. It sounds like a cop-out and is not:
it is in the stdlib, it needs no device setup, it plays WAV bytes straight
from memory, and it therefore adds nothing to the frozen exe. The cost is
that it refuses SND_MEMORY together with SND_ASYNC, so playback has to be
synchronous - which is fine on a worker thread and fatal on the UI one.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from . import export
from .recipe import render


class Player:
    """Queued playback. Never blocks the caller, never raises at it."""

    def __init__(self):
        self._q = queue.Queue(maxsize=3)
        self._thread = None
        self.error = None
        self.volume = 1.0

    def _worker(self):
        import winsound
        while True:
            data = self._q.get()
            if data is None:
                return
            try:
                winsound.PlaySound(data, winsound.SND_MEMORY |
                                   winsound.SND_NODEFAULT)
            except Exception as exc:                     # noqa: BLE001
                self.error = repr(exc)

    def play(self, audio):
        """`audio` is a float array or ready-made WAV bytes."""
        try:
            data = audio if isinstance(audio, (bytes, bytearray)) else \
                export.wav_bytes(_scaled(audio, self.volume))
        except Exception as exc:                         # noqa: BLE001
            self.error = repr(exc)
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker,
                                            name="sfxforge-audio", daemon=True)
            self._thread.start()
        try:
            self._q.put_nowait(data)
        except queue.Full:
            pass          # already playing; dropping beats a growing backlog

    def stop(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:                                # noqa: BLE001
            pass


def _scaled(audio, volume):
    return audio if abs(volume - 1.0) < 0.001 else audio * volume


class Renderer:
    """Renders recipes off the UI thread and caches by parameter identity.

    The cache key is the recipe dict, not the object, so dragging a slider
    back to where it was is free - which matters, because that is exactly
    what auditioning a parameter looks like.
    """

    def __init__(self, workers=4, cache_size=96):
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="sfxforge-render")
        self._cache = {}
        self._pending = {}
        self._lock = threading.Lock()
        self.cache_size = cache_size

    @staticmethod
    def key(recipe):
        d = recipe.to_dict()
        d.pop("name", None)
        d.pop("prompt", None)
        return repr(sorted(d.items(), key=lambda kv: kv[0]))

    def cached(self, recipe):
        return self._cache.get(self.key(recipe))

    def get(self, recipe):
        """Blocking render, cached. Use for one short sound on demand."""
        k = self.key(recipe)
        hit = self._cache.get(k)
        if hit is None:
            hit = render(recipe)
            self._store(k, hit)
        return hit

    def request(self, recipe, done=None):
        """Non-blocking. `done(audio)` fires on a worker thread, so treat it
        as a place to stash a result - not to touch ImGui state."""
        k = self.key(recipe)
        hit = self._cache.get(k)
        if hit is not None:
            if done:
                done(hit)
            return hit
        with self._lock:
            if k in self._pending:
                return None
            self._pending[k] = True

        def job():
            try:
                audio = render(recipe)
                self._store(k, audio)
                if done:
                    done(audio)
            finally:
                with self._lock:
                    self._pending.pop(k, None)

        self._pool.submit(job)
        return None

    def busy(self):
        with self._lock:
            return len(self._pending)

    def _store(self, k, audio):
        with self._lock:
            self._cache[k] = audio
            while len(self._cache) > self.cache_size:
                self._cache.pop(next(iter(self._cache)))
