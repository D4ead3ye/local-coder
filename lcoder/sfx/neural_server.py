"""Neural audio server — loads Stable Audio Open once and generates on request.

Runs inside the project's CUDA venv (ai-venv), where torch/diffusers live, and
speaks a tiny stdlib HTTP API so the main app (a different Python, no torch)
can drive it exactly like it drives llama-server. The model is ~4 GB and slow
to bring up, so it loads once on the first request and stays resident.

    python neural_server.py --port 8130

Stable Audio Open 1.0 is text-to-audio up to ~47 s: short, framed prompts give
clean isolated SFX; longer, plain prompts give music/loops. Hence two modes.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "stabilityai/stable-audio-open-1.0"
SR = 44100

FRAME_SFX = ("%s, single isolated sound effect, clean studio recording, close "
             "mic, no music, no speech, silence before and after")
NEG_SFX = ("music, melody, song, speech, voice, talking, noise floor, hum, "
           "reverb tail, low quality, distortion, clipping")
NEG_MUSIC = "low quality, distortion, clipping, harsh noise"

_pipe = None
_lock = threading.Lock()
_state = {"status": "cold", "device": "", "detail": ""}


def load():
    global _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        _state["status"] = "loading"
        import torch
        from diffusers import StableAudioPipeline
        cuda = torch.cuda.is_available()
        _state["device"] = torch.cuda.get_device_name(0) if cuda else "cpu"
        _state["detail"] = "carregando pesos em " + _state["device"]
        dt = torch.float16 if cuda else torch.float32
        try:
            pipe = StableAudioPipeline.from_pretrained(MODEL_ID, dtype=dt)
        except TypeError:
            pipe = StableAudioPipeline.from_pretrained(MODEL_ID, torch_dtype=dt)
        pipe = pipe.to("cuda" if cuda else "cpu")
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe
        _state["status"] = "ready"
        _state["detail"] = ""
        return _pipe


def generate(prompt, mode, seconds, steps, seed, out_path):
    import numpy as np
    import soundfile as sf
    import torch
    pipe = load()
    text = prompt if mode == "music" else FRAME_SFX % prompt
    neg = NEG_MUSIC if mode == "music" else NEG_SFX
    _state["status"] = "generating"
    try:
        gen = torch.Generator(device=pipe.device.type).manual_seed(int(seed))
        with torch.inference_mode():
            out = pipe([text], negative_prompt=[neg],
                       num_inference_steps=int(steps),
                       audio_end_in_s=float(seconds),
                       num_waveforms_per_prompt=1, generator=gen)
        a = out.audios[0].float().cpu().numpy()
        if a.ndim == 2:
            a = a.T
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        peak = float(np.max(np.abs(a))) or 1.0
        a = (a / peak * 0.98).astype(np.float32)
        sf.write(out_path, a, SR, subtype="PCM_16")   # stdlib wave can read this
        return {"path": out_path, "seconds": round(len(a) / SR, 2), "sr": SR,
                "device": _state["device"]}
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        _state["status"] = "ready"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            self._send(200, dict(_state))
        elif self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            res = generate(req["prompt"], req.get("mode", "sfx"),
                           req.get("seconds", 3.0), req.get("steps", 50),
                           req.get("seed", 0), req["out"])
            self._send(200, res)
        except Exception as e:
            _state["status"] = "error"
            _state["detail"] = f"{type(e).__name__}: {e}"
            self._send(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--preload", action="store_true")
    a = ap.parse_args()
    if a.preload:
        threading.Thread(target=load, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[neural] listening on 127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
