"""Neural image server — loads a diffusion pipeline once and renders on request.

Runs inside the project's CUDA venv (ai-venv), the same one as the audio
server, and speaks the same tiny stdlib HTTP API so the torch-free main app
can drive it. Models are several GB and slow to bring up, so the pipeline
loads once on first use and stays resident.

    python image_server.py --port 8140 --model sdxl

Default model is SDXL base 1.0 (1024px, fits a 12 GB card in fp16). The MODELS
table is where more slot in — a turbo/fast variant, Flux, a community
checkpoint — exactly like the audio server's two modes.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# key -> (repo id, pipeline-class name, default steps, default guidance, native px)
MODELS = {
    "sdxl":       ("stabilityai/stable-diffusion-xl-base-1.0", "StableDiffusionXLPipeline", 30, 7.0, 1024),
    "sdxl-turbo": ("stabilityai/sdxl-turbo",                   "AutoPipelineForText2Image", 4,  0.0, 512),
    "sd15":       ("runwayml/stable-diffusion-v1-5",           "AutoPipelineForText2Image", 28, 7.5, 512),
}
DEFAULT_MODEL = "sdxl"
NEG = ("lowres, bad anatomy, bad hands, text, watermark, signature, jpeg "
       "artifacts, blurry, deformed, extra limbs, worst quality, low quality")

_pipe = None
_pipe_key = None
_lock = threading.Lock()
_state = {"status": "cold", "device": "", "model": DEFAULT_MODEL, "detail": ""}


def _load_pipe(key):
    import torch
    import diffusers
    repo, cls_name, _, _, _ = MODELS[key]
    cuda = torch.cuda.is_available()
    _state["device"] = torch.cuda.get_device_name(0) if cuda else "cpu"
    _state["detail"] = f"carregando {repo}"
    dt = torch.float16 if cuda else torch.float32
    cls = getattr(diffusers, cls_name)
    try:
        pipe = cls.from_pretrained(repo, dtype=dt, use_safetensors=True,
                                   variant="fp16" if cuda else None)
    except Exception:
        # some repos ship no fp16 variant files; fall back to default weights
        pipe = cls.from_pretrained(repo, dtype=dt, use_safetensors=True)
    if cuda:
        pipe = pipe.to("cuda")
        # VAE tiling gives headroom for >1024px on a 12 GB card; the method
        # moved onto the VAE in newer diffusers, so try both spellings.
        for fn in (getattr(getattr(pipe, "vae", None), "enable_tiling", None),
                   getattr(pipe, "enable_vae_tiling", None)):
            if callable(fn):
                try:
                    fn(); break
                except Exception:
                    pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        pipe = pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load(key=DEFAULT_MODEL):
    global _pipe, _pipe_key
    with _lock:
        if _pipe is not None and _pipe_key == key:
            return _pipe
        if _pipe is not None and _pipe_key != key:
            _pipe = None                   # free the old one before swapping
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        _state["status"] = "loading"
        _state["model"] = key
        _pipe = _load_pipe(key)
        _pipe_key = key
        _state["status"] = "ready"
        _state["detail"] = ""
        return _pipe


def generate(prompt, model, negative, steps, guidance, width, height, seed, out_path):
    import torch
    key = model if model in MODELS else DEFAULT_MODEL
    _, _, d_steps, d_guid, native = MODELS[key]
    pipe = load(key)
    _state["status"] = "generating"
    try:
        w = int(width or native); h = int(height or native)
        w -= w % 8; h -= h % 8
        gen = torch.Generator(device=pipe.device.type).manual_seed(int(seed))
        kw = dict(prompt=prompt,
                  num_inference_steps=int(steps or d_steps),
                  guidance_scale=float(guidance if guidance is not None else d_guid),
                  width=w, height=h, generator=gen)
        if key != "sdxl-turbo":            # turbo is trained with no negative/CFG
            kw["negative_prompt"] = negative or NEG
        with torch.inference_mode():
            img = pipe(**kw).images[0]
        img.save(out_path, format="PNG")
        return {"path": out_path, "width": w, "height": h, "model": key,
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
            self._send(200, dict(_state, models=list(MODELS)))
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
            res = generate(req["prompt"], req.get("model", DEFAULT_MODEL),
                           req.get("negative", ""), req.get("steps"),
                           req.get("guidance"), req.get("width"),
                           req.get("height"), req.get("seed", 0), req["out"])
            self._send(200, res)
        except Exception as e:
            _state["status"] = "error"
            _state["detail"] = f"{type(e).__name__}: {e}"
            self._send(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--preload", action="store_true")
    a = ap.parse_args()
    if a.preload:
        threading.Thread(target=lambda: load(a.model), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[image] listening on 127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
