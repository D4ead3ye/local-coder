"""The agent's hands: files, search, shell, background processes, the web.

Every tool takes the parsed JSON arguments the model sent and returns a
string, which goes back to the model verbatim. Errors are returned as text
too - an exception the model never sees is a retry it cannot make.

Relative paths resolve against the workspace; absolute paths are allowed.
"""

from __future__ import annotations

import base64
import fnmatch
import html
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "build",
             "dist", ".vs", ".idea", "target", "obj", "bin", ".mypy_cache"}
OUT_LIMIT = 14000         # chars of command output handed back per call
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# kind decides the approval rule: read runs freely, edit and exec may ask
KIND = {"list_dir": "read", "read_file": "read", "search": "read",
        "fetch_url": "read", "bg_output": "read",
        "screenshot": "read", "view_image": "read",
        "write_file": "edit", "edit_file": "edit",
        "run": "exec", "bg_start": "exec", "bg_stop": "exec"}


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": list(required)}}}


S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}

SCHEMAS = [
    _fn("list_dir", "List files and folders as a tree. Skips .git, node_modules, venvs and build output.",
        {"path": {**S, "description": "folder, default the workspace"},
         "depth": {**I, "description": "levels to descend, 1-4, default 2"}}),
    _fn("read_file", "Read a text file with line numbers. Read before you edit.",
        {"path": S, "offset": {**I, "description": "first line, 1-based"},
         "limit": {**I, "description": "max lines, default 600"}}, ["path"]),
    _fn("search", "Regex search across files (Python re syntax). Returns path:line: text.",
        {"pattern": S, "path": {**S, "description": "folder or file, default workspace"},
         "glob": {**S, "description": "filename filter like *.lua or *.cpp"},
         "ignore_case": B}, ["pattern"]),
    _fn("write_file", "Create or overwrite a file with the complete content. Parent folders are created.",
        {"path": S, "content": S}, ["path", "content"]),
    _fn("edit_file", "Replace an exact snippet in a file. old_text must match exactly, including "
        "indentation, and be unique unless replace_all is true. Prefer this over rewriting big files.",
        {"path": S, "old_text": S, "new_text": S, "replace_all": B},
        ["path", "old_text", "new_text"]),
    _fn("run", "Run a PowerShell 5.1 command and wait for it. Use it to build, test, run, and install "
        "dependencies. Returns exit code and output. Do not start servers or GUI apps here - use bg_start.",
        {"command": S, "cwd": {**S, "description": "working folder, default workspace"},
         "timeout": {**I, "description": "seconds, default 300, max 1800"}}, ["command"]),
    _fn("bg_start", "Start a long-running command in the background (servers, watchers, GUI apps). "
        "Returns an id; read its output with bg_output.",
        {"command": S, "cwd": S}, ["command"]),
    _fn("bg_output", "Output so far and status of a background command.",
        {"id": I}, ["id"]),
    _fn("bg_stop", "Kill a background command and its children.", {"id": I}, ["id"]),
    _fn("fetch_url", "HTTP GET a URL and return its text (HTML tags stripped). For docs and APIs.",
        {"url": S}, ["url"]),
    _fn("screenshot", "Capture a window (or the whole screen) as an image and look at it. Use it to check a "
        "GUI you built: start the app with bg_start, wait, then screenshot it by its window title.",
        {"window_title": {**S, "description": "part of the window title; omit for the whole screen"}}),
    _fn("view_image", "Look at an image file (png/jpg/bmp), e.g. a reference design or a saved screenshot.",
        {"path": S}, ["path"]),
]


def _clip(text: str, limit=OUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 4
    return (text[:head] + f"\n... [{len(text) - limit} caracteres omitidos] ...\n"
            + text[-(limit - head):])


def _kill_tree(pid: int):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                   capture_output=True, creationflags=FLAGS)


def _ps_args(command: str):
    """PowerShell with the command passed base64-encoded, so no quote in it
    is ever re-parsed by the Windows argument splitter."""
    # UTF-8 *without* BOM: in PowerShell 5.1, [Text.Encoding]::UTF8 as
    # $OutputEncoding puts a BOM in front of everything piped into a native
    # program, so the first stdin line a compiled program reads is garbage.
    # `cmd` on this PATH is devkitPro msys2's extensionless script; running
    # it makes Windows pop "pick an app to open cmd" - pin the real one.
    pre = ("$ProgressPreference='SilentlyContinue';"
           "$OutputEncoding=[Console]::OutputEncoding=New-Object Text.UTF8Encoding $false;"
           "Set-Alias -Name cmd -Value \"$env:SystemRoot\\System32\\cmd.exe\" -Option AllScope;")
    enc = base64.b64encode((pre + command).encode("utf-16-le")).decode()
    return ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-EncodedCommand", enc]


def child_env() -> dict:
    env = dict(os.environ)
    # devkitPro's msys2 leaves TMP pointing at C:\WINDOWS for its cmake,
    # which makes gcc fail to create temp files; give every child a real one
    tmp = str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Temp")
    env["TMP"] = env["TEMP"] = tmp
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return env


class Background:
    def __init__(self, id_, proc, command):
        self.id, self.proc, self.command = id_, proc, command
        self.buf: list[str] = []
        self.read_upto = 0
        self.started = time.time()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.buf.append(line)


class Toolbox:
    def __init__(self, workspace: str):
        self.workspace = Path(workspace)
        self.bg: dict[int, Background] = {}
        self._next_bg = 1
        self.on_output = None      # callback(line) for live command output
        self.cancel: threading.Event | None = None
        self.images: list[Path] = []   # attached to the history after the tool results
        self.vision = True             # False when the server runs without a projector
        self._shot = 0

    def path(self, p: str | None) -> Path:
        if not p:
            return self.workspace
        q = Path(os.path.expandvars(os.path.expanduser(p)))
        return q if q.is_absolute() else (self.workspace / q)

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return f"erro: ferramenta desconhecida '{name}'. Disponíveis: {', '.join(KIND)}"
        try:
            return fn(**args)
        except TypeError as e:
            return f"erro: argumentos inválidos para {name}: {e}"
        except Exception as e:
            return f"erro: {type(e).__name__}: {e}"

    # -- files -----------------------------------------------------------

    def t_list_dir(self, path=None, depth=2):
        root = self.path(path)
        if not root.exists():
            return f"erro: {root} não existe"
        depth = max(1, min(4, int(depth or 2)))
        out, count = [str(root)], 0

        def walk(d: Path, level: int, pre: str):
            nonlocal count
            try:
                items = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except OSError as e:
                out.append(pre + f"[sem acesso: {e}]")
                return
            for it in items:
                if count >= 500:
                    return
                count += 1
                if it.is_dir():
                    if it.name in SKIP_DIRS:
                        out.append(pre + it.name + "/ (ignorado)")
                        continue
                    out.append(pre + it.name + "/")
                    if level < depth:
                        walk(it, level + 1, pre + "  ")
                else:
                    try:
                        sz = it.stat().st_size
                    except OSError:
                        sz = 0
                    out.append(f"{pre}{it.name}  ({sz} B)")

        walk(root, 1, "  ")
        if count >= 500:
            out.append("... (lista cortada em 500 itens)")
        return "\n".join(out)

    def t_read_file(self, path, offset=1, limit=600):
        p = self.path(path)
        if not p.is_file():
            return f"erro: arquivo não encontrado: {p}"
        data = p.read_bytes()
        if b"\0" in data[:4096]:
            return f"erro: {p} parece binário ({len(data)} bytes)"
        lines = data.decode("utf-8", errors="replace").splitlines()
        offset = max(1, int(offset or 1))
        limit = max(1, min(2000, int(limit or 600)))
        chunk = lines[offset - 1: offset - 1 + limit]
        body = "\n".join(f"{offset + i:5}| {ln[:2000]}" for i, ln in enumerate(chunk))
        tail = ""
        if offset - 1 + limit < len(lines):
            tail = f"\n[... {len(lines)} linhas no total; continue com offset={offset + limit}]"
        return (body or "[arquivo vazio]") + tail

    def t_write_file(self, path, content):
        p = self.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content, encoding="utf-8", newline="")
        n = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"{'sobrescrito' if existed else 'criado'}: {p} ({n} linhas)"

    def t_edit_file(self, path, old_text, new_text, replace_all=False):
        p = self.path(path)
        if not p.is_file():
            return f"erro: arquivo não encontrado: {p}"
        text = p.read_text(encoding="utf-8", errors="replace")
        n = text.count(old_text)
        if n == 0:
            # the usual cause is whitespace; say so instead of just failing
            loose = re.escape(old_text.strip()).replace(r"\ ", r"\s+")
            hint = (" Um trecho parecido existe com espaços/indentação diferentes;"
                    " leia o arquivo de novo e copie exatamente."
                    if old_text.strip() and re.search(loose, text) else
                    " Leia o arquivo de novo antes de editar.")
            return f"erro: old_text não encontrado em {p}.{hint}"
        if n > 1 and not replace_all:
            return (f"erro: old_text aparece {n} vezes em {p}. Inclua mais contexto"
                    " para ficar único, ou use replace_all=true.")
        text = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
        p.write_text(text, encoding="utf-8", newline="")
        return f"editado: {p} ({n if replace_all else 1} substituição)"

    def t_search(self, pattern, path=None, glob=None, ignore_case=False):
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error as e:
            return f"erro: regex inválida: {e}"
        root = self.path(path)
        files = [root] if root.is_file() else []
        if root.is_dir():
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if d not in SKIP_DIRS]
                for f in fns:
                    if not glob or fnmatch.fnmatch(f, glob):
                        files.append(Path(dp) / f)
        hits = []
        for f in files:
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:2048]:
                continue
            for i, ln in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if rx.search(ln):
                    hits.append(f"{f}:{i}: {ln.strip()[:240]}")
                    if len(hits) >= 200:
                        return "\n".join(hits) + "\n... (cortado em 200 resultados)"
        return "\n".join(hits) if hits else "nenhum resultado"

    # -- shell -----------------------------------------------------------

    def t_run(self, command, cwd=None, timeout=300):
        wd = self.path(cwd)
        if not wd.is_dir():
            return f"erro: pasta não existe: {wd}"
        timeout = max(5, min(1800, int(timeout or 300)))
        proc = subprocess.Popen(
            _ps_args(command), cwd=str(wd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=child_env(),
            creationflags=FLAGS, text=True, encoding="utf-8", errors="replace")
        out: list[str] = []

        def pump():
            for line in proc.stdout:
                out.append(line)
                if self.on_output:
                    self.on_output(line.rstrip("\n"))

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        t0, status = time.time(), None
        while proc.poll() is None:
            if self.cancel is not None and self.cancel.is_set():
                _kill_tree(proc.pid)
                status = "cancelado pelo usuário"
                break
            if time.time() - t0 > timeout:
                _kill_tree(proc.pid)
                status = (f"TIMEOUT após {timeout}s - processo encerrado. Se é um servidor"
                          " ou app que não termina, use bg_start.")
                break
            time.sleep(0.05)
        proc.wait()
        th.join(2)
        text = "".join(out).rstrip()
        head = status or f"exit code {proc.returncode}"
        return f"[{head}] ({time.time() - t0:.1f}s)\n{_clip(text) if text else '(sem saída)'}"

    def t_bg_start(self, command, cwd=None):
        wd = self.path(cwd)
        proc = subprocess.Popen(
            _ps_args(command), cwd=str(wd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=child_env(),
            creationflags=FLAGS, text=True, encoding="utf-8", errors="replace")
        b = Background(self._next_bg, proc, command)
        self.bg[b.id] = b
        self._next_bg += 1
        time.sleep(1.5)
        return f"iniciado id={b.id} pid={proc.pid}\n" + self.t_bg_output(b.id)

    def t_bg_output(self, id):
        b = self.bg.get(int(id))
        if not b:
            return f"erro: nenhum processo em segundo plano com id {id}"
        new = "".join(b.buf[b.read_upto:])
        b.read_upto = len(b.buf)
        code = b.proc.poll()
        state = "rodando" if code is None else f"terminou (exit {code})"
        return f"[id {b.id} {state}, {time.time() - b.started:.0f}s]\n{_clip(new) if new else '(sem saída nova)'}"

    def t_bg_stop(self, id):
        b = self.bg.pop(int(id), None)
        if not b:
            return f"erro: nenhum processo em segundo plano com id {id}"
        if b.proc.poll() is None:
            _kill_tree(b.proc.pid)
        return f"parado id={b.id}"

    def stop_all(self):
        for b in list(self.bg.values()):
            if b.proc.poll() is None:
                _kill_tree(b.proc.pid)
        self.bg.clear()

    # -- images ----------------------------------------------------------

    def t_screenshot(self, window_title=None):
        if not self.vision:
            return "erro: o modelo carregado não tem visão (mmproj ausente); não consigo ver imagens"
        from . import vision
        self._shot += 1
        out = self.workspace / ".lcoder_shots" / time.strftime(f"shot-%H%M%S-{self._shot}.png")
        what = vision.screenshot(out, window_title or None)
        self.images.append(out)
        return f"captura de {what} salva em {out}; a imagem segue na próxima mensagem"

    def t_view_image(self, path):
        if not self.vision:
            return "erro: o modelo carregado não tem visão (mmproj ausente); não consigo ver imagens"
        p = self.path(path)
        if not p.is_file():
            return f"erro: imagem não encontrada: {p}"
        self.images.append(p)
        return f"imagem {p} segue na próxima mensagem"

    # -- web -------------------------------------------------------------

    def _get(self, url, accept=None):
        h = {"User-Agent": "Mozilla/5.0 LocalCoder"}
        if accept:
            h["Accept"] = accept
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
            return r.read(5_000_000).decode("utf-8", errors="replace"), r.headers.get("Content-Type", "")

    def _github(self, url):
        """GitHub's HTML is mostly chrome; a repo page becomes its README plus
        the file tree, a file page becomes the raw file, a folder its listing."""
        m = re.match(r"https?://(?:www\.)?github\.com/([^/]+)/([^/#?]+)(?:/(blob|tree)/([^/]+)/?(.*))?", url)
        if not m:
            return None
        owner, repo, kind, ref, sub = m.groups()
        repo = repo.removesuffix(".git")
        api = f"https://api.github.com/repos/{owner}/{repo}"
        if kind == "blob":
            text, _ = self._get(f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{sub}")
            return f"[arquivo bruto {owner}/{repo}@{ref}:{sub}]\n" + text
        if kind == "tree" and sub:
            text, _ = self._get(f"{api}/contents/{sub}?ref={ref}")
            rows = [f"{e['type']:4} {e['path']}" for e in json.loads(text)]
            return f"[pasta {owner}/{repo}@{ref}:{sub}]\n" + "\n".join(rows)
        parts = []
        try:
            readme, _ = self._get(f"{api}/readme", accept="application/vnd.github.raw")
            parts.append("[README]\n" + readme)
        except Exception as e:
            parts.append(f"[sem README: {e}]")
        try:
            tree, _ = self._get(f"{api}/git/trees/{ref or 'HEAD'}?recursive=1")
            paths = [e["path"] for e in json.loads(tree).get("tree", []) if e["type"] == "blob"]
            parts.insert(0, f"[repositório {owner}/{repo}: {len(paths)} arquivos]\n"
                         + "\n".join(paths[:400]) + ("\n..." if len(paths) > 400 else "")
                         + f"\n\nLer um arquivo: fetch_url https://github.com/{owner}/{repo}/blob/{ref or 'HEAD'}/<caminho>"
                         + f"\nPara usar/estudar o código todo: git clone --depth 1 https://github.com/{owner}/{repo}")
        except Exception as e:
            parts.insert(0, f"[árvore indisponível: {e}]")
        return "\n\n".join(parts)

    def t_fetch_url(self, url):
        text = self._github(url)
        if text is None:
            text, ctype = self._get(url)
            if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
                text = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", text)
                text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h\d|tr|pre)>", "\n", text)
                text = html.unescape(re.sub(r"<[^>]+>", "", text))
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = text.strip()
        if len(text) <= 20000:
            return text
        # too long for one tool result: keep it all on disk, page with read_file
        cache = self.workspace / ".lcoder_web"
        cache.mkdir(exist_ok=True)
        name = re.sub(r"[^\w.-]+", "_", url.split("://", 1)[-1])[-90:] + ".txt"
        (cache / name).write_text(text, encoding="utf-8")
        return (f"[página longa: {len(text)} caracteres; conteúdo completo salvo em "
                f"{cache / name} - leia o resto com read_file e offset]\n" + text[:16000])
