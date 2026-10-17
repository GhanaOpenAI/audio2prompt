#!/usr/bin/env python3
"""Serve a review folder on localhost, with Gemini transcription when you pass a clip.

    GEMINI_API_KEY=... python serve.py review --language "Asante Twi"
    # open http://localhost:8765

Opened this way (instead of double-clicking index.html), the review page:
  * transcribes every clip you mark good with Gemini (as trimmed and cut at that moment) for you to correct;
  * can download the finished dataset (audio/ + metadata.csv, zipped) or push it to a Hugging Face
    dataset repo: paste a token in the page (or use this machine's `huggingface-cli login`), pick your
    account or one of your organizations, and name the dataset.
Keys and tokens stay in this process; the page never sees them. Without a Gemini key it skips transcription.
"""
import argparse
import functools
import hashlib
import io
import json
import os
import sys
import shutil
import tempfile
import threading
import zipfile
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from apply_picks import export_dataset, keep_segments, splice  # noqa: E402
from transcribe import DEFAULT_MODEL, gemini_key, transcribe_one  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    cfg = {}
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # only log API calls, not every audio request
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0].endswith("/api/progress"):
            f = os.path.join(self.cfg["root"], "progress.json")
            return self._json(200, json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {})
        if self.path.split("?")[0].endswith("/api/status"):
            c = self.cfg
            return self._json(200, {"transcribe": bool(c["key"]), "model": c["model"], "language": c["language"],
                                    "export": True, "hf_user": hf_user()})
        return super().do_GET()

    def _build(self, req, out_dir):
        """Write the dataset for the picks sent by the page into out_dir."""
        root, srcs = self.cfg["root"], req.get("srcs", {})

        def path_of(f):
            p = os.path.realpath(os.path.join(root, srcs.get(f, f)))
            if not p.startswith(root + os.sep):
                raise ValueError(f"bad path for {f}")
            return p
        return export_dataset(req["picks"], path_of, out_dir)

    def do_POST(self):
        route = self.path.split("?")[0]
        if route.endswith("/api/export"):
            return self._export()
        if route.endswith("/api/push"):
            return self._push()
        if route.endswith("/api/hf/connect"):
            return self._hf_connect()
        if route.endswith("/api/progress"):
            return self._save_progress()
        if not route.endswith("/api/transcribe"):
            return self._json(404, {"error": "not found"})
        c = self.cfg
        if not c["key"]:
            return self._json(400, {"error": "no Gemini key: set GEMINI_API_KEY or --key-file"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            path = os.path.realpath(os.path.join(c["root"], req["src"]))
            if not path.startswith(c["root"] + os.sep) or not os.path.isfile(path):
                return self._json(400, {"error": "unknown clip"})
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            start, end = req.get("trim") or [0.0, len(data) / sr]
            audio = splice(data, sr, keep_segments(start, end, req.get("cuts") or []))
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
            wav = buf.getvalue()
            key = hashlib.sha1(wav).hexdigest() + f"|{c['model']}|{c['language'] or ''}"
            with self.lock:
                cached = c["cache"].get(key)
            if cached is None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav)
                try:
                    cached = transcribe_one(tmp.name, c["key"], c["model"], c["language"])
                finally:
                    os.unlink(tmp.name)
                with self.lock:
                    c["cache"][key] = cached
                    with open(c["cache_file"], "w", encoding="utf-8") as f:
                        json.dump(c["cache"], f, ensure_ascii=False, indent=1)
            return self._json(200, {"text": cached})
        except Exception as e:
            return self._json(500, {"error": str(e)[:300]})

    def _read(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def _export(self):
        """Zip of audio/*.wav + metadata.csv for the good clips."""
        try:
            req = self._read()
            with tempfile.TemporaryDirectory() as tmp:
                rows, short, long_ = self._build(req, tmp)
                if not rows:
                    return self._json(400, {"error": "no clips to export: mark some clips good first"})
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for d, _, names in os.walk(tmp):
                        for n in names:
                            full = os.path.join(d, n)
                            z.write(full, os.path.join("dataset", os.path.relpath(full, tmp)))
            body = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="dataset.zip"')
            self.send_header("X-Clips", str(len(rows)))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            return self._json(500, {"error": str(e)[:300]})

    def _save_progress(self):
        """Keep the reviewer's work (ratings, trims, cuts, transcripts, labels) in <review>/progress.json,
        so it survives other browsers, cleared browser data and restarts."""
        try:
            state = self._read()
            f = os.path.join(self.cfg["root"], "progress.json")
            with self.lock:
                with open(f + ".tmp", "w", encoding="utf-8") as out:
                    json.dump(state, out, ensure_ascii=False)
                os.replace(f + ".tmp", f)
            return self._json(200, {"saved": True})
        except Exception as e:
            return self._json(500, {"error": str(e)[:200]})

    def _hf_connect(self):
        """Check a Hugging Face token (or this machine's login) and list the namespaces it can push to.
        The token is kept in this process's memory for the session only."""
        try:
            from huggingface_hub import HfApi
            token = (self._read().get("token") or "").strip() or None
            me = HfApi(token=token).whoami()
            orgs = [o["name"] for o in me.get("orgs", []) if o.get("name")]
            with self.lock:
                self.cfg["hf_token"] = token  # None = use this machine's login
            return self._json(200, {"user": me["name"], "orgs": orgs})
        except Exception as e:
            msg = str(e)
            if "401" in msg or "Invalid" in msg:
                msg = "that token was not accepted by Hugging Face"
            return self._json(400, {"error": msg[:300]})

    def _push(self):
        """Push the dataset to a Hugging Face dataset repo (audiofolder layout)."""
        try:
            from huggingface_hub import HfApi
            req = self._read()
            repo, private = (req.get("repo_id") or "").strip(), bool(req.get("private", True))
            if not repo or repo.count("/") != 1:
                return self._json(400, {"error": "give the dataset as owner/name"})
            if "hf_token" not in self.cfg:
                return self._json(400, {"error": "connect to Hugging Face first"})
            api = HfApi(token=self.cfg["hf_token"])
            with tempfile.TemporaryDirectory() as tmp:
                rows, short, long_ = self._build(req, tmp)
                if not rows:
                    return self._json(400, {"error": "no clips to push: mark some clips good first"})
                api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
                if not api.file_exists(repo, "README.md", repo_type="dataset"):
                    with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
                        f.write(dataset_card(repo, rows, self.cfg["language"]))
                info = api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="dataset",
                                         delete_patterns=["audio/**", "metadata.csv"],
                                         commit_message=f"audio2prompt: {len(rows)} reviewed clips")
            return self._json(200, {"url": f"https://huggingface.co/datasets/{repo}", "clips": len(rows),
                                    "commit": getattr(info, "commit_url", "")})
        except Exception as e:
            return self._json(500, {"error": str(e)[:400]})


_HF_USER = {}


def hf_user():
    """Name of the Hugging Face account logged in on this machine (None if not logged in)."""
    if "name" not in _HF_USER:
        try:
            from huggingface_hub import HfApi
            _HF_USER["name"] = HfApi().whoami()["name"]
        except Exception:
            _HF_USER["name"] = None
    return _HF_USER["name"]


def dataset_card(repo, rows, language):
    total = sum(r["duration_sec"] for r in rows)
    labels = Counter(r["label"] for r in rows if r["label"])
    lines = [
        "---", "task_categories:", "- text-to-speech", "- automatic-speech-recognition", "---", "",
        f"# {repo.split('/')[-1]}", "",
        f"{len(rows)} speech clips ({total / 60:.1f} minutes){' in ' + language if language else ''}, curated by ear "
        "with [audio2prompt](https://github.com/GhanaOpenAI/audio2prompt): silences removed, each clip listened to, "
        "trimmed or cut where needed, and its transcription checked.", "",
        "| column | |", "|---|---|", "| `audio` | the clip (WAV) |", "| `transcription` | what is said |",
        "| `label` | reviewer's label |", "| `duration_sec` | length in seconds |",
        "| `source_file`, `start_sec`, `end_sec`, `cuts` | where the clip came from and how it was trimmed/cut |", "",
    ]
    if labels:
        lines += ["Labels: " + ", ".join(f"{k} ({v})" for k, v in labels.most_common()), ""]
    lines += ["```python", "from datasets import load_dataset", f'ds = load_dataset("{repo}", split="train")', "```", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_dir", nargs="?", default="review", help="folder with index.html (from build.py / screen.py)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your network")
    ap.add_argument("--language", default=None, help='spoken language, e.g. "Asante Twi", "Ewe" (default: auto-detect)')
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    ap.add_argument("--key-file", default="~/.gemini_api_key", help="file holding the key if $GEMINI_API_KEY is unset")
    a = ap.parse_args()

    root = os.path.realpath(a.review_dir)
    if not os.path.isfile(os.path.join(root, "index.html")):
        sys.exit(f"No index.html in {root}; build one with build.py or screen.py first")
    cache_file = os.path.join(root, "gemini_transcripts.json")
    key = gemini_key(a.key_file)
    Handler.cfg = {"root": root, "key": key, "model": a.model, "language": a.language, "cache_file": cache_file,
                   "cache": json.load(open(cache_file, encoding="utf-8")) if os.path.exists(cache_file) else {}}
    srv = ThreadingHTTPServer((a.host, a.port), functools.partial(Handler, directory=root))
    print(f"Review page: http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}/")
    print(f"Gemini transcription: {'on (' + a.model + (', ' + a.language if a.language else '') + ')' if key else 'off (no key)'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
