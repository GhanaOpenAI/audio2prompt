#!/usr/bin/env python3
"""Serve a review folder on localhost, with Gemini transcription when you pass a clip.

    GEMINI_API_KEY=... python serve.py review --language "Asante Twi"
    # open http://localhost:8765

Opened this way (instead of double-clicking index.html), the review page transcribes every clip you mark
good with Gemini, as trimmed and cut at that moment, and fills in its transcript for you to correct.
The API key stays in this process; the page never sees it. Without a key it simply serves the page.
"""
import argparse
import functools
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from apply_picks import keep_segments, splice  # noqa: E402
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
        if self.path.split("?")[0].endswith("/api/status"):
            c = self.cfg
            return self._json(200, {"transcribe": bool(c["key"]), "model": c["model"], "language": c["language"]})
        return super().do_GET()

    def do_POST(self):
        if not self.path.split("?")[0].endswith("/api/transcribe"):
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
