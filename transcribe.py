#!/usr/bin/env python3
"""Transcribe audio clips with Gemini (serve.py uses analyze_one; this script transcribes a whole folder).

    GEMINI_API_KEY=... python transcribe.py path/to/clips --language "Asante Twi" --out transcripts.csv

The key comes from $GEMINI_API_KEY or --key-file. Results are cached by audio content, so
re-running only sends clips that are new or changed.
"""
import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MODEL = "gemini-flash-latest"
MIME = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mp3", ".ogg": "audio/ogg", ".opus": "audio/ogg",
        ".oga": "audio/ogg", ".aiff": "audio/aiff", ".aif": "audio/aiff"}


def prompt_for(language):
    lang = f"in {language}" if language else "in the language that is spoken"
    return (f"Transcribe this speech exactly {lang}, word for word, using that language's standard "
            "orthography and its special letters and tone marks where they are normally written "
            "(for Ghanaian languages, for example ɛ ɔ ŋ ɖ ƒ ɣ ʋ). Do not translate, summarise or add "
            "anything. If a word is unclear, write your best guess. Output only the transcription text.")


VOICES = ["young male", "adult male", "elderly male", "young female", "adult female", "elderly female", "child"]
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted", "calm", "excited",
            "tender", "stern", "pleading", "sarcastic", "curious"]


def make_label(voice, emotion):
    return f"{voice} - {emotion}"


def analyze_one(path, key, model, language, voices=VOICES, emotions=EMOTIONS, retries=5):
    """One Gemini call -> {"text", "voice", "emotion", "label"}: the transcription plus the speaker's voice
    characteristic and perceived emotion, each chosen from the given lists."""
    data = open(path, "rb").read()
    lang = f"in {language}" if language else "in the language that is spoken"
    prompt = (f"Listen to this speech clip.\n1. transcription: {prompt_for(language)}\n"
              f"2. voice: the speaker's voice characteristic, chosen from the allowed values.\n"
              f"3. emotion: the emotion the speaker conveys (tone of voice, not just the words), chosen from the "
              f"allowed values.")
    schema = {"type": "OBJECT", "properties": {
        "transcription": {"type": "STRING"},
        "voice": {"type": "STRING", "enum": list(voices)},
        "emotion": {"type": "STRING", "enum": list(emotions)}},
        "required": ["transcription", "voice", "emotion"]}
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": MIME.get(os.path.splitext(path)[1].lower(), "audio/wav"),
                             "data": base64.b64encode(data).decode()}},
            {"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": schema},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        json.dumps(body).encode(), {"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(retries):
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
            out = json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
            return {"text": out["transcription"].strip(), "voice": out["voice"], "emotion": out["emotion"],
                    "label": make_label(out["voice"], out["emotion"])}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}")
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            if attempt < retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise


def gemini_key(key_file=None):
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    if key_file and os.path.exists(os.path.expanduser(key_file)):
        return open(os.path.expanduser(key_file)).read().strip()
    return None


def transcribe_one(path, key, model, language, retries=5):
    data = open(path, "rb").read()
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": MIME.get(os.path.splitext(path)[1].lower(), "audio/wav"),
                             "data": base64.b64encode(data).decode()}},
            {"text": prompt_for(language)},
        ]}],
        "generationConfig": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        json.dumps(body).encode(), {"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(retries):
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
            parts = resp["candidates"][0]["content"]["parts"]
            return " ".join(p.get("text", "") for p in parts).strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise


def transcribe_many(paths, key, model=DEFAULT_MODEL, language=None, cache_file=None, workers=6, log=print):
    """paths: {name: file path} -> {name: text}. Uses/updates a JSON cache keyed by audio hash."""
    cache = json.load(open(cache_file, encoding="utf-8")) if cache_file and os.path.exists(cache_file) else {}
    lock, out, todo = threading.Lock(), {}, []
    for name, p in paths.items():
        h = hashlib.sha1(open(p, "rb").read()).hexdigest() + f"|{model}|{language or ''}"
        if h in cache:
            out[name] = cache[h]
        else:
            todo.append((name, p, h))
    if todo:
        log(f"transcribing {len(todo)} clips with {model} ({len(out)} cached) ...")

    def work(item):
        name, p, h = item
        try:
            text = transcribe_one(p, key, model, language)
        except Exception as e:
            log(f"  transcription failed for {name}: {e}")
            return
        with lock:
            out[name] = cache[h] = text
            if cache_file:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=1)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    return out


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build import find_audio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_dir")
    ap.add_argument("--out", default="transcripts.csv")
    ap.add_argument("--language", default=None, help='e.g. "Asante Twi", "Ewe", "Dagbani" (default: auto)')
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key-file", default="~/.gemini_api_key", help="file holding the key if $GEMINI_API_KEY is unset")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    key = gemini_key(a.key_file) or sys.exit("Set GEMINI_API_KEY or pass --key-file")
    files = find_audio(a.audio_dir)
    texts = transcribe_many({f: os.path.join(a.audio_dir, f) for f in files}, key, a.model, a.language,
                            cache_file=os.path.splitext(a.out)[0] + ".cache.json", workers=a.workers)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "text"])
        w.writerows([(name, texts.get(name, "")) for name in files])
    print(f"{len(texts)} of {len(files)} clips transcribed -> {a.out}")


if __name__ == "__main__":
    main()
