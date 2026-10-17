#!/usr/bin/env python3
"""Build a self-contained HTML page for listening to, rating and trimming audio clips.

    python build.py path/to/audio                              # every audio file under the folder
    python build.py path/to/audio --meta clips.csv             # add text / notes per clip
    python build.py path/to/audio --out review --copy          # portable folder you can zip and share
    python build.py path/to/audio --min-duration 2 --max-duration 15   # allowed kept length per clip

Open <out>/index.html in a browser. Ratings and trims are saved in the browser as you go;
"Download picks" exports them as picks.json for apply_picks.py.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aiff", ".aif"}
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviewer", "template.html")


def find_audio(root):
    files = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            if os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                files.append(os.path.relpath(os.path.join(dirpath, n), root))
    return sorted(files)


def load_meta(path):
    """Rows keyed by file. Accepts .csv, .json (list of objects) or .jsonl."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8") as f:
        if ext == ".csv":
            return list(csv.DictReader(f))
        if ext == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
        return data if isinstance(data, list) else [dict(v, file=k) for k, v in data.items()]


def speech_regions_energy(mono, sr, floor_db=-35.0, min_speech=0.1):
    """Fallback speech detector: 20 ms frames louder than (peak + floor_db)."""
    n = int(0.02 * sr)
    if len(mono) < n:
        return []
    fr = mono[: len(mono) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    on = db > db.max() + floor_db
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if (j - i) * 0.02 >= min_speech:
                runs.append((i * 0.02, j * 0.02))
            i = j
        else:
            i += 1
    return runs


class SpeechDetector:
    """Silero VAD (via screen.py, CPU/ONNX) when available, else an energy threshold."""

    def __init__(self, mode):
        self.mode, self.vad = mode, None
        if mode == "vad":
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import screen
                self.screen, self.vad = screen, screen.SileroVAD(threads=max(1, (os.cpu_count() or 2) // 2))
            except Exception as e:  # onnxruntime/scipy missing or no network for the model
                print(f"Silero VAD unavailable ({e}); using the energy detector for silences", file=sys.stderr)
                self.mode = "energy"

    def regions(self, data, sr):
        mono = data.mean(axis=1)
        if self.mode == "vad":
            return self.vad.speech_runs(self.screen.to_mono16k(data, sr))
        return speech_regions_energy(mono, sr)


def peaks(path, n=300, detector=None):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    speech = [[round(a, 3), round(b, 3)] for a, b in detector.regions(data, sr)] if detector else None
    x = np.abs(data).max(axis=1)
    dur = len(x) / sr
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    b = len(x) // n
    p = x[: b * n].reshape(n, b).max(axis=1)
    p = p / (p.max() + 1e-9)
    return round(dur, 3), [round(float(v), 2) for v in p], speech


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_dir", help="folder containing the audio clips (searched recursively)")
    ap.add_argument("--meta", help="optional .csv / .json / .jsonl with one row per clip")
    ap.add_argument("--out", default="review", help="output folder (default: review)")
    ap.add_argument("--title", default="Audio clip review")
    ap.add_argument("--min-duration", type=float, default=3.0,
                    help="seconds a clip must keep after trimming to be markable good; shorter clips are "
                         "marked bad automatically (default: 3.0, 0 = no minimum)")
    ap.add_argument("--max-duration", type=float, default=10.0,
                    help="seconds a clip may keep and still pass; longer clips must be trimmed before they can be "
                         "marked good (default: 10.0, 0 = no maximum)")
    ap.add_argument("--copy", action="store_true",
                    help="copy the audio into <out>/audio so the folder is self-contained (default: link by relative path)")
    ap.add_argument("--silences", choices=["vad", "energy", "off"], default="vad",
                    help="detect silences for automatic removal: Silero VAD (default), an energy threshold, or off")
    ap.add_argument("--no-auto-silence", action="store_true",
                    help="detect silences but don't remove them automatically when the page opens")
    ap.add_argument("--min-gap", type=float, default=0.3, help="inner silences at least this long are shortened (s)")
    ap.add_argument("--keep-gap", type=float, default=0.15, help="pause left in place of a removed silence (s)")
    ap.add_argument("--edge-pad", type=float, default=0.1, help="silence kept before the first / after the last word (s)")
    ap.add_argument("--file-col", default="file", help="meta column holding the file name / relative path")
    ap.add_argument("--text-col", default="text", help="meta column shown in bold (e.g. transcript)")
    ap.add_argument("--subtitle-col", default="subtitle", help="meta column shown under the text (e.g. translation)")
    ap.add_argument("--tags-col", default="tags", help="meta column with short tags (list, or ';'-separated)")
    ap.add_argument("--notes-col", default="notes", help="meta column hidden until 'show notes' is ticked")
    a = ap.parse_args()

    audio_dir = os.path.abspath(a.audio_dir)
    out = os.path.abspath(a.out)
    files = find_audio(audio_dir)
    if not files:
        sys.exit(f"No audio files ({', '.join(sorted(AUDIO_EXTS))}) found under {audio_dir}")

    meta = {}
    if a.meta:
        for row in load_meta(a.meta):
            key = str(row.get(a.file_col, "")).strip()
            if key:
                meta[key] = row
                meta.setdefault(os.path.basename(key), row)

    os.makedirs(out, exist_ok=True)
    detector = SpeechDetector(a.silences) if a.silences != "off" else None
    clips, skipped = [], []
    for n, rel in enumerate(files, 1):
        path = os.path.join(audio_dir, rel)
        try:
            dur, pk, speech = peaks(path, detector=detector)
        except Exception as e:  # unreadable / unsupported codec
            skipped.append(f"{rel}: {e}")
            continue
        if a.copy:
            dst = os.path.join(out, "audio", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(path, dst)
            src = os.path.join("audio", rel)
        else:
            src = os.path.relpath(path, out)
        row = meta.get(rel) or meta.get(os.path.basename(rel)) or {}
        tags = row.get(a.tags_col) or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(";") if t.strip()]
        clips.append({
            "n": n, "file": rel.replace(os.sep, "/"), "src": src.replace(os.sep, "/"),
            "text": str(row.get(a.text_col) or ""), "subtitle": str(row.get(a.subtitle_col) or ""),
            "tags": [str(t) for t in tags], "notes": str(row.get(a.notes_col) or ""),
            "dur": dur, "peaks": pk, "speech": speech,
        })

    store_id = hashlib.sha1((a.title + "\n" + "\n".join(c["file"] for c in clips)).encode()).hexdigest()[:12]
    data = json.dumps(clips, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = a.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = (open(TEMPLATE, encoding="utf-8").read()
            .replace("__TITLE__", title).replace("__CLIPS__", data).replace("__STORE_ID__", json.dumps(store_id))
            .replace("__MIN_DURATION__", json.dumps(max(0.0, a.min_duration)))
            .replace("__MAX_DURATION__", json.dumps(max(0.0, a.max_duration)))
            .replace("__SILENCE__", json.dumps({"auto": detector is not None and not a.no_auto_silence,
                                                "min_gap": a.min_gap, "keep": a.keep_gap, "pad": a.edge_pad})))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"{len(clips)} clips -> {os.path.join(out, 'index.html')}")
    if a.min_duration > 0:
        short = sum(1 for c in clips if c["dur"] < a.min_duration)
        print(f"minimum kept length {a.min_duration:g}s: {short} clips are already shorter and start as bad")
    if a.max_duration > 0:
        long_ = sum(1 for c in clips if c["dur"] > a.max_duration)
        print(f"maximum kept length {a.max_duration:g}s: {long_} clips are longer and must be trimmed to pass")
    if a.meta:
        print(f"metadata matched for {sum(1 for c in clips if c['text'] or c['subtitle'] or c['notes'] or c['tags'])} clips")
    for s in skipped:
        print("skipped", s, file=sys.stderr)


if __name__ == "__main__":
    main()
