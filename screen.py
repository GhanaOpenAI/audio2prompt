#!/usr/bin/env python3
"""Screen a large pool of audio clips on CPU and shortlist the best N for human review.

    python screen.py --folder path/to/clips --top 200
    python screen.py --hf user/dataset --split train --audio-column audio --text-column text --top 200

Every clip is scored with two small ONNX models (CPU only, no PyTorch):

  * Silero VAD (2 MB)   where the speech is: trims silence off both ends, measures the longest
                         pause inside the speech and how much of the clip is speech
  * DNSMOS (1 MB)       Microsoft's non-intrusive speech-quality predictor: SIG (speech),
                         BAK (background: higher = cleaner) and OVRL (overall), each 1-5

Clips that fail a hard rule (length outside --min/--max-duration after trimming, a pause longer than
--max-pause, too little speech, clipping) are rejected; the rest are ranked by DNSMOS and the top N are
written, edge-trimmed, to <out>/audio together with a review page (<out>/index.html, built by build.py).
<out>/screening.csv has the scores and verdict for every clip scanned.
"""
import argparse
import csv
import hashlib
import heapq
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "audio-clip-reviewer")
MODELS = {
    "silero_vad.onnx": ("https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx",
                        "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"),
    "sig_bak_ovr.onnx": ("https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
                         "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"),
}
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aiff", ".aif"}
SR = 16000


def model_path(name):
    url, sha = MODELS[name]
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        print(f"downloading {name} ...", flush=True)
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != sha:
            sys.exit(f"checksum mismatch for {name} from {url}")
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    return path


def ort_session(path, threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


class SileroVAD:
    """Silero VAD v5 (ONNX), 16 kHz, 32 ms frames."""
    FRAME, CONTEXT = 512, 64

    def __init__(self, threads):
        self.sess = ort_session(model_path("silero_vad.onnx"), threads)

    def probs(self, x):
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros((1, self.CONTEXT), np.float32)
        n = len(x) // self.FRAME
        out = np.empty(n, np.float32)
        sr = np.array(SR, np.int64)
        for i in range(n):
            chunk = np.concatenate([context, x[None, i * self.FRAME:(i + 1) * self.FRAME]], axis=1).astype(np.float32)
            p, state = self.sess.run(None, {"input": chunk, "state": state, "sr": sr})
            out[i] = p[0, 0]
            context = chunk[:, -self.CONTEXT:]
        return out

    def analyse(self, x, threshold=0.5, min_speech=0.1):
        """-> (start_s, end_s, speech_ratio_within, longest_internal_pause_s) or None if no speech."""
        fd = self.FRAME / SR
        p = self.probs(x)
        on = p >= threshold
        # hysteresis: stay "speech" while p stays above threshold - 0.15
        for i in range(1, len(on)):
            if not on[i] and on[i - 1] and p[i] >= threshold - 0.15:
                on[i] = True
        runs, i = [], 0  # speech runs as (start_frame, end_frame)
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and on[j]:
                    j += 1
                if (j - i) * fd >= min_speech:
                    runs.append((i, j))
                i = j
            else:
                i += 1
        if not runs:
            return None
        start, end = runs[0][0] * fd, runs[-1][1] * fd
        gaps = [(runs[k + 1][0] - runs[k][1]) * fd for k in range(len(runs) - 1)]
        speech = sum((b - a) * fd for a, b in runs)
        return start, end, speech / max(end - start, 1e-6), max(gaps, default=0.0)


class DNSMOS:
    """DNSMOS P.835 (sig_bak_ovr.onnx), as in Microsoft's DNS-Challenge reference script."""
    LEN = int(9.01 * SR)
    P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

    def __init__(self, threads):
        self.sess = ort_session(model_path("sig_bak_ovr.onnx"), threads)

    def score(self, x):
        # Short clips are looped to fill exactly one 9.01 s window (the reference script doubles them,
        # which can mean scoring up to 9 overlapping windows of the same looped audio).
        a = np.resize(x, self.LEN) if len(x) < self.LEN else x
        hops = max(1, int(np.floor(len(a) / SR) - 9.01) + 1)
        segs = np.stack([a[h * SR: h * SR + self.LEN] for h in range(hops) if len(a[h * SR: h * SR + self.LEN]) == self.LEN])
        raw = self.sess.run(None, {"input_1": segs.astype(np.float32)})[0]
        sig, bak, ovr = raw[:, 0], raw[:, 1], raw[:, 2]
        return float(np.mean(self.P_SIG(sig))), float(np.mean(self.P_BAK(bak))), float(np.mean(self.P_OVR(ovr)))


# ---------------------------------------------------------------------------
# sources: yield (key, audio float32 [n, ch], sr, text)
# ---------------------------------------------------------------------------
def from_folder(root, text_by_file):
    files = sorted(os.path.relpath(os.path.join(d, n), root) for d, _, ns in os.walk(root) for n in ns
                   if os.path.splitext(n)[1].lower() in AUDIO_EXTS)
    for rel in files:
        try:
            x, sr = sf.read(os.path.join(root, rel), dtype="float32", always_2d=True)
        except Exception as e:
            yield rel, None, str(e), ""
            continue
        yield rel, x, sr, text_by_file.get(rel) or text_by_file.get(os.path.basename(rel), "")


def from_hf(name, config, split, audio_col, text_col, token):
    try:
        from datasets import Audio, load_dataset
    except ImportError:
        sys.exit("--hf needs the datasets package: pip install datasets")
    ds = load_dataset(name, config, split=split, streaming=True, token=token)
    ds = ds.cast_column(audio_col, Audio(decode=False))  # decode with soundfile: no torch/torchcodec needed
    for i, row in enumerate(ds):
        a = row[audio_col]
        key = f"{i:06d}"
        try:
            src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
            x, sr = sf.read(src, dtype="float32", always_2d=True)
        except Exception as e:
            yield key, None, str(e), ""
            continue
        yield key, x, sr, str(row.get(text_col) or "") if text_col else ""


_W = {}


def _init_worker(threads, rules):
    _W["vad"], _W["mos"], _W["rules"] = SileroVAD(threads), DNSMOS(threads), rules


def _analyse(m):
    """Worker: 16 kHz mono -> (vad result, (sig, bak, ovrl) of the speech span).
    DNSMOS (the slow part) is skipped when VAD alone already fails the clip."""
    res = _W["vad"].analyse(m)
    if res is None:
        return None, None
    s, e, ratio, pause = res
    min_dur, pad, max_pause, min_ratio = _W["rules"]
    if min(len(m) / SR, e + pad) - max(0.0, s - pad) < min_dur or pause > max_pause or ratio < min_ratio:
        return res, None
    return res, _W["mos"].score(m[int(s * SR): int(e * SR)])


def to_mono16k(x, sr):
    m = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        m = resample_poly(m, SR // g, int(sr) // g).astype(np.float32)
    return m


def safe(s, n=40):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:n].strip("_") or "clip"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--folder", help="folder of audio clips (searched recursively)")
    src.add_argument("--hf", help="Hugging Face dataset id with an audio column")
    ap.add_argument("--config", default=None, help="HF dataset config / subset")
    ap.add_argument("--split", default="train", help="HF split (default: train)")
    ap.add_argument("--audio-column", default="audio")
    ap.add_argument("--text-column", default=None, help="HF column (or --meta column) shown as the clip's text")
    ap.add_argument("--meta", help="for --folder: optional .csv/.json with a 'file' column and a text column")
    ap.add_argument("--token", default=None, help="HF token for gated/private datasets (default: your login)")
    ap.add_argument("--limit", type=int, default=0, help="stop after scanning this many clips (0 = all)")
    ap.add_argument("--top", type=int, default=200, help="how many clips to shortlist (default: 200)")
    ap.add_argument("--out", default="shortlist", help="output folder (default: shortlist)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--min-duration", type=float, default=3.0, help="min speech length after edge trim (default 3)")
    ap.add_argument("--max-duration", type=float, default=10.0,
                    help="max length after edge trim (default 10); longer clips are still shortlisted if they "
                         "pass everything else; the review page makes you crop them before they can pass")
    ap.add_argument("--reject-longer-than", type=float, default=30.0,
                    help="hard limit: skip clips longer than this outright (default 30)")
    ap.add_argument("--max-pause", type=float, default=0.4, help="longest pause allowed inside the speech (default 0.4 s)")
    ap.add_argument("--min-speech-ratio", type=float, default=0.75, help="share of the trimmed clip that must be speech")
    ap.add_argument("--max-clipping", type=float, default=0.001, help="max share of samples at full scale")
    ap.add_argument("--min-ovrl", type=float, default=0.0, help="optional DNSMOS OVRL floor (1-5)")
    ap.add_argument("--rank-by", choices=["ovrl", "bak", "sig", "bak+ovrl"], default="bak+ovrl",
                    help="ranking score among clips that pass (default: bak+ovrl, favours clean backgrounds)")
    ap.add_argument("--pad", type=float, default=0.15, help="silence kept around the speech when edge-trimming")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="parallel CPU worker processes (default: all cores)")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)
    for name in MODELS:  # download once, before the workers start
        model_path(name)

    if a.folder:
        text_by_file = {}
        if a.meta:
            sys.path.insert(0, HERE)
            from build import load_meta
            col = a.text_column or "text"
            text_by_file = {str(r.get("file", "")): str(r.get(col) or "") for r in load_meta(a.meta)}
        source = from_folder(a.folder, text_by_file)
        origin = os.path.abspath(a.folder)
    else:
        source = from_hf(a.hf, a.config, a.split, a.audio_column, a.text_column, a.token)
        origin = f"hf://{a.hf}" + (f"/{a.config}" if a.config else "") + f"[{a.split}]"

    rank_key = {"ovrl": lambda r: r["ovrl"], "bak": lambda r: r["bak"], "sig": lambda r: r["sig"],
                "bak+ovrl": lambda r: r["bak"] + r["ovrl"]}[a.rank_by]
    rows, heap, t0, seq = [], [], time.time(), 0

    def prepared():
        """Decode + resample in the main process; yields (row, x, sr, mono16k or None)."""
        for key, x, sr, text in source:
            if a.limit and len(rows) >= a.limit:
                return
            row = {"key": key, "text": text, "duration": "", "trimmed_duration": "", "speech_ratio": "",
                   "max_pause": "", "clipping": "", "sig": "", "bak": "", "ovrl": "", "verdict": "", "reason": "",
                   "rank": ""}
            rows.append(row)
            if x is None:
                row.update(verdict="error", reason=f"unreadable: {sr}"[:120])
                continue
            dur = len(x) / sr
            row["duration"] = round(dur, 2)
            if dur > a.reject_longer_than:
                row.update(verdict="reject", reason=f"longer than {a.reject_longer_than:g}s")
                continue
            yield row, x, sr, to_mono16k(x, sr)

    import itertools
    import multiprocessing as mp
    pending = prepared()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker,
                                         initargs=(1, (a.min_duration, a.pad, a.max_pause, a.min_speech_ratio))) as pool:
        while True:
            batch = list(itertools.islice(pending, a.workers * 8))
            if not batch:
                break
            for (row, x, sr, m), (res, scores) in zip(batch, pool.map(_analyse, [b[3] for b in batch])):
                if res is None:
                    row.update(verdict="reject", reason="no speech")
                    continue
                dur = len(x) / sr
                s, e, ratio, pause = res
                sig, bak, ovrl = scores or ("", "", "")
                s, e = max(0.0, s - a.pad), min(dur, e + a.pad)
                clipping = float(np.mean(np.abs(x[int(s * sr): int(e * sr)]) >= 0.999))
                row.update(trimmed_duration=round(e - s, 2), speech_ratio=round(ratio, 2), max_pause=round(pause, 2),
                           clipping=round(clipping, 5))
                if scores:
                    row.update(sig=round(sig, 2), bak=round(bak, 2), ovrl=round(ovrl, 2))
                why = []
                if e - s < a.min_duration:
                    why.append(f"under {a.min_duration:g}s")
                if pause > a.max_pause:
                    why.append(f"pause {pause:.2f}s")
                if ratio < a.min_speech_ratio:
                    why.append(f"speech {ratio:.0%}")
                if clipping > a.max_clipping:
                    why.append("clipping")
                if scores and ovrl < a.min_ovrl:
                    why.append(f"OVRL {ovrl:.2f}")
                if why:
                    row.update(verdict="reject", reason="; ".join(why))
                    continue
                row["verdict"] = "pass"
                score = rank_key(row)
                item = (score, seq, row, x[int(s * sr): int(e * sr)], sr)
                seq += 1
                if len(heap) < a.top:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)
            el = time.time() - t0
            print(f"scanned {len(rows)} ({len(rows) / el:.1f} clips/s), "
                  f"{sum(r['verdict'] == 'pass' for r in rows)} pass", flush=True)

    best = sorted(heap, key=lambda it: -it[0])
    meta = []
    for rank, (score, _, row, clip, sr) in enumerate(best, 1):
        name = f"{rank:04d}_{safe(row['key'])}.wav"
        sf.write(os.path.join(a.out, "audio", name), clip, sr, subtype="PCM_16")
        row["rank"] = rank
        meta.append({"file": name, "text": row["text"], "subtitle": f"source: {row['key']}",
                     "tags": [f"rank {rank}", f"OVRL {row['ovrl']}", f"BAK {row['bak']}"],
                     "notes": (f"DNSMOS SIG {row['sig']} · BAK {row['bak']} · OVRL {row['ovrl']} · "
                               f"speech {row['speech_ratio']:.0%} · longest pause {row['max_pause']}s · "
                               f"trimmed {row['duration']}s → {row['trimmed_duration']}s")})
    for r in rows:
        if r["verdict"] == "pass" and not r["rank"]:
            r["verdict"] = "pass (not in top)"

    with open(os.path.join(a.out, "screening.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["key"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "shortlist.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    el = time.time() - t0
    passed = sum(r["verdict"].startswith("pass") for r in rows)
    print(f"scanned {len(rows)} clips in {el:.0f}s ({len(rows) / max(el, 1e-6):.1f}/s): {passed} passed, "
          f"{len(meta)} shortlisted -> {a.out}/audio")

    if meta:
        title = a.title or f"Shortlist: top {len(meta)} of {len(rows)} from {origin}"
        subprocess.run([sys.executable, os.path.join(HERE, "build.py"), os.path.join(a.out, "audio"),
                        "--meta", os.path.join(a.out, "shortlist.json"), "--out", a.out, "--title", title,
                        "--min-duration", str(a.min_duration), "--max-duration", str(a.max_duration)], check=True)


if __name__ == "__main__":
    main()
