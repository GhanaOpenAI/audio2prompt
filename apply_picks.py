#!/usr/bin/env python3
"""Export the clips you rated "good" (with your trims and cuts applied) from a picks.json.

    python apply_picks.py picks.json path/to/audio exported/
    python apply_picks.py picks.json path/to/audio exported/ --include-unrated

Files keep their relative paths and audio format. A manifest.csv lists what was written.
"""
import argparse
import csv
import json
import os

import numpy as np
import soundfile as sf

FADE = 0.01  # seconds of crossfade where a cut joins two pieces (avoids clicks)


def keep_segments(start, end, cuts):
    """The [start, end) window minus the cut ranges -> list of (a, b) seconds to keep."""
    segs, pos = [], start
    for a, b in sorted(cuts):
        a, b = max(a, start), min(b, end)
        if b <= a:
            continue
        if a > pos:
            segs.append((pos, a))
        pos = max(pos, b)
    if end > pos:
        segs.append((pos, end))
    return segs


def silence_plan(speech, dur, min_gap=0.3, keep=0.15, pad=0.1):
    """Speech regions [(a, b), ...] -> the (a, b) pieces to keep once silences are removed:
    silent ends trimmed to `pad`, every inner pause of `min_gap` or more shortened to `keep`."""
    if not speech:
        return []
    start, end = max(0.0, speech[0][0] - pad), min(dur, speech[-1][1] + pad)
    segs, pos = [], start
    for (_, a), (b, _) in zip(speech[:-1], speech[1:]):
        if b - a >= min_gap:
            segs.append((pos, a + keep / 2))
            pos = b - keep / 2
    segs.append((pos, end))
    return segs


def splice(data, sr, segs):
    """Concatenate the kept pieces with a short equal-power crossfade at each join."""
    out = None
    n = int(FADE * sr)
    for a, b in segs:
        piece = data[int(round(a * sr)): int(round(b * sr))]
        if out is None:
            out = piece
            continue
        k = min(n, len(out), len(piece))
        if k > 0:
            ramp = np.linspace(0, np.pi / 2, k)[:, None]
            mixed = out[-k:] * np.cos(ramp) + piece[:k] * np.sin(ramp)
            out = np.concatenate([out[:-k], mixed, piece[k:]])
        else:
            out = np.concatenate([out, piece])
    return out if out is not None else data[:0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picks", help="picks.json downloaded from the review page")
    ap.add_argument("audio_dir", help="the same audio folder you passed to build.py")
    ap.add_argument("out_dir", help="where to write the selected (trimmed) clips")
    ap.add_argument("--include-unrated", action="store_true", help="also export clips you did not rate")
    ap.add_argument("--min-duration", type=float, default=None,
                    help="skip clips shorter than this after trimming (default: the minimum saved in picks.json)")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="skip clips longer than this after trimming (default: the maximum saved in picks.json)")
    a = ap.parse_args()

    picks = json.load(open(a.picks, encoding="utf-8"))
    files = list(picks.get("good", [])) + (list(picks.get("unrated", [])) if a.include_unrated else [])
    trims = picks.get("trims", {})
    cuts = picks.get("cuts", {})
    min_dur = a.min_duration if a.min_duration is not None else float(picks.get("min_duration") or 0)
    max_dur = a.max_duration if a.max_duration is not None else float(picks.get("max_duration") or 0)
    os.makedirs(a.out_dir, exist_ok=True)

    rows, too_short, too_long = [], [], []
    for rel in files:
        src = os.path.join(a.audio_dir, rel)
        dst = os.path.join(a.out_dir, rel)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        info = sf.info(src)
        data, sr = sf.read(src, always_2d=True)
        start, end = trims.get(rel, [0.0, len(data) / sr])
        segs = keep_segments(start, end, cuts.get(rel, []))
        kept = sum(b - a for a, b in segs)  # same measure the review page enforces (before crossfades)
        if min_dur > 0 and kept < min_dur - 1e-6:
            too_short.append(f"{rel} ({kept:.2f}s)")
            continue
        if max_dur > 0 and kept > max_dur + 1e-6:
            too_long.append(f"{rel} ({kept:.2f}s)")
            continue
        data = splice(data, sr, segs)
        sf.write(dst, data, sr, subtype=info.subtype, format=info.format)
        rows.append({"file": rel, "start_sec": round(start, 3), "end_sec": round(end, 3),
                     "cuts": json.dumps(cuts.get(rel, [])), "duration_sec": round(len(data) / sr, 3),
                     "trimmed": rel in trims or bool(cuts.get(rel))})

    with open(os.path.join(a.out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "start_sec", "end_sec", "cuts", "duration_sec", "trimmed"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} clips ({sum(r['trimmed'] for r in rows)} trimmed or cut) to {a.out_dir}")
    if too_short:
        print(f"skipped {len(too_short)} clips shorter than {min_dur:g}s: " + ", ".join(too_short))
    if too_long:
        print(f"skipped {len(too_long)} clips longer than {max_dur:g}s: " + ", ".join(too_long))


if __name__ == "__main__":
    main()
