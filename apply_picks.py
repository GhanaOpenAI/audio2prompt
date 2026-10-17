#!/usr/bin/env python3
"""Export the clips you rated "good" (with your trims applied) from a picks.json.

    python apply_picks.py picks.json path/to/audio exported/
    python apply_picks.py picks.json path/to/audio exported/ --include-unrated

Files keep their relative paths and audio format. A manifest.csv lists what was written.
"""
import argparse
import csv
import json
import os

import soundfile as sf


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
        data = data[int(round(start * sr)): int(round(end * sr))]
        if min_dur > 0 and len(data) / sr < min_dur - 1e-6:
            too_short.append(f"{rel} ({len(data) / sr:.2f}s)")
            continue
        if max_dur > 0 and len(data) / sr > max_dur + 1e-6:
            too_long.append(f"{rel} ({len(data) / sr:.2f}s)")
            continue
        sf.write(dst, data, sr, subtype=info.subtype, format=info.format)
        rows.append({"file": rel, "start_sec": round(start, 3), "end_sec": round(end, 3),
                     "duration_sec": round(len(data) / sr, 3), "trimmed": rel in trims})

    with open(os.path.join(a.out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "start_sec", "end_sec", "duration_sec", "trimmed"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} clips ({sum(r['trimmed'] for r in rows)} trimmed) to {a.out_dir}")
    if too_short:
        print(f"skipped {len(too_short)} clips shorter than {min_dur:g}s: " + ", ".join(too_short))
    if too_long:
        print(f"skipped {len(too_long)} clips longer than {max_dur:g}s: " + ", ".join(too_long))


if __name__ == "__main__":
    main()
