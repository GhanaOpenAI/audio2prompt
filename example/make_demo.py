#!/usr/bin/env python3
"""Generate a few synthetic clips (tone 'syllables' with junk at the edges) to try the reviewer on."""
import csv, os
import numpy as np
import soundfile as sf

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def syllables(n, f0):
    out = []
    for _ in range(n):
        d = rng.uniform(.12, .28); t = np.arange(int(d * SR)) / SR
        f = f0 * rng.uniform(.85, 1.2)
        tone = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6)) * np.hanning(len(t))
        out += [tone * rng.uniform(.4, .8), np.zeros(int(rng.uniform(.03, .09) * SR))]
    return np.concatenate(out)


CLIPS = [
    ("clean_short.wav", "A clean short line", 6, 140, None, None),
    ("click_at_start.wav", "Has a click at the start", 8, 210, "click", None),
    ("noise_at_end.wav", "Trailing noise at the end", 7, 120, None, "noise"),
    ("long_pause.wav", "Pause in the middle", 10, 180, None, None),
    ("both_ends.wav", "Junk at both ends", 9, 160, "noise", "click"),
    ("clean_long.wav", "A longer clean line", 14, 130, None, None),
]

os.makedirs(os.path.join(HERE, "audio"), exist_ok=True)
rows = []
for name, text, n, f0, head, tail in CLIPS:
    x = syllables(n, f0)
    if name == "long_pause.wav":
        mid = len(x) // 2
        x = np.concatenate([x[:mid], np.zeros(int(.8 * SR)), x[mid:]])
    junk = {"click": lambda: np.concatenate([np.zeros(800), np.r_[1.0, -0.8, 0.6], np.zeros(4000)]),
            "noise": lambda: rng.normal(0, .15, int(.5 * SR))}
    x = np.concatenate([junk[head]() if head else np.zeros(1600), x, junk[tail]() if tail else np.zeros(1600)])
    sf.write(os.path.join(HERE, "audio", name), (x / np.abs(x).max() * .9).astype(np.float32), SR, subtype="PCM_16")
    rows.append({"file": name, "text": text, "subtitle": f"synthetic demo · {n} syllables",
                 "tags": "demo;" + ("needs trim" if head or tail else "clean"),
                 "notes": f"start: {head or 'clean'} · end: {tail or 'clean'}"})

with open(os.path.join(HERE, "clips.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file", "text", "subtitle", "tags", "notes"])
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} demo clips to {HERE}/audio")
