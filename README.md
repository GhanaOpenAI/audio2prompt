# audio2prompt

Turn a large pool of speech (a folder or a Hugging Face dataset) into clean, ear-checked prompt clips
for zero-shot TTS / voice cloning:

1. **Screen** (`screen.py`): score every clip from a folder or a Hugging Face dataset with two tiny
   CPU models and shortlist the top N (default 200).
2. **Review** (`build.py`): silences are removed first and clips without enough speech are dropped;
   then you listen in a single local HTML page, mark each clip **good** or **bad**, and trim or cut
   anything still wrong.
3. **Export** (`apply_picks.py`): write out the good clips with your trims and cuts applied.

Built for curating speech datasets (picking clean voice-cloning reference clips, spot-checking
TTS/ASR data), but the reviewer works for any short audio.

![Reviewer screenshot](docs/screenshot.png)

## Screening a big pool (CPU only)

```bash
pip install -r requirements.txt            # + `pip install datasets` for Hugging Face input

python screen.py --folder path/to/clips --top 200
python screen.py --hf user/dataset --split train --audio-column audio --text-column text --top 200
# -> shortlist/index.html (review page), shortlist/audio/, shortlist/screening.csv
```

Each clip goes through two small ONNX models, with no PyTorch and no GPU. They're downloaded once to
`~/.cache/audio2prompt` and checksum-verified:

| model | size | what it measures |
|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) v5 | 2.3 MB | where the speech is: trims silence off both ends, longest pause inside the speech, share of the clip that is speech |
| [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) P.835 | 1.2 MB | predicted listener scores (1–5): **SIG** speech quality, **BAK** background (higher = cleaner), **OVRL** overall |

Length is measured **without silences**, the way the review page presents a clip: silent ends
trimmed and every pause of `--min-gap` 0.3 s or more shortened to `--keep-gap` 0.15 s. Clips are
**rejected** if that cleaned length is under `--min-duration` (3 s), if less than `--min-speech-ratio`
(75 %) of it is speech, or if they clip. DNSMOS only runs on clips that survive the VAD checks, which is what keeps it fast.
The rest are **ranked** by `--rank-by` (default `bak+ovrl`, favouring clean backgrounds) and the top
`--top` are written edge-trimmed to `shortlist/audio/` with a ready-to-use review page.
`screening.csv` lists every clip scanned with its scores and the reason for any rejection.

Speed: about **24 clips/s on a 6-core laptop CPU** for typical 2–8 s clips (folder input), roughly
85,000 clips an hour. Hugging Face input is usually limited by download speed.

Clips longer than `--max-duration` (10 s) can still be shortlisted; the review page makes you trim
them before they can pass. `--reject-longer-than` (30 s) drops very long clips outright.

## Quick start

```bash
git clone https://github.com/GhanaOpenAI/audio2prompt.git
cd audio2prompt
pip install -r requirements.txt          # numpy, soundfile

# try it on the synthetic demo clips
python example/make_demo.py
python build.py example/audio --meta example/clips.csv --out example/review --title "Demo review"
# open example/review/index.html in your browser
```

On your own data:

```bash
python build.py path/to/clips                       # -> review/index.html
python build.py path/to/clips --meta clips.csv      # show transcripts / notes next to each clip
python build.py path/to/clips --copy --out review   # copies audio in, so you can zip and share the folder
python build.py path/to/clips --min-duration 2 --max-duration 15   # allowed kept length (defaults 3 and 10 s, 0 = off)
```

## Reviewing

| Key | Action |
|---|---|
| `Space` / `Enter` | play / pause |
| `G` / `B` | mark good / bad, then jump to the next clip and play it |
| `U` | clear the rating |
| `↓` `J` / `↑` `K` | next / previous clip |
| `R` | replay |
| `[` / `]` | set trim start / end at the playhead |
| `A` | remove the detected silences in this clip (if you undid the automatic pass) |
| `Delete` / `C` | with both markers around a middle part: cut that part out |
| `X` | undo all trims and cuts on the clip |

- **You review clips with their silences already removed.** `build.py` finds the speech in every clip
  with Silero VAD (CPU, the same model `screen.py` uses), trims the silent ends (keeping `--edge-pad`
  0.1 s) and shortens every pause of `--min-gap` 0.3 s or more *in the middle* to a natural
  `--keep-gap` 0.15 s. The cleaned audio is written to `<out>/clips/` and that's what the page shows
  and plays.
- **Only viable clips are shown.** A clip's real length is its length without silences. Clips under
  `--min-duration` once silences are removed, or with no speech at all, are left out of the page
  entirely; `<out>/precut.csv` lists every clip with its original and cleaned length and why it was
  hidden. Clips over `--max-duration` are shown, but must be trimmed before they can pass.
- Prefer the original audio? `--no-precut` shows it unmodified and removes silences in the page
  instead (each detected silence is a dotted box; `A` or "remove N silences" applies them, `X` undoes).
  `--silences off` skips detection altogether.
- **Trimming the ends:** drag the orange handles on a clip's waveform. Trimmed-off parts are shaded.
- **Cutting something out of the middle** (a cough, a stray word, a pause) uses the same orange
  markers: move both of them around the part you don't want. Under the waveform,
  **✂ Cut this part out instead** appears; click it (or press `Delete` / `C`) and that part is cut, and the
  markers jump back to the ends, ready for the next cut. If you don't click it, the markers simply
  trim, keeping only the part between them.
  Cut ranges stay visible, crossed out; click one to restore it. On export the pieces are joined with
  a 10 ms crossfade so the join doesn't click.
- **Playback only covers what you're keeping**: it starts at the trim start, jumps over cuts and stops
  at the trim end, so you hear exactly what will be exported.
- **Transcript and label for every clip you pass:** marking a clip ✓ good opens a transcript box
  (pre-filled with the clip's text, if any) and a label field. Labels are free text, so you don't
  need any loaded, but every label you type becomes a choice in the field's dropdown, so you only
  type it once. Pre-fill labels from your metadata with `--label-col` (default `label`).
  - Under the boxes, a character pad has the letters of Ghanaian orthographies and IPA that most
    keyboards lack (ɛ ɔ ŋ ɖ ƒ ɣ ʋ ɩ ʊ ʒ ɲ, Hausa ɓ ɗ ƙ ƴ, ə ɪ ɨ ʉ ɑ ʃ ʔ ɡ ɟ ɸ β), ⇧ for capitals, and
    tone/nasal marks (◌́ ◌̀ ◌̂ ◌̌ ◌̃ ◌̄) that attach to the letter before the cursor. Click to insert.
  - Keyboard shortcuts are off while you type in these boxes.
- **Allowed length:** what a clip keeps after trims and cuts must be between `--min-duration` and
  `--max-duration` (defaults **3 s** and **10 s**, the usual range for voice-cloning reference audio).
  - Too short, whether it started short or you trimmed it down: **marked bad automatically**, ✓ disabled.
    Extend the trim back past the minimum and the automatic "bad" is lifted (a "bad" you chose yourself stays).
  - Too long: ✓ stays disabled until you **trim it down** to the maximum or less.
- **Click the waveform** to play from that point.
- **Filters:** All / Unrated / Good / Bad. Use *Unrated* to work through what's left.
- **Notes are hidden by default** so automatic checks don't bias your ears; tick *show notes* to see them.
- **Progress is saved in the browser** (localStorage) as you go, so you can close the page and come back.
  Rebuilding with the same title and files keeps your progress.
- **Download picks** saves `picks.json`; **Load picks** reads one back in (to resume on another machine,
  merge a colleague's review, or carry progress over to a rebuilt page). Keys can be file paths or clip numbers.

```json
{ "good": ["a.wav"], "bad": ["b.wav"], "unrated": ["c.wav"],
  "trims": { "a.wav": [0.52, 4.62] }, "cuts": { "a.wav": [[1.9, 2.4]] },
  "transcripts": { "a.wav": "Me pɛ sɛ wo ba ha" }, "labels": { "a.wav": "greeting" },
  "min_duration": 3.0, "max_duration": 10.0 }
```

## Exporting the good clips

```bash
python apply_picks.py picks.json review/clips selected/                      # good clips, trimmed
python apply_picks.py picks.json review/clips selected/ --include-unrated    # plus the ones you skipped
# (pass the folder the page played from: <out>/clips, or your original folder with --no-precut)
```

Clips keep their relative paths and audio format; `selected/manifest.csv` records each clip's
transcript, label, trim, cuts and final duration. Clips outside the allowed length after trimming are skipped (the limits
saved in `picks.json`, or `--min-duration` / `--max-duration`).

## Metadata file

Optional. A `.csv`, `.json` (list of objects) or `.jsonl` with one row per clip, matched on file
name or relative path:

| column | shown as |
|---|---|
| `file` | the clip to match (required) |
| `text` | bold line, e.g. the transcript |
| `subtitle` | line under it, e.g. a translation |
| `tags` | small labels (`;`-separated in CSV, or a list in JSON) |
| `label` | pre-fills the editable label |
| `notes` | hidden until *show notes* is ticked, e.g. automatic QC results |

Different column names? Map them: `--text-col transcript --subtitle-col english --notes-col qc`.

## Notes

- Supported formats: anything [soundfile](https://python-soundfile.readthedocs.io) can read:
  WAV, FLAC, OGG/Opus, MP3, AIFF.
- The page loads audio by relative path. Without `--copy`, keep the audio where it was when you
  built the page. With `--copy`, the output folder is self-contained.
- No server, no uploads: everything stays on your machine.

## License

MIT. The screening models are downloaded from their own projects at run time and keep their own
licences ([Silero VAD](https://github.com/snakers4/silero-vad/blob/master/LICENSE): MIT;
[DNSMOS](https://github.com/microsoft/DNS-Challenge): see that repository).
