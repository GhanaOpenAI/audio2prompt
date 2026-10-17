# audio-clip-reviewer

Listen to a folder of audio clips, mark each one **good** or **bad**, and **trim** junk off the
start or end, all in a single HTML page that runs locally in your browser. Then export the good
clips, cut to your trims.

Built for curating speech datasets (picking clean voice-cloning reference clips, spot-checking
TTS/ASR data), but it works for any short audio.

![Reviewer screenshot](docs/screenshot.png)

## Quick start

```bash
git clone https://github.com/GhanaOpenAI/audio-clip-reviewer.git
cd audio-clip-reviewer
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
| `X` | undo trim |

- **Trimming:** drag the orange handles on a clip's waveform. Trimmed-off parts are shaded, and
  playback only covers what you're keeping, so you hear exactly what will be exported.
- **Click the waveform** to play from that point.
- **Filters:** All / Unrated / Good / Bad. Use *Unrated* to work through what's left.
- **Notes are hidden by default** so automatic checks don't bias your ears; tick *show notes* to see them.
- **Progress is saved in the browser** (localStorage) as you go, so you can close the page and come back.
  Rebuilding with the same title and files keeps your progress.
- **Download picks** saves `picks.json`:

```json
{ "good": ["a.wav"], "bad": ["b.wav"], "unrated": ["c.wav"], "trims": { "a.wav": [0.52, 2.62] } }
```

## Exporting the good clips

```bash
python apply_picks.py picks.json path/to/clips selected/                     # good clips, trimmed
python apply_picks.py picks.json path/to/clips selected/ --include-unrated   # plus the ones you skipped
```

Clips keep their relative paths and audio format; `selected/manifest.csv` records each clip's
trim and final duration.

## Metadata file

Optional. A `.csv`, `.json` (list of objects) or `.jsonl` with one row per clip, matched on file
name or relative path:

| column | shown as |
|---|---|
| `file` | the clip to match (required) |
| `text` | bold line, e.g. the transcript |
| `subtitle` | line under it, e.g. a translation |
| `tags` | small labels (`;`-separated in CSV, or a list in JSON) |
| `notes` | hidden until *show notes* is ticked, e.g. automatic QC results |

Different column names? Map them: `--text-col transcript --subtitle-col english --notes-col qc`.

## Notes

- Supported formats: anything [soundfile](https://python-soundfile.readthedocs.io) can read:
  WAV, FLAC, OGG/Opus, MP3, AIFF.
- The page loads audio by relative path. Without `--copy`, keep the audio where it was when you
  built the page. With `--copy`, the output folder is self-contained.
- No server, no uploads: everything stays on your machine.

## License

MIT
