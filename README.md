# Kikitori

Long recordings (meetings, calls, lectures) to text, with a speaker on every line and a timestamp on every sentence.
Chinese, English and Japanese. Recognition is sent over HTTP to a `llama-server` running Qwen3-ASR, the same server
[Kakitori](https://github.com/hanepudding/kakitori) dictates through; voice activity detection, speaker diarization
and alignment run on this machine.

This was vibe coded and is kept deliberately small. If you are reading it yourself, the short version is:

1. Run a Qwen3-ASR `llama-server` (below). If you already run Kakitori, use that one.
2. Accept the licence of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   and create a Hugging Face token.
3. Install, copy `.env.example` to `.env` with the token filled in, and copy `vocab.example.txt` to `vocab.txt`.
4. Run `python serve.py` and open http://127.0.0.1:8090, or set `ASR_SOURCE` in `.env` and run `python run.py`.

## What runs where

| Step | Model | Runs on |
|---|---|---|
| Recognition | `ggml-org/Qwen3-ASR-1.7B-GGUF` | the llama-server |
| Speech detection | silero-vad | this machine |
| Speaker diarization | `pyannote/speaker-diarization-community-1` | this machine; needs the HF token and licence |
| Alignment | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | this machine |

Qwen3-ASR returns a chunk's text with no timestamps. The aligner puts every character of it back on the timeline:
that gives each sentence its timestamps, and lets a chunk that holds two speakers be split at the character where
the speaker changes. llama.cpp does not run the aligner, so it runs here, through PyTorch. The local models go on
Apple Silicon's GPU on macOS, on CUDA elsewhere when available, and on the CPU otherwise. The first run downloads
about 1.7 GB of their weights.

## Server

```
llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0 --host 127.0.0.1 --port 8080 -ngl 99 -np 8 -c 16384 --no-webui
```

Kikitori sends `ASR_BATCH_SIZE` chunks at once, and llama-server decodes as many in parallel as it has slots (`-np`),
each slot getting the context divided by the slot count; a 45 s chunk fits in 2048 tokens. Kakitori's single-slot
server works unchanged, it just decodes one chunk at a time. Eight slots of 2048 hold about 1.6 GB more than one
(Kakitori's README has the measurements).

## Install

`ffmpeg` has to be on `PATH`. On Windows, PyPI's `torch` is CPU-only: install the CUDA build from pytorch.org first.
Then, from the project directory, into the machine's own interpreter:

```
uv pip install -e ".[app]"
cp .env.example .env
cp vocab.example.txt vocab.txt
```

`.env.example` documents every setting. `ASR_DIARIZE=0` skips diarization and needs no token.

## Use

### Web page

`python serve.py`, then open http://127.0.0.1:8090. Drop a file on the page; uploads queue and are transcribed one at
a time. Language, number of speakers, background and vocabulary can be set per upload and default to `.env`. On a
finished transcript, a click on a timestamp plays the audio from there, the speaker fields rename the speakers, and
the download is the `.txt` with those names in it.

Each job is a directory under `SERVE_JOBS_DIR`, and the job list is those directories. The upload waits there
until its job ends, whether done or failed, and is then removed; what stays is `job.json`, `segments.json` and
`audio.m4a`, a 48 kbps copy for the player at about 20 MB an hour. Delete on the page, or
`DELETE /api/jobs/<id>`, removes a job with its directory; a running job cannot be deleted. A job the server was
working on when it stopped starts over on the next launch.

The server has no authentication. It binds `127.0.0.1`; set `SERVE_HOST=0.0.0.0` only on a network you trust.
The page requests everything by relative path, so a reverse proxy can mount it under a sub-path such as
`/kikitori/`, trailing slash included.

### In the background

On Windows, `windows/kikitori.ps1` registers `serve.py` as the scheduled task `kikitori`, started at logon and every
hour. A task has no console, so set `SERVE_LOG_FILE` in `.env` first. Run the script once from a PowerShell that is
not elevated: the interpreter it records is whatever `python` resolves to there, and uv's interpreter links do not
resolve in an elevated shell. `Start-ScheduledTask` and `Stop-ScheduledTask` control it; to keep it off for a while,
disable the task, since the hourly trigger restarts an ended one. With `SERVE_HOST=0.0.0.0`, other machines also need an inbound
firewall rule for `SERVE_PORT`; a process without a desktop never raises the allow prompt, so its packets are
dropped silently.

On macOS, put your interpreter, this directory and the log path into `launchd/local.kikitori.plist`, copy it to
`~/Library/LaunchAgents/`, then `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/local.kikitori.plist`. launchd
runs it at login and whenever it exits. Reload with `launchctl kickstart -k gui/$UID/local.kikitori` after editing
`.env` or the plist.

### run.py

Set `ASR_SOURCE` in `.env` and run `python run.py`. The transcript goes to `ASR_OUT_DIR` as `<folder>_<stem>.txt`:
file names such as `macro.mp4` repeat across folders, and keying on the file name alone would overwrite one
transcript with another. After a first run shows who is who, `ASR_SPEAKER_NAMES` renames the speakers.

### As a library

Install into another project's interpreter with `uv pip install -e path/to/kikitori` (the `app` extra is only for
the two scripts above):

```python
from kikitori import transcribe, transcribe_file

# writes out/call.txt and returns its path
transcribe_file("call.mp4", "out", "call", vocab_files=["vocab.txt"], hf_token="hf_...",
                server="http://127.0.0.1:8080")

# list[Segment], each with start, end, text and speaker; nothing is written to disk
segments = transcribe("call.mp4", hf_token=None)
```

### Output

One line per segment, speaker omitted when diarization is off:

```
[00:01:02.340 --> 00:01:05.120]  [speaker_SPEAKER_00]  text
```

## How it works

```
kikitori/audio.py      ffmpeg decode to 16 kHz mono
kikitori/vad.py        silero-vad finds the speech regions
kikitori/diarize.py    pyannote speaker diarization
kikitori/segment.py    packs speech regions into the chunks sent to the model
kikitori/asr.py        sends chunks to llama-server for recognition
kikitori/align.py      Qwen3-ForcedAligner maps the text back onto the timeline
kikitori/attribute.py  assigns a speaker to each word; breaks lines at speaker changes and sentence ends
kikitori/vocab.py      vocabulary -> biasing prompt
kikitori/output.py     writes the timestamped text
kikitori/device.py     picks the accelerator by platform
kikitori/pipeline.py   chains all of the above into transcribe() and transcribe_file()
run.py                 transcribes one file, configured through .env
serve.py, web/         the web page
settings.py            reads .env for run.py and serve.py
windows/, launchd/     serve.py as a background service
```

### Chunks and speakers

Recognition and line breaking are two separate steps.

A chunk sent to the model is constrained only by length and silence. Chunks are kept as long as possible and may
hold two people talking, so the model sees a whole question and answer. Chunk boundaries fall only on silences
found by VAD, never inside a word. Qwen3-ASR carries no state from one chunk to the next, and consistency across
chunks rests on the vocabulary alone, so shorter chunks directly cost accuracy.

Speakers are assigned after recognition, on the word timeline, and lines are broken only then: at a speaker
change, and at sentence-final punctuation once the line is longer than `ASR_MIN_SEG_SEC`. A cut on the timeline
costs nothing, so lines can be much finer than chunks. A pause between lines shorter than `ASR_GAP_FILL_SEC`
merges into the previous line. A speaker run too short to be a real turn is treated as diarization jitter and
merged into a neighbour; that threshold is `min_turn_sec` in `attribute.to_segments`, not in `.env`.

### Vocabulary

Vocabulary files hold one term per line, with `#` starting a comment. Every chunk carries them (context biasing):
names and domain terms get corrected by the list, not by switching models. The terms are joined with the
ideographic comma, closed with the ideographic full stop, prefixed with `Vocabulary: `, and placed in the system
message after `ASR_BACKGROUND`. A long list makes the model write in terms nobody said, so keep only what the
recordings actually mention.

A chunk the model cannot hear sometimes comes back as the vocabulary list itself. Such chunks are decoded again
without it, and the log says `N chunks echoed the vocabulary, re-decoding without it`; that is not an error.

### Failures

`the timestamp head collapsed` means the aligner placed a whole chunk's text at a single instant, which is what
text the model produced without hearing it looks like. The file fails rather than yield a segment that spans the
chunk and reads like real speech; it happens after recognition, so the whole file has to be run again.
