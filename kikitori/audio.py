import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def run_key(source: str | Path) -> str:
    """Name artifacts after the source's folder as well as its file.

    Several calls in the corpus share a stem — FG.m4a under two brokers' FG
    folders, macro.mp4 under one date and another — so a stem-keyed wav cache
    transcribes the wrong audio and the transcript overwrites its neighbour.
    """
    source = Path(source)
    return f"{source.parent.name}_{source.stem}"


def to_wav16k(source: str | Path, work_dir: str | Path, name: str | None = None) -> Path:
    """Decode any media file to the 16 kHz mono PCM the ASR stack expects.

    Sources live in other projects' directories, which this pipeline does not
    write to, so the wav goes to work_dir.
    """
    source = Path(source)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{name or run_key(source)}_16k.wav"
    if out.exists():
        return out
    # Convert aside and rename, so an interrupted run cannot leave a truncated
    # wav that the next run reads as cached. -f wav is required: the .part
    # suffix hides the container from ffmpeg's extension sniffing.
    part = out.with_name(out.name + ".part")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source),
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", str(part)],
        check=True, capture_output=True,
    )
    part.replace(out)
    return out


def load(wav: str | Path) -> np.ndarray:
    samples, sr = sf.read(str(wav), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{wav} is {sr} Hz, expected {SAMPLE_RATE}; run to_wav16k first")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples


def clips(samples: np.ndarray, spans, sample_rate: int = SAMPLE_RATE) -> list[np.ndarray]:
    return [samples[int(s.start * sample_rate):int(s.end * sample_rate)] for s in spans]
