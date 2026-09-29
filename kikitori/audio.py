import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def run_key(source: str | Path) -> str:
    """Name the transcript after the source's folder as well as its file.

    Several calls in the corpus share a stem — FG.m4a under two brokers' FG
    folders, macro.mp4 under one date and another — so a stem-keyed transcript
    overwrites its neighbour.
    """
    source = Path(source)
    return f"{source.parent.name}_{source.stem}"


def decode(source: str | Path) -> np.ndarray:
    """Decode any media file to the 16 kHz mono samples the ASR stack expects, with no file in between."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(source),
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(r.stdout, dtype="<i2").astype(np.float32) / 32768


def clips(samples: np.ndarray, spans, sample_rate: int = SAMPLE_RATE) -> list[np.ndarray]:
    return [samples[int(s.start * sample_rate):int(s.end * sample_rate)] for s in spans]
