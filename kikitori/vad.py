import numpy as np
import torch

from .audio import SAMPLE_RATE
from .types import Span


def detect_speech(
    samples: np.ndarray,
    min_silence_ms: int = 400,
    speech_pad_ms: int = 200,
    threshold: float = 0.5,
) -> list[Span]:
    """Find speech regions with silero-vad.

    min_silence_ms is how long a pause must be to end a region. Below ~300 it
    breaks at breathing pauses inside a sentence, which `segment.pack` can only
    glue back together, never undo.
    """
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    stamps = get_speech_timestamps(
        torch.from_numpy(samples),
        model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        threshold=threshold,
        return_seconds=True,
    )
    return [Span(s["start"], s["end"]) for s in stamps]
