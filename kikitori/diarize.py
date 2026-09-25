import numpy as np
import torch

from . import device
from .audio import SAMPLE_RATE
from .types import Span

PIPELINE_ID = "pyannote/speaker-diarization-community-1"


def diarize(
    samples: np.ndarray,
    hf_token: str,
    pipeline_id: str = PIPELINE_ID,
    num_speakers: int | None = None,
) -> list[Span]:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    pipeline = Pipeline.from_pretrained(pipeline_id, token=hf_token)
    if pipeline is None:
        raise ValueError(f"failed to load {pipeline_id} — check the HF token and licence acceptance")
    pipeline.to(torch.device(device.best()))

    # Hand pyannote the decoded waveform, not the file. Given a path it decodes
    # through torchcodec, which needs FFmpeg's shared libraries on the dynamic
    # loader path; Homebrew's are not on it, so macOS fails at this call.
    audio = {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": SAMPLE_RATE}
    with ProgressHook() as hook:
        output = pipeline(audio, hook=hook, num_speakers=num_speakers)

    return [
        Span(turn.start, turn.end, f"speaker_{speaker}")
        for turn, speaker in output.speaker_diarization
    ]
