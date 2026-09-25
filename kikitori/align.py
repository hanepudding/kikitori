from dataclasses import dataclass

import numpy as np
import torch

from . import device
from .audio import SAMPLE_RATE

MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"


@dataclass(frozen=True)
class Word:
    """One aligned unit — a character for Chinese, a whole word for space-delimited scripts.

    Times are relative to the start of the clip that was aligned, not the source file.
    """
    text: str
    start: float
    end: float


class ForcedAligner:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device_map: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 4,
    ) -> None:
        device_map = device_map or device.best()
        from transformers import AutoProcessor, Qwen3ASRForTokenClassification

        # The `-hf` suffix is load-bearing. The repo without it is the original Qwen
        # format: every tensor is named thinker.* and the config nests under
        # thinker_config, so transformers matches zero keys, keeps the random init
        # without complaint, and dies much later on a hidden-size mismatch whose
        # message prints the two token counts, which are equal.
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3ASRForTokenClassification.from_pretrained(
            model_id, device_map=device_map, dtype=dtype,
        )
        self.batch_size = batch_size

    def align(
        self,
        clips: list[np.ndarray],
        texts: list[str],
        language: str = "Chinese",
    ) -> list[list[Word]]:
        out: list[list[Word]] = []
        for i in range(0, len(clips), self.batch_size):
            audio = clips[i:i + self.batch_size]
            transcript = texts[i:i + self.batch_size]
            inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
                audio=audio, transcript=transcript, language=language,
            )
            inputs = inputs.to(self.model.device, self.model.dtype)
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            decoded = self.processor.decode_forced_alignment(
                logits, inputs["input_ids"], word_lists,
                self.model.config.timestamp_token_id,
            )
            for j, (clip, words) in enumerate(zip(audio, decoded)):
                _check(words, len(clip) / SAMPLE_RATE, i + j, transcript[j])
            out += [[Word(w["text"], w["start_time"], w["end_time"]) for w in words]
                    for words in decoded]
        return out


def _check(words: list[dict], clip_sec: float, index: int, text: str) -> None:
    """Reject an alignment whose timestamps collapsed.

    Text the model produced without hearing it — a transcribed prompt, a
    hallucinated tail — aligns to a single instant. Downstream that becomes one
    well-formed segment spanning the chunk, indistinguishable from real speech.
    """
    span = words[-1]["end_time"] - words[0]["start_time"]
    if span < 0.5 * clip_sec:
        raise ValueError(
            f"clip {index}: aligned span {span:.2f}s over {clip_sec:.1f}s of audio"
            f" — the timestamp head collapsed; text: {text[:40]!r}"
        )
    starts = [w["start_time"] for w in words]
    run = longest = 1
    for a, b in zip(starts, starts[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    if longest > max(4, 0.25 * len(words)):
        raise ValueError(
            f"clip {index}: {longest}/{len(words)} consecutive units share one timestamp"
        )
