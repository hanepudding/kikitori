import base64
import io
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import soundfile as sf

from .audio import SAMPLE_RATE

SERVER = "http://127.0.0.1:8080"


class Qwen3ASR:
    """Client for Qwen3-ASR served by llama-server; the model lives in the server process."""

    def __init__(
        self,
        server: str = SERVER,
        max_new_tokens: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.client = httpx.Client(base_url=server, timeout=None)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def transcribe(
        self,
        clips: list[np.ndarray],
        prompt: str | None = None,
        language: str | None = "Chinese",
    ) -> list[str]:
        """Transcribe independent audio clips.

        `prompt` is the context-biasing text (domain terms, speaker names). Each
        clip is decoded on its own, so the prompt is the only context carried
        across them. `batch_size` requests are in flight at once; llama-server
        serves them in parallel only with at least that many slots (-np).
        """
        from transformers.models.qwen3_asr.processing_qwen3_asr import (
            LANGUAGE_CODE_TO_NAME,
            _parse_single_output,
            prepare_language_inputs,
        )

        # Build the request the way transformers' apply_transcription_request does: language codes
        # become full names, and a forced language prefills the assistant turn
        name = prepare_language_inputs(language, 1, LANGUAGE_CODE_TO_NAME, return_code=False)[0]

        def one(clip: np.ndarray) -> str:
            wav = io.BytesIO()
            sf.write(wav, clip, SAMPLE_RATE, format="WAV", subtype="FLOAT")
            messages = []
            if prompt is not None:
                messages.append({"role": "system", "content": prompt})
            messages.append({"role": "user", "content": [{
                "type": "input_audio",
                "input_audio": {"data": base64.b64encode(wav.getvalue()).decode(), "format": "wav"},
            }]})
            if name is not None:
                messages.append({"role": "assistant", "content": f"language {name}<asr_text>"})
            r = self.client.post("/v1/chat/completions", json={
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": 0,
            })
            r.raise_for_status()
            # The content comes back with the prefilled prefix included. Parse it with the function behind
            # transformers' decode(return_format="transcription_only"), repetition fix included
            return _parse_single_output(r.json()["choices"][0]["message"]["content"])["transcription"]

        with ThreadPoolExecutor(self.batch_size) as pool:
            return list(pool.map(one, clips))
