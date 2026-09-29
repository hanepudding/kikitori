"""The whole pipeline as one call."""
from pathlib import Path

from . import attribute, audio, device, output, segment, vad, vocab
from .align import ForcedAligner
from .asr import SERVER, Qwen3ASR
from .types import Segment


def transcribe(
    source: str | Path,
    language: str = "Chinese",
    vocab_files: list[str | Path] = [],
    background: str = "",
    hf_token: str | None = None,
    num_speakers: int | None = None,
    target_sec: float = 30.0,
    max_sec: float = 45.0,
    min_seg_sec: float = 1.5,
    gap_fill_sec: float = 0.5,
    batch_size: int = 8,
    server: str = SERVER,
) -> list[Segment]:
    """Transcribe one media file into segments. Nothing is written to disk.

    hf_token None means no speaker diarization: lines still break at sentence
    ends, they just carry no speaker.
    """
    print("=== ffmpeg ===")
    samples = audio.decode(source)
    print(f"{Path(source).name}  {len(samples) / audio.SAMPLE_RATE / 60:.1f} min")

    turns = []
    if hf_token:
        from .diarize import diarize
        print("=== pyannote ===")
        turns = diarize(samples, hf_token, num_speakers=num_speakers)
        print(f"{len(turns)} turns, {len({t.speaker for t in turns})} speakers")

    print("=== vad ===")
    speech = vad.detect_speech(samples)
    chunks = segment.pack(speech, target_sec, max_sec)
    print(f"{len(speech)} speech regions -> {len(chunks)} chunks, "
          f"{sum(c.duration for c in chunks) / len(chunks):.1f} s avg")

    print("=== qwen3-asr ===")
    terms = vocab.load_vocab(*vocab_files) if vocab_files else []
    prompt = vocab.build_prompt(terms, background)
    clips = audio.clips(samples, chunks)
    model = Qwen3ASR(server=server, batch_size=batch_size)
    texts = model.transcribe(clips, prompt, language)

    echoed = [i for i, t in enumerate(texts) if vocab.is_echo(t, terms)]
    if echoed:
        print(f"{len(echoed)} chunks echoed the vocabulary, re-decoding without it")
        for i, t in zip(echoed, model.transcribe([clips[i] for i in echoed], None, language)):
            texts[i] = t
    del model
    device.free()

    # An empty chunk (silence VAD mistook for speech) has no text to align, and the aligner errors on it
    spoken = [(c, clip, t) for c, clip, t in zip(chunks, clips, texts) if t.strip()]
    print(f"{len(spoken)}/{len(chunks)} chunks transcribed")

    print("=== forced aligner ===")
    aligner = ForcedAligner(batch_size=max(1, batch_size // 2))
    words = aligner.align([c for _, c, _ in spoken], [t for _, _, t in spoken], language)
    del aligner
    device.free()

    return [
        s
        for (chunk, _, text), w in zip(spoken, words)
        for s in attribute.to_segments(chunk, text, w, turns, min_seg_sec,
                                       gap_fill_sec=gap_fill_sec)
    ]


def transcribe_file(
    source: str | Path,
    out_dir: str | Path,
    name: str | None = None,
    speaker_names: dict[str, str] | None = None,
    **options,
) -> Path:
    """Transcribe one media file into out_dir/<name>.txt and return that path.

    name defaults to the source's folder and stem, because stems repeat across
    the corpus; pass one when the caller already gives every run its own
    directory. options are transcribe()'s keyword arguments.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or audio.run_key(source)
    out_path = out_dir / (name + ".txt")

    segments = transcribe(source, **options)
    if speaker_names:
        segments = output.rename_speakers(segments, speaker_names)
    output.write(segments, out_path)
    print(f"=== {len(segments)} segments, "
          f"{sum(s.end - s.start for s in segments) / len(segments):.1f} s avg "
          f"-> {out_path} ===")
    return out_path
