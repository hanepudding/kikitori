"""Transcribe one media file. Edit .env, then run: python run.py"""
from kikitori import transcribe_file
from settings import get, hf_token, num_speakers, pipeline_options, resolve, vocab_files


def speaker_names(raw: str) -> dict[str, str]:
    names = {}
    for pair in raw.split(";"):
        if not pair.strip():
            continue
        label, sep, name = pair.partition("=")
        if not sep or "=" in name or not label.strip() or not name.strip():
            raise ValueError(f"{pair!r} in ASR_SPEAKER_NAMES is not label=name; separate pairs with ;")
        names[label.strip()] = name.strip()
    return names


if __name__ == "__main__":
    source = get("ASR_SOURCE")
    if source is None:
        raise SystemExit("ASR_SOURCE is not set: put the file to transcribe in .env")

    transcribe_file(
        source,
        resolve(get("ASR_OUT_DIR", "out")),
        speaker_names=speaker_names(get("ASR_SPEAKER_NAMES", "")),
        language=get("ASR_LANGUAGE", "Chinese"),
        vocab_files=vocab_files(),
        background=get("ASR_BACKGROUND", ""),
        hf_token=hf_token(),
        num_speakers=num_speakers(),
        **pipeline_options(),
    )
