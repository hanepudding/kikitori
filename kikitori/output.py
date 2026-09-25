from pathlib import Path

from .types import Segment


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_segments(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        stamp = f"[{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}]"
        speaker = f"  [{seg.speaker}]" if seg.speaker else ""
        lines.append(f"{stamp}{speaker}  {seg.text}")
    return "\n".join(lines)


def rename_speakers(segments: list[Segment], names: dict[str, str]) -> list[Segment]:
    return [
        Segment(s.start, s.end, s.text, names.get(s.speaker, s.speaker))
        for s in segments
    ]


def write(segments: list[Segment], path: str | Path) -> None:
    Path(path).write_text(format_segments(segments) + "\n", encoding="utf-8")
