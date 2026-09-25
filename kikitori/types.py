from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """A time range in the source audio, in seconds.

    Used for VAD speech regions (speaker None), diarization turns, and the
    chunks handed to the ASR model.
    """
    start: float
    end: float
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Segment:
    """A transcribed chunk."""
    start: float
    end: float
    text: str
    speaker: str | None = None
