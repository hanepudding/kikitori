from .types import Span


def pack(
    speech: list[Span],
    target_sec: float = 30.0,
    max_sec: float = 45.0,
) -> list[Span]:
    """Group speech regions into ASR decoding units.

    Every boundary falls in a silence found by the VAD, so no chunk starts or
    ends mid-word. Chunks run as long as the limits allow and may hold more than
    one speaker: the model decodes the exchange in context, and `attribute` cuts
    it afterwards on the word timeline, where a boundary costs nothing.
    """
    chunks: list[Span] = []
    current: list[Span] = []

    def flush() -> None:
        if current:
            chunks.append(Span(current[0].start, current[-1].end))
            current.clear()

    for span in _split_long(speech, max_sec):
        if current and span.end - current[0].start > max_sec:
            flush()
        current.append(span)
        if current[-1].end - current[0].start >= target_sec:
            flush()
    flush()
    return chunks


def _split_long(speech: list[Span], max_sec: float) -> list[Span]:
    """Cut regions of uninterrupted speech that no silence breaks up.

    Nothing here lands in a silence, so the cut can fall inside a word. Equal
    division keeps every piece as long as the limit allows.
    """
    out: list[Span] = []
    for span in speech:
        if span.duration <= max_sec:
            out.append(span)
            continue
        parts = int(span.duration // max_sec) + 1
        step = span.duration / parts
        out += [Span(span.start + i * step, span.start + (i + 1) * step)
                for i in range(parts)]
    return out
