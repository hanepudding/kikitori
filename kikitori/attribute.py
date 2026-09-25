from transformers.models.qwen3_asr.processing_qwen3_asr import _is_cjk_char, _is_kept_char

from .align import Word
from .types import Segment, Span

SENTENCE_END = "\N{IDEOGRAPHIC FULL STOP}\N{FULLWIDTH QUESTION MARK}\N{FULLWIDTH EXCLAMATION MARK}?!"

# Aligner timestamp resolution. A character that starts and ends inside one cell
# comes back with start == end, which overlaps no turn at all.
STEP = 0.08


def speaker_at(start: float, end: float, turns: list[Span]) -> str | None:
    overlap: dict[str, float] = {}
    for turn in turns:
        shared = min(end, turn.end) - max(start, turn.start)
        if shared > 0:
            overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + shared
    if not overlap:
        return None
    return max(overlap, key=overlap.__getitem__)


def word_spans(text: str) -> list[tuple[int, int]]:
    """Replay the aligner's tokenisation over the source text to locate each unit."""
    spans: list[tuple[int, int]] = []
    start = end = None

    def flush() -> None:
        nonlocal start, end
        if start is not None:
            spans.append((start, end))
            start = end = None

    for i, char in enumerate(text):
        if _is_cjk_char(char):
            flush()
            spans.append((i, i + 1))
        elif char.isspace():
            flush()
        elif _is_kept_char(char):
            if start is None:
                start = i
            # The span ends at the last kept char, not at the one that flushed it:
            # the tokeniser drops the full-width comma after "OK" but it still belongs in the text.
            end = i + 1
    flush()
    return spans


def attach_punctuation(text: str, words: list[Word]) -> list[tuple[Word, str]]:
    """Pair each aligned word with its own source text plus everything up to the next.

    A word is not always a substring of the transcript: the tokeniser drops any
    character that is not a letter, digit, CJK or apostrophe without breaking the
    token there, so "3.5" comes back as "35" and "OK" + full-width comma + "OK" as "OKOK".
    """
    spans = word_spans(text)
    if len(spans) != len(words):
        raise ValueError(f"{len(words)} aligned words but {len(spans)} tokens in {text[:60]!r}")
    return [
        (word, text[start:(spans[i + 1][0] if i + 1 < len(spans) else len(text))])
        for i, (word, (start, _)) in enumerate(zip(words, spans))
    ]


def smooth(speakers: list[str], words: list[Word], min_turn_sec: float) -> list[str]:
    """Absorb speaker runs too short to be a real turn.

    Diarization flickers for a word or two around a turn's edge. A single
    character handed to the other party is noise, and it costs a spurious line
    break in the output plus two wrong labels.
    """
    out = list(speakers)
    start = 0
    for i in range(1, len(out) + 1):
        if i < len(out) and out[i] == out[start]:
            continue
        if words[i - 1].end - words[start].start < min_turn_sec:
            if start > 0:
                out[start:i] = [out[start - 1]] * (i - start)
            elif i < len(out):
                # A run at the chunk head has no run before it; the next one is
                # its only neighbour, so this one absorbs forwards.
                out[start:i] = [out[i]] * (i - start)
        start = i
    return out


def to_segments(
    chunk: Span,
    text: str,
    words: list[Word],
    turns: list[Span],
    min_sec: float = 1.5,
    min_turn_sec: float = 0.3,
    gap_fill_sec: float = 0.5,
) -> list[Segment]:
    """Cut one decoded chunk into segments at speaker changes and sentence ends.

    Decoding stays on the whole chunk so the model keeps its context; the cutting
    happens afterwards on the word timeline, where a boundary costs nothing.
    """
    if not words:
        return []
    pairs = attach_punctuation(text, words)
    fallback = speaker_at(chunk.start, chunk.end, turns)
    speakers = [
        speaker_at(chunk.start + w.start, chunk.start + max(w.end, w.start + STEP), turns)
        or fallback
        for w in words
    ]
    speakers = smooth(speakers, words, min_turn_sec)

    segments: list[Segment] = []
    buf: list[str] = []
    first = 0

    def flush(last: int) -> None:
        nonlocal buf, first
        body = "".join(buf).strip()
        if body:
            segments.append(Segment(
                chunk.start + words[first].start,
                chunk.start + max(words[last].end, words[first].start + 0.02),
                body,
                speakers[first],
            ))
        buf = []

    for i, (word, piece) in enumerate(pairs):
        if buf and speakers[i] != speakers[i - 1]:
            flush(i - 1)
            first = i
        if not buf:
            first = i
        buf.append(piece)
        if piece[-1:] in tuple(SENTENCE_END) and words[i].end - words[first].start >= min_sec:
            flush(i)
    flush(len(words) - 1)
    if segments:
        # The chunk edges are VAD onsets and offsets; the first and last words sit
        # inside them. Snapping out covers the lead-in and the decay, which the
        # word timeline leaves outside every segment.
        segments[0] = Segment(chunk.start, segments[0].end, segments[0].text, segments[0].speaker)
        segments[-1] = Segment(segments[-1].start, chunk.end, segments[-1].text, segments[-1].speaker)
    for i in range(len(segments) - 1):
        # A breath between two lines belongs to the line before it. Longer pauses
        # are left as real gaps, so a timestamp still means what it says.
        if 0 < segments[i + 1].start - segments[i].end <= gap_fill_sec:
            segments[i] = Segment(segments[i].start, segments[i + 1].start,
                                  segments[i].text, segments[i].speaker)
    return segments
