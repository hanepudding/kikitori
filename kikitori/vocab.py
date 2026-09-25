from pathlib import Path


def load_vocab(*paths: str | Path) -> list[str]:
    """Read one term per line; blank lines and # comments are ignored."""
    terms: list[str] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                terms.append(line)
    return list(dict.fromkeys(terms))


def build_prompt(terms: list[str], background: str = "") -> str | None:
    """Format the biasing prompt Qwen3-ASR reads as a system message."""
    parts = []
    if background:
        parts.append(background.strip())
    if terms:
        parts.append("Vocabulary: " + "\N{IDEOGRAPHIC COMMA}".join(terms) + "\N{IDEOGRAPHIC FULL STOP}")
    return "\n".join(parts) or None


def is_echo(text: str, terms: list[str], min_hits: int = 8) -> bool:
    """True when the model transcribed the biasing prompt instead of the audio.

    A chunk the model cannot hear falls back to copying the vocabulary list,
    verbatim and in file order. Eight distinct terms in one chunk does not
    happen in speech.
    """
    return sum(1 for t in terms if t in text) >= min_hits
