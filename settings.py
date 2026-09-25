"""Settings for the entry scripts, read from .env in the project root."""
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}


def get(key: str, default: str | None = None) -> str | None:
    """An unset or blank key means "use the default"."""
    return os.environ.get(key, "").strip() or default


def flag(key: str, default: bool) -> bool:
    value = get(key)
    if value is None:
        return default
    if value.lower() in TRUE:
        return True
    if value.lower() in FALSE:
        return False
    raise ValueError(f"{key}={value!r} is not a boolean; use one of {sorted(TRUE | FALSE)}")


def resolve(value: str) -> Path:
    """Relative paths in .env are relative to the project, not to the shell's cwd."""
    return Path(value) if Path(value).is_absolute() else HERE / value


def server_url() -> str:
    """Where the Qwen3-ASR llama-server listens."""
    return f"http://{get('ASR_SERVER_HOST', '127.0.0.1')}:{get('ASR_SERVER_PORT', '8080')}"


def hf_token() -> str | None:
    """The token pyannote needs, or None when diarization is off."""
    if not flag("ASR_DIARIZE", True):
        return None
    token = get("HF_TOKEN")
    if token is None:
        raise SystemExit("ASR_DIARIZE is on but HF_TOKEN is empty: fill it in .env, or set ASR_DIARIZE to 0")
    return token


def vocab_files() -> list[Path]:
    return [resolve(p.strip()) for p in get("ASR_VOCAB", "").split(",") if p.strip()]


def num_speakers() -> int | None:
    value = get("ASR_NUM_SPEAKERS")
    return int(value) if value else None


def pipeline_options() -> dict:
    """transcribe()'s tuning arguments, the ones every entry script takes from .env."""
    return dict(
        target_sec=float(get("ASR_TARGET_SEC", "30")),
        max_sec=float(get("ASR_MAX_SEC", "45")),
        min_seg_sec=float(get("ASR_MIN_SEG_SEC", "1.5")),
        gap_fill_sec=float(get("ASR_GAP_FILL_SEC", "0.5")),
        batch_size=int(get("ASR_BATCH_SIZE", "8")),
        server=server_url(),
    )
