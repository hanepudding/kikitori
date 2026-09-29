"""Local web page: upload a recording, read and download the transcript, name the speakers.
Edit .env, then run: python serve.py"""
import json
import queue
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from kikitori import Segment, output, transcribe
from settings import HERE, get, hf_token, pipeline_options, resolve, vocab_files

JOBS_DIR = resolve(get("SERVE_JOBS_DIR", "jobs"))
HF_TOKEN = hf_token()
AUDIO = "audio.m4a"

app = FastAPI()
jobs: dict[str, dict] = {}
lock = threading.Lock()
pending: queue.Queue[str] = queue.Queue()


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def save(job: dict) -> None:
    (job_dir(job["id"]) / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")


def update(job_id: str, **fields) -> None:
    with lock:
        jobs[job_id].update(fields)
        save(jobs[job_id])


def lookup(job_id: str) -> dict:
    """A copy of the job: the worker updates the stored dict while responses serialize."""
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return dict(job)


def load_segments(job_id: str) -> list[Segment]:
    raw = json.loads((job_dir(job_id) / "segments.json").read_text(encoding="utf-8"))
    return [Segment(**s) for s in raw]


def to_m4a(source: Path, dest: Path) -> None:
    """The copy the page plays: a small fraction of the source or of 16 kHz PCM."""
    # +faststart puts the index first, so the player can seek before the whole file has arrived
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(source), "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "48k",
         "-movflags", "+faststart", str(dest)],
        check=True, capture_output=True,
    )


def worker() -> None:
    # One job at a time: the aligner and pyannote share the GPU with llama-server
    while True:
        job_id = pending.get()
        with lock:
            job = jobs.get(job_id)
            if job is None:
                # Deleted while it waited in the queue
                continue
            job.update(status="running", started=time.time())
            save(job)
        d = job_dir(job_id)
        source, vocab = d / job["source"], d / "vocab.txt"
        try:
            segments = transcribe(
                source,
                language=job["language"],
                vocab_files=[vocab] if vocab.exists() else [],
                background=job["background"],
                hf_token=HF_TOKEN,
                num_speakers=job["num_speakers"],
                **pipeline_options(),
            )
            to_m4a(source, d / AUDIO)
            (d / "segments.json").write_text(
                json.dumps([asdict(s) for s in segments], ensure_ascii=False), encoding="utf-8")
            update(job_id, status="done", finished=time.time())
        except Exception as e:
            # The page is the only place a failed job is visible; keep the server up for the next one
            traceback.print_exc()
            update(job_id, status="error", finished=time.time(), error=f"{type(e).__name__}: {e}")
        finally:
            # A finished job keeps only job.json, segments.json and the playback copy
            source.unlink(missing_ok=True)
            vocab.unlink(missing_ok=True)


@app.get("/")
def page():
    return FileResponse(HERE / "web" / "index.html")


@app.get("/api/defaults")
def defaults():
    return {
        "language": get("ASR_LANGUAGE", "Chinese"),
        "background": get("ASR_BACKGROUND", ""),
        "vocab": "\n".join(p.read_text(encoding="utf-8").strip() for p in vocab_files()),
        "diarize": HF_TOKEN is not None,
    }


@app.get("/api/jobs")
def list_jobs():
    with lock:
        return sorted((dict(j) for j in jobs.values()), key=lambda j: j["created"], reverse=True)


@app.post("/api/jobs")
def create_job(
    file: UploadFile = File(...),
    language: str = Form(...),
    background: str = Form(""),
    vocab: str = Form(""),
    num_speakers: str = Form(""),
):
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = job_dir(job_id)
    d.mkdir(parents=True)
    filename = Path(file.filename).name
    source = "source" + Path(filename).suffix
    with open(d / source, "wb") as f:
        shutil.copyfileobj(file.file, f, 1 << 20)
    if vocab.strip():
        (d / "vocab.txt").write_text(vocab, encoding="utf-8")
    job = {
        "id": job_id,
        "filename": filename,
        "source": source,
        "created": time.time(),
        "status": "queued",
        "language": language.strip(),
        "background": background.strip(),
        "num_speakers": int(num_speakers) if num_speakers.strip() else None,
        "names": {},
    }
    with lock:
        jobs[job_id] = job
        save(job)
    pending.put(job_id)
    return lookup(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = lookup(job_id)
    segments = [asdict(s) for s in load_segments(job_id)] if job["status"] == "done" else None
    return {**job, "segments": segments}


@app.put("/api/jobs/{job_id}/names")
def set_names(job_id: str, names: dict[str, str]):
    lookup(job_id)
    update(job_id, names={label: name.strip() for label, name in names.items() if name.strip()})
    return lookup(job_id)


@app.get("/api/jobs/{job_id}/transcript.txt")
def download(job_id: str):
    job = lookup(job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"job {job_id} is {job['status']}")
    segments = output.rename_speakers(load_segments(job_id), job["names"])
    name = Path(job["filename"]).stem + ".txt"
    return PlainTextResponse(
        output.format_segments(segments) + "\n",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
    )


@app.get("/api/jobs/{job_id}/audio")
def audio(job_id: str):
    lookup(job_id)
    path = job_dir(job_id) / AUDIO
    if not path.exists():
        raise HTTPException(404, f"job {job_id} has no audio")
    return FileResponse(path, media_type="audio/mp4")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        if job["status"] == "running":
            raise HTTPException(409, f"job {job_id} is running; delete it once it has finished")
        del jobs[job_id]
    shutil.rmtree(job_dir(job_id))
    return {"deleted": job_id}


def restore() -> None:
    """Reload past jobs; the ones a previous run left unfinished go back in the queue."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    unfinished = []
    for path in JOBS_DIR.glob("*/job.json"):
        job = json.loads(path.read_text(encoding="utf-8"))
        jobs[job["id"]] = job
        if job["status"] in ("queued", "running"):
            unfinished.append(job)
    for job in sorted(unfinished, key=lambda j: j["created"]):
        update(job["id"], status="queued")
        pending.put(job["id"])


if __name__ == "__main__":
    restore()
    threading.Thread(target=worker, daemon=True).start()
    host, port = get("SERVE_HOST", "127.0.0.1"), int(get("SERVE_PORT", "8090"))
    print(f"open http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
