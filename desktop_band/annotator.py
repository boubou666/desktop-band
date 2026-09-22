"""Local browser-based audio annotation tool for Gopnik Band.

The browser decodes dropped audio and uploads a temporary 44.1 kHz PCM WAV to
this localhost-only server.  StemgenRT and the V2 role router propose timeline
segments; the human corrections are the labels that are persisted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.resources
import json
import mimetypes
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.parse
import wave
import webbrowser

import numpy as np

from .analysis import AudioAnalyzer
from .detection import RoleRouter
from .diagnostics import label_review_bookmark, list_review_bookmarks
from .formations import FormationStore
from .learning import TrainingManager, dataset_summary, export_model_bundle
from .stemgen import STEMGEN_SAMPLE_RATE, StemgenAnalyzer
from .training_tools import (
    activate_model,
    compare_models,
    generate_synthetic_mixes,
    list_models,
)


ANNOTATION_ROLES = (
    "singer",
    "drummer",
    "bassist",
    "keyboard",
    "guitarist",
    "percussion",
)
ROLE_LABELS = {
    "singer": "Chant",
    "drummer": "Batterie",
    "bassist": "Basse",
    "keyboard": "Clavier / synthé",
    "guitarist": "Guitare",
    "percussion": "Percussions",
}

_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


@dataclass(frozen=True, slots=True)
class AnalysisFrame:
    start: float
    end: float
    scores: dict[str, float]


def safe_audio_name(name: str) -> str:
    """Return a predictable WAV name without accepting path traversal."""

    stem = Path(urllib.parse.unquote(name)).stem
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._")
    return (stem[:120] or "audio") + ".wav"


def is_youtube_url(url: str) -> bool:
    """Accept ordinary YouTube links while rejecting arbitrary downloader URLs."""

    try:
        parsed = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme in {"http", "https"} and (
        host in _YOUTUBE_HOSTS or host.endswith(".youtube.com")
    )


def download_youtube_audio(url: str, cache_dir: Path) -> dict[str, object]:
    """Download one browser-decodable YouTube audio stream with yt-dlp."""

    if not is_youtube_url(url):
        raise ValueError("seuls les liens YouTube et youtu.be sont acceptés")
    try:
        from yt_dlp import YoutubeDL
    except ImportError as error:
        raise RuntimeError(
            "support YouTube absent ; installe le projet avec sa dépendance yt-dlp"
        ) from error

    cache_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
        "outtmpl": str(cache_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "playlistend": 1,
        "max_filesize": 250 * 1024 * 1024,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "overwrites": False,
    }
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(url.strip(), download=True)
        if not info:
            raise RuntimeError("YouTube n'a retourné aucun média")
        duration = float(info.get("duration") or 0.0)
        if duration > 30 * 60:
            raise ValueError("vidéo trop longue : maximum 30 minutes")
        requested = info.get("requested_downloads") or []
        filename = requested[0].get("filepath") if requested else None
        candidate = Path(filename) if filename else Path(downloader.prepare_filename(info))

    if not candidate.is_file():
        video_id = re.sub(r"[^A-Za-z0-9_-]", "", str(info.get("id", "")))
        matches = sorted(cache_dir.glob(video_id + ".*"))
        if not matches:
            raise RuntimeError("audio YouTube téléchargé mais fichier introuvable")
        candidate = matches[-1]
    try:
        candidate.resolve().relative_to(cache_dir.resolve())
    except ValueError as error:
        raise RuntimeError("chemin de cache YouTube invalide") from error
    return {
        "title": str(info.get("title") or candidate.stem),
        "duration": duration,
        "filename": candidate.name,
        "mime": mimetypes.guess_type(candidate.name)[0] or "application/octet-stream",
    }


def read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read the PCM WAV produced by the browser into normalized float audio."""

    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if channels not in (1, 2):
        raise ValueError("le WAV d'analyse doit être mono ou stéréo")
    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("le WAV d'analyse doit être PCM 16 ou 32 bits")
    return samples.reshape(-1, channels), sample_rate


def _bridge_role_segments(
    frames: list[AnalysisFrame], role: str, *, maximum_gap: float = 0.32
) -> list[dict]:
    active = [frame for frame in frames if frame.scores.get(role, 0.0) > 0.0]
    if not active:
        return []
    runs: list[list[AnalysisFrame]] = [[active[0]]]
    for frame in active[1:]:
        if frame.start - runs[-1][-1].end <= maximum_gap:
            runs[-1].append(frame)
        else:
            runs.append([frame])

    minimum_duration = 0.16 if role in {"drummer", "percussion"} else 0.34
    segments = []
    for run in runs:
        start = run[0].start
        end = run[-1].end
        if end - start < minimum_duration:
            continue
        confidence = sum(frame.scores[role] for frame in run) / len(run)
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "labels": [role],
            "confidence": round(confidence, 3),
            "source": "automatic",
        })
    return segments


def segments_from_frames(frames: list[AnalysisFrame]) -> list[dict]:
    segments = []
    for role in ANNOTATION_ROLES:
        segments.extend(_bridge_role_segments(frames, role))
    return sorted(segments, key=lambda item: (item["start"], item["labels"][0]))


def review_windows_from_frames(frames: list[AnalysisFrame]) -> list[dict]:
    """Return ambiguous windows worth a human's limited attention."""

    flagged = []
    pairs = (("drummer", "percussion"), ("guitarist", "keyboard"))
    for frame in frames:
        for left, right in pairs:
            first = float(frame.scores.get(left, 0.0))
            second = float(frame.scores.get(right, 0.0))
            strongest = max(first, second)
            if strongest >= 0.18 and abs(first - second) <= 0.12:
                flagged.append({
                    "start": frame.start,
                    "end": frame.end,
                    "labels": [left, right],
                    "confidence": round(strongest, 3),
                    "reason": "rôles proches presque à égalité",
                })
                break
    merged = []
    for item in flagged:
        if (
            merged
            and merged[-1]["labels"] == item["labels"]
            and item["start"] - merged[-1]["end"] <= 0.35
        ):
            merged[-1]["end"] = item["end"]
            merged[-1]["confidence"] = max(
                merged[-1]["confidence"], item["confidence"]
            )
        else:
            merged.append(dict(item))
    return [
        {**item, "start": round(item["start"], 3), "end": round(item["end"], 3)}
        for item in merged if item["end"] - item["start"] >= 0.18
    ]


def analyze_wav(
    path: Path, model_path: str | Path | None = None
) -> dict[str, object]:
    """Create editable role suggestions for one browser-normalized WAV."""

    samples, sample_rate = read_pcm_wav(path)
    if sample_rate != STEMGEN_SAMPLE_RATE:
        raise ValueError(
            f"échantillonnage inattendu : {sample_rate} Hz au lieu de 44100 Hz"
        )
    try:
        analyzer = StemgenAnalyzer(model_path)
        engine = "StemgenRT + classifieur ONNX"
    except Exception:
        analyzer = AudioAnalyzer()
        engine = "analyse légère (modèle StemgenRT absent)"

    roles = tuple(ANNOTATION_ROLES)
    router = RoleRouter("balanced")
    block_size = 4096
    duration = samples.shape[0] / sample_rate
    frames: list[AnalysisFrame] = []
    for offset in range(0, samples.shape[0], block_size):
        block = samples[offset:offset + block_size]
        if block.shape[0] < 32:
            continue
        real_size = block.shape[0]
        if real_size < block_size:
            block = np.pad(block, ((0, block_size - real_size), (0, 0)))
        now = (offset + real_size / 2) / sample_rate
        features = analyzer.process(block, sample_rate, now=now)
        decision = router.update(features, roles, real_size / sample_rate)
        scores = {
            role: round(float(decision.scores.get(role, 0.0)), 4)
            if decision.states.get(role) == "playing"
            else 0.0
            for role in roles
        }
        frames.append(AnalysisFrame(
            start=offset / sample_rate,
            end=min(duration, (offset + real_size) / sample_rate),
            scores=scores,
        ))

    segments = segments_from_frames(frames)
    global_labels = sorted({
        label for segment in segments for label in segment["labels"]
    }, key=ANNOTATION_ROLES.index)
    return {
        "duration": round(duration, 3),
        "engine": engine,
        "global_labels": global_labels,
        "segments": segments,
        "review_windows": review_windows_from_frames(frames),
    }


class AnnotationStore:
    """Persist corrected annotations in the existing training manifest."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.training_dir = self.workspace / "training"
        self.audio_dir = self.training_dir / "audio"
        self.youtube_dir = self.training_dir / "youtube"
        self.manifest_path = self.training_dir / "annotations.json"
        self.history_dir = self.training_dir / "history"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.youtube_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_audio(self, original_name: str, payload: bytes) -> Path:
        if len(payload) < 44:
            raise ValueError("fichier audio vide ou invalide")
        if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
            raise ValueError("le navigateur n'a pas produit un WAV valide")
        target = self.audio_dir / safe_audio_name(original_name)
        target.write_bytes(payload)
        return target

    def save_annotation(self, document: dict) -> Path:
        audio_name = safe_audio_name(str(document.get("audio", "")))
        audio_path = self.audio_dir / audio_name
        if not audio_path.is_file():
            raise ValueError("audio inconnu ; relance d'abord l'analyse")
        duration = max(0.0, float(document.get("duration", 0.0)))
        global_labels = self._validated_labels(document.get("global_labels", []))
        exclusive_label = document.get("exclusive_label") or None
        if exclusive_label is not None:
            validated = self._validated_labels([exclusive_label])
            if not validated:
                raise ValueError("instrument exclusif absent")
            exclusive_label = validated[0]
            global_labels = [exclusive_label]
        segments = []
        for candidate in document.get("segments", []):
            start = max(0.0, float(candidate["start"]))
            end = min(duration, float(candidate["end"]))
            labels = self._validated_labels(candidate.get("labels", []))
            if end <= start or not labels:
                continue
            segments.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "labels": labels,
            })
        clip = {
            "audio": f"audio/{audio_name}",
            "global_labels": global_labels,
            "segments": sorted(segments, key=lambda item: item["start"]),
        }
        if exclusive_label is not None:
            clip["exclusive_label"] = exclusive_label
        with self._lock:
            manifest = {"clips": []}
            if self.manifest_path.is_file():
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                stamp = time.strftime("%Y%m%d-%H%M%S")
                history = self.history_dir / f"{stamp}-{Path(audio_name).stem}.json"
                suffix = 1
                while history.exists():
                    history = self.history_dir / (
                        f"{stamp}-{Path(audio_name).stem}-{suffix}.json"
                    )
                    suffix += 1
                shutil.copy2(self.manifest_path, history)
            clips = [
                item for item in manifest.get("clips", [])
                if item.get("audio") != clip["audio"]
            ]
            clips.append(clip)
            manifest = {"clips": sorted(clips, key=lambda item: item["audio"])}
            self.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return self.manifest_path

    def list_history(self) -> list[dict[str, object]]:
        return [
            {
                "name": path.name,
                "modified": int(path.stat().st_mtime),
                "size": path.stat().st_size,
            }
            for path in sorted(self.history_dir.glob("*.json"), reverse=True)[:50]
        ]

    def restore_history(self, name: str) -> Path:
        candidate = self.history_dir / Path(name).name
        if not candidate.is_file():
            raise ValueError("révision inconnue")
        document = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(document.get("clips"), list):
            raise ValueError("révision invalide")
        with self._lock:
            if self.manifest_path.is_file():
                stamp = time.strftime("%Y%m%d-%H%M%S")
                shutil.copy2(
                    self.manifest_path,
                    self.history_dir / f"{stamp}-avant-restauration.json",
                )
            self.manifest_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return self.manifest_path

    @staticmethod
    def _validated_labels(labels) -> list[str]:
        selected = {str(label) for label in labels}
        unknown = selected.difference(ANNOTATION_ROLES)
        if unknown:
            raise ValueError("instrument inconnu : " + sorted(unknown)[0])
        return [role for role in ANNOTATION_ROLES if role in selected]


def _ui_html() -> bytes:
    return (
        importlib.resources.files("desktop_band")
        .joinpath("annotator_ui.html")
        .read_bytes()
    )


def make_handler(
    store: AnnotationStore,
    model_path: str | None,
    trainer: TrainingManager | None = None,
):
    trainer = trainer or TrainingManager(store.workspace)
    formations = FormationStore(store.training_dir / "formations.json")

    class AnnotationHandler(BaseHTTPRequestHandler):
        server_version = "GopnikBandAnnotator/1.0"

        def log_message(self, format_string, *args) -> None:
            print("annotator:", format_string % args)

        def _json(self, status: HTTPStatus, value: object) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            if path == "/":
                payload = _ui_html()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"ok": True})
                return
            if path == "/api/dataset":
                summary = dataset_summary(store.training_dir)
                summary["training"] = trainer.snapshot()
                self._json(HTTPStatus.OK, summary)
                return
            if path == "/api/models":
                self._json(HTTPStatus.OK, {"models": list_models(store.workspace)})
                return
            if path == "/api/model-compare":
                self._json(HTTPStatus.OK, compare_models(store.workspace))
                return
            if path == "/api/history":
                self._json(HTTPStatus.OK, {"revisions": store.list_history()})
                return
            if path == "/api/formations":
                self._json(HTTPStatus.OK, {"formations": formations.list()})
                return
            if path == "/api/live-reviews":
                self._json(HTTPStatus.OK, {
                    "bookmarks": list_review_bookmarks(store.training_dir)
                })
                return
            if path == "/api/model-export":
                filename, payload = export_model_bundle(store.workspace)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/zip")
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}"'
                )
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/api/youtube-media":
                name = urllib.parse.parse_qs(parsed.query).get("name", [""])[0]
                candidate = store.youtube_dir / Path(name).name
                if not name or not candidate.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = candidate.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(candidate.name)[0]
                    or "application/octet-stream",
                )
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            try:
                if parsed.path == "/api/analyze":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 512 * 1024 * 1024:
                        raise ValueError("taille audio invalide (maximum 512 Mio)")
                    name = urllib.parse.parse_qs(parsed.query).get("name", ["audio"])[0]
                    audio_path = store.save_audio(name, self.rfile.read(length))
                    result = analyze_wav(audio_path, model_path)
                    result["audio"] = audio_path.name
                    self._json(HTTPStatus.OK, result)
                    return
                if parsed.path == "/api/save":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 2 * 1024 * 1024:
                        raise ValueError("annotation trop volumineuse")
                    document = json.loads(self.rfile.read(length).decode("utf-8"))
                    output = store.save_annotation(document)
                    self._json(HTTPStatus.OK, {
                        "ok": True,
                        "path": str(output),
                    })
                    return
                if parsed.path == "/api/youtube":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 4096:
                        raise ValueError("requête YouTube invalide")
                    document = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = download_youtube_audio(
                        str(document.get("url", "")), store.youtube_dir
                    )
                    result["media_url"] = (
                        "/api/youtube-media?name="
                        + urllib.parse.quote(str(result["filename"]))
                    )
                    self._json(HTTPStatus.OK, result)
                    return
                if parsed.path == "/api/train":
                    self._json(HTTPStatus.ACCEPTED, trainer.start())
                    return
                if parsed.path == "/api/synthetic":
                    length = int(self.headers.get("Content-Length", "0"))
                    document = (
                        json.loads(self.rfile.read(length).decode("utf-8"))
                        if length else {}
                    )
                    self._json(HTTPStatus.OK, generate_synthetic_mixes(
                        store.training_dir,
                        count=int(document.get("count", 12)),
                        seconds=float(document.get("seconds", 12.0)),
                    ))
                    return
                if parsed.path == "/api/model-activate":
                    length = int(self.headers.get("Content-Length", "0"))
                    document = json.loads(self.rfile.read(length).decode("utf-8"))
                    self._json(HTTPStatus.OK, activate_model(
                        store.workspace, str(document.get("name", ""))
                    ))
                    return
                if parsed.path == "/api/history/restore":
                    length = int(self.headers.get("Content-Length", "0"))
                    document = json.loads(self.rfile.read(length).decode("utf-8"))
                    output = store.restore_history(str(document.get("name", "")))
                    self._json(HTTPStatus.OK, {"ok": True, "path": str(output)})
                    return
                if parsed.path == "/api/formations":
                    length = int(self.headers.get("Content-Length", "0"))
                    document = json.loads(self.rfile.read(length).decode("utf-8"))
                    saved = formations.save(
                        str(document.get("name", "")), document
                    )
                    self._json(HTTPStatus.OK, {"ok": True, "formation": saved})
                    return
                if parsed.path == "/api/live-reviews/label":
                    length = int(self.headers.get("Content-Length", "0"))
                    document = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = label_review_bookmark(
                        store.training_dir,
                        str(document.get("name", "")),
                        list(document.get("labels", [])),
                    )
                    self._json(HTTPStatus.OK, result)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._json(HTTPStatus.BAD_REQUEST, {
                    "ok": False,
                    "error": str(error),
                })

    return AnnotationHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gopnik-band-annotator",
        description="Annotateur local de morceaux pour le classifieur Gopnik Band.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--model-path")
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = AnnotationStore(args.workspace)
    trainer = TrainingManager(store.workspace)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(store, args.model_path, trainer)
    )
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Annotateur Gopnik Band : {url}")
    print(f"Annotations : {store.manifest_path}")
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
