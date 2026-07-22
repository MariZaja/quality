import argparse
import csv
import io
import json
import os
import struct
import tempfile
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

load_dotenv()

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

BUCKET = "silver"
ENTITY_RESOLUTION_PREFIX = "01_entity_resolution/eav/files"
QUALITY_FLAGS_PREFIX = "04_quality_flags_model"

ENTITIES = [f"e{n:02d}" for n in range(1, 43)]  # e01 .. e42
MODALITIES = ("audio", "video", "eeg")

WINDOW_SECONDS = 1.0
TRIAL_SECONDS = 20.0
MAX_WINDOWS_PER_TRIAL = int(TRIAL_SECONDS / WINDOW_SECONDS)

AUDIO_ZERO_EPS = 1e-6
AUDIO_CLIP_THRESHOLD = 0.999
AUDIO_RMS_DB_BAD_THRESHOLD = -60.0
AUDIO_ZERO_RATIO_BAD_THRESHOLD = 0.5
AUDIO_CLIP_RATIO_BAD_THRESHOLD = 0.001
AUDIO_CLIP_RUN_BAD_THRESHOLD = 3

VIDEO_GRAY_CLIP_LOW = 5
VIDEO_GRAY_CLIP_HIGH = 250
VIDEO_FACE_DETECTION_MIN_CONFIDENCE = 0.5
VIDEO_BLUR_BAD_THRESHOLD = 35.0
VIDEO_CLIPPING_BAD_THRESHOLD = 0.01
VIDEO_FACE_DETECTION_RATE_BAD_THRESHOLD = 0.9

def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def build_modality_prefix(eid: str, modality: str) -> str:
    return f"{ENTITY_RESOLUTION_PREFIX}/entity={eid}/modality={modality}/"


def resolve_modalities(modality_args: list[str]) -> tuple[str, ...]:
    requested = [m.strip().lower() for m in modality_args]
    if "all" in requested:
        return MODALITIES

    invalid = [m for m in requested if m not in MODALITIES]
    if invalid:
        raise ValueError(
            f"Nieznana modalnosc: {invalid}. Dozwolone: {list(MODALITIES)} lub 'all'."
        )
    return tuple(dict.fromkeys(requested))


def resolve_entities(start_entity: str) -> list[str]:
    start_entity = start_entity.strip().lower()
    if start_entity not in ENTITIES:
        raise ValueError(
            f"Nieznane entity: {start_entity}. Dozwolone: {ENTITIES[0]}..{ENTITIES[-1]}."
        )
    start_index = ENTITIES.index(start_entity)
    return ENTITIES[start_index:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality check plikow z MinIO.")
    parser.add_argument(
        "--modality",
        nargs="+",
        default=["all"],
        help="Modalnosc/i do sprawdzenia: 'all', 'audio', 'video', 'eeg' "
        "lub dowolna kombinacja podana jako kilka wartosci (np. --modality audio eeg). "
        "Domyslnie 'all'.",
    )
    parser.add_argument(
        "--entity",
        default=ENTITIES[0],
        help=f"Entity od ktorego zaczac iteracje (np. e05). Zakres: {ENTITIES[0]}..{ENTITIES[-1]}. "
        f"Domyslnie {ENTITIES[0]}.",
    )
    return parser.parse_args()


def list_modality_files(client: Minio, eid: str, modality: str) -> list[str]:
    prefix = build_modality_prefix(eid, modality)
    objects = client.list_objects(BUCKET, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]


def parse_wav(data: bytes) -> tuple[np.ndarray, int]:
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Nieprawidlowy naglowek WAV")

    pos = 12
    fmt = None
    audio_bytes = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + chunk_size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            audio_bytes = body
        pos += 8 + chunk_size + (chunk_size % 2)

    if fmt is None or audio_bytes is None:
        raise ValueError("Brak chunku fmt/data w pliku WAV")

    audio_format, channels, sample_rate, _, _, bits_per_sample = fmt
    if (audio_format, channels, bits_per_sample) != (3, 1, 32):
        raise ValueError(
            f"Nieobslugiwany format WAV: format={audio_format}, channels={channels}, bits={bits_per_sample}"
        )

    return np.frombuffer(audio_bytes, dtype="<f4"), sample_rate


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    diffs = np.diff(padded)
    run_lengths = np.where(diffs == -1)[0] - np.where(diffs == 1)[0]
    return int(run_lengths.max())


def compute_window_metrics(segment: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
    with np.errstate(divide="ignore"):
        rms_db = 20.0 * np.log10(rms) if rms > 0 else float("-inf")

    n = segment.size
    zero_ratio = float(np.count_nonzero(np.abs(segment) < AUDIO_ZERO_EPS)) / n
    clip_mask = np.abs(segment) >= AUDIO_CLIP_THRESHOLD
    clip_ratio = float(np.count_nonzero(clip_mask)) / n
    clip_run_max = _longest_run(clip_mask)

    is_bad = (
        rms_db < AUDIO_RMS_DB_BAD_THRESHOLD
        or zero_ratio > AUDIO_ZERO_RATIO_BAD_THRESHOLD
        or clip_ratio > AUDIO_CLIP_RATIO_BAD_THRESHOLD
        or clip_run_max >= AUDIO_CLIP_RUN_BAD_THRESHOLD
    )

    return {
        "rms_db": round(rms_db, 4) if np.isfinite(rms_db) else rms_db,
        "zero_ratio": round(zero_ratio, 6),
        "clip_ratio": round(clip_ratio, 6),
        "clip_run_max": int(clip_run_max),
        "quality_flag": "BAD" if is_bad else "GOOD",
    }


def trial_number_from_object_name(object_name: str) -> str:
    basename = object_name.rsplit("/", 1)[-1]
    return basename.split("_", 1)[0]


def compute_audio_quality_windows(object_name: str, data: bytes) -> list[dict]:
    samples, sample_rate = parse_wav(data)
    window_len = int(round(WINDOW_SECONDS * sample_rate))
    if window_len <= 0:
        raise ValueError(f"Nieprawidlowa dlugosc okna dla {object_name}")

    trial_id = trial_number_from_object_name(object_name)
    num_windows = min(len(samples) // window_len, MAX_WINDOWS_PER_TRIAL)

    rows = []
    for idx in range(num_windows):
        segment = samples[idx * window_len : (idx + 1) * window_len]
        metrics = compute_window_metrics(segment)
        rows.append({"window_id": f"{trial_id}_{idx}", **metrics})
    return rows


def compute_frame_video_metrics(frame_bgr: np.ndarray, face_detector) -> tuple[float, float, int]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clip_ratio = float(
        np.count_nonzero((gray < VIDEO_GRAY_CLIP_LOW) | (gray > VIDEO_GRAY_CLIP_HIGH))
    ) / gray.size

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)
    face_detected = 1 if result.detections else 0

    return blur, clip_ratio, face_detected


def compute_window_video_metrics(frames: list[np.ndarray], face_detector) -> dict:
    blurs = []
    clip_ratios = []
    face_hits = []
    for frame in frames:
        blur, clip_ratio, face_detected = compute_frame_video_metrics(frame, face_detector)
        blurs.append(blur)
        clip_ratios.append(clip_ratio)
        face_hits.append(face_detected)

    window_blur = float(np.mean(blurs))
    window_clipping = float(np.mean(clip_ratios))
    face_detection_rate = float(np.mean(face_hits))

    is_bad = (
        window_blur < VIDEO_BLUR_BAD_THRESHOLD
        or window_clipping > VIDEO_CLIPPING_BAD_THRESHOLD
        or face_detection_rate < VIDEO_FACE_DETECTION_RATE_BAD_THRESHOLD
    )

    return {
        "blur": round(window_blur, 4),
        "clipping": round(window_clipping, 6),
        "face_detection_rate": round(face_detection_rate, 4),
        "quality_flag": "BAD" if is_bad else "GOOD",
    }


def compute_video_quality_windows(object_name: str, video_path: str) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise ValueError(f"Nieprawidlowy fps dla {object_name}")
        window_len = int(round(WINDOW_SECONDS * fps))

        trial_id = trial_number_from_object_name(object_name)
        rows = []
        frames_in_window = []
        window_idx = 0

        with mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=VIDEO_FACE_DETECTION_MIN_CONFIDENCE
        ) as face_detector:
            while window_idx < MAX_WINDOWS_PER_TRIAL:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_in_window.append(frame)
                if len(frames_in_window) == window_len:
                    metrics = compute_window_video_metrics(frames_in_window, face_detector)
                    rows.append({"window_id": f"{trial_id}_{window_idx}", **metrics})
                    window_idx += 1
                    frames_in_window = []
        return rows
    finally:
        cap.release()


def check_eeg_quality(client: Minio, eid: str, object_name: str) -> dict:
    """Sprawdza jakosc pojedynczego pliku EEG.

    TODO:
        - wczytac plik EEG z MinIO (client.get_object)
        - sprawdzic liczbe kanalow, czestotliwosc probkowania, dlugosc zapisu
        - wykryc artefakty, brakujace odcinki, plaskie/uszkodzone kanaly
        - ustalic kryteria PASS/FAIL i zwrocic flagi jakosci
    """
    return {
        "entity": eid,
        "modality": "eeg",
        "object_name": object_name,
        "status": "TODO",
    }


MODALITY_CHECKERS = {
    "eeg": check_eeg_quality,
}

AUDIO_REPORT_FIELDNAMES = ["window_id", "rms_db", "zero_ratio", "clip_ratio", "clip_run_max", "quality_flag"]
VIDEO_REPORT_FIELDNAMES = ["window_id", "blur", "clipping", "face_detection_rate", "quality_flag"]


def build_audio_quality_report(client: Minio, eid: str) -> list[dict]:
    rows = []
    for object_name in list_modality_files(client, eid, "audio"):
        try:
            response = client.get_object(BUCKET, object_name)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
            rows.extend(compute_audio_quality_windows(object_name, data))
        except (S3Error, ValueError) as exc:
            print(f"[WARN] Pominieto {object_name}: {exc}")
    return rows


def save_audio_quality_report(client: Minio, eid: str, rows: list[dict]) -> str:
    object_name = f"{QUALITY_FLAGS_PREFIX}/{eid}/audio/{eid}_audio_quality_flags.csv"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=AUDIO_REPORT_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

    payload = buffer.getvalue().encode("utf-8")
    client.put_object(
        BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="text/csv",
    )
    return object_name


def run_audio_quality_flags(client: Minio, entities: list[str]) -> None:
    for eid in entities:
        rows = build_audio_quality_report(client, eid)
        if not rows:
            print(f"[WARN] Brak danych audio dla {eid}, pomijam zapis raportu.")
            continue
        saved_path = save_audio_quality_report(client, eid, rows)
        n_bad = sum(1 for r in rows if r["quality_flag"] == "BAD")
        print(f"{eid}: zapisano {BUCKET}/{saved_path} ({len(rows)} okien, {n_bad} BAD)")


def build_video_quality_report(client: Minio, eid: str) -> list[dict]:
    rows = []
    for object_name in list_modality_files(client, eid, "video"):
        suffix = os.path.splitext(object_name)[1] or ".mp4"
        try:
            response = client.get_object(BUCKET, object_name)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()

            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_file:
                tmp_file.write(data)
                tmp_file.flush()
                rows.extend(compute_video_quality_windows(object_name, tmp_file.name))
        except (S3Error, ValueError) as exc:
            print(f"[WARN] Pominieto {object_name}: {exc}")
    return rows


def save_video_quality_report(client: Minio, eid: str, rows: list[dict]) -> str:
    object_name = f"{QUALITY_FLAGS_PREFIX}/{eid}/video/{eid}_video_quality_flags.csv"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=VIDEO_REPORT_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

    payload = buffer.getvalue().encode("utf-8")
    client.put_object(
        BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="text/csv",
    )
    return object_name


def run_video_quality_flags(client: Minio, entities: list[str]) -> None:
    for eid in entities:
        rows = build_video_quality_report(client, eid)
        if not rows:
            print(f"[WARN] Brak danych video dla {eid}, pomijam zapis raportu.")
            continue
        saved_path = save_video_quality_report(client, eid, rows)
        n_bad = sum(1 for r in rows if r["quality_flag"] == "BAD")
        print(f"{eid}: zapisano {BUCKET}/{saved_path} ({len(rows)} okien, {n_bad} BAD)")


def run_quality_checks(
    client: Minio,
    entities: list[str] | None = None,
    modalities: tuple[str, ...] | None = None,
) -> list[dict]:
    entities = entities if entities is not None else ENTITIES
    modalities = modalities if modalities is not None else MODALITIES
    results = []

    for eid in entities:
        for modality in modalities:
            checker = MODALITY_CHECKERS[modality]
            try:
                object_names = list_modality_files(client, eid, modality)
            except S3Error as exc:
                print(f"[WARN] Nie udalo sie wylistowac {eid}/{modality}: {exc}")
                continue

            for object_name in object_names:
                results.append(checker(client, eid, object_name))

    return results


def save_report(client: Minio, report: list[dict]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    object_name = f"{QUALITY_FLAGS_PREFIX}/quality_report_{timestamp}.json"

    payload = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    client.put_object(
        BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )
    return object_name


def main() -> None:
    args = parse_args()
    modalities = resolve_modalities(args.modality)
    entities = resolve_entities(args.entity)

    print(f"Modalnosci: {list(modalities)}")
    print(f"Entities: {entities[0]}..{entities[-1]} ({len(entities)} szt.)")

    client = get_minio_client()

    if "audio" in modalities:
        run_audio_quality_flags(client, entities)

    if "video" in modalities:
        run_video_quality_flags(client, entities)

    other_modalities = tuple(m for m in modalities if m not in ("audio", "video"))
    if other_modalities:
        report = run_quality_checks(client, entities=entities, modalities=other_modalities)
        saved_path = save_report(client, report)
        print(f"Zapisano raport jakosci: {BUCKET}/{saved_path} ({len(report)} wpisow)")


if __name__ == "__main__":
    main()
