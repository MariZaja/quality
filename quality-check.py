import argparse
import csv
import io
import os
import struct
import tempfile
from math import gcd

import cv2
import mediapipe as mp
import numpy as np
import scipy.io
import torch
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from scipy.signal import butter, filtfilt, periodogram, resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

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

AUDIO_VAD_SAMPLE_RATE = 16000
AUDIO_VAD_SPEECH_RATIO_BAD_THRESHOLD = 0.5

VIDEO_GRAY_CLIP_LOW = 5
VIDEO_GRAY_CLIP_HIGH = 250
VIDEO_FACE_DETECTION_MIN_CONFIDENCE = 0.5
VIDEO_BLUR_BAD_THRESHOLD = 220.0
VIDEO_CLIPPING_BAD_THRESHOLD = 0.0008
VIDEO_FACE_DETECTION_RATE_BAD_THRESHOLD = 0.9

EEG_LABEL_SUFFIX = "_label.mat"
EEG_FLAT_LINE_STD_BAD_THRESHOLD_UV = 0.5
EEG_HIGHPASS_HZ = 1.0
EEG_HIGHPASS_ORDER = 4
EEG_RNSR_SIGNAL_BAND_HZ = (1.0, 40.0)
EEG_RNSR_NOISE_BAND_HZ = (40.0, 250.0)
EEG_RNSR_ZSCORE_BAD_THRESHOLD = 3.0
EEG_PEAK_TO_PEAK_BAD_THRESHOLD_UV = 800.0

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


_VAD_MODEL = None


def get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def resample_for_vad(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == AUDIO_VAD_SAMPLE_RATE:
        return segment
    factor = gcd(sample_rate, AUDIO_VAD_SAMPLE_RATE)
    up, down = AUDIO_VAD_SAMPLE_RATE // factor, sample_rate // factor
    return resample_poly(segment, up, down).astype(np.float32)


def compute_speech_ratio(segment: np.ndarray, sample_rate: int) -> float:
    vad_segment = resample_for_vad(segment, sample_rate)
    speech_timestamps = get_speech_timestamps(
        torch.from_numpy(vad_segment),
        get_vad_model(),
        sampling_rate=AUDIO_VAD_SAMPLE_RATE,
    )
    speech_samples = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    return speech_samples / vad_segment.size


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    diffs = np.diff(padded)
    run_lengths = np.where(diffs == -1)[0] - np.where(diffs == 1)[0]
    return int(run_lengths.max())


def compute_window_metrics(segment: np.ndarray, sample_rate: int) -> dict:
    rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
    with np.errstate(divide="ignore"):
        rms_db = 20.0 * np.log10(rms) if rms > 0 else float("-inf")

    n = segment.size
    zero_ratio = float(np.count_nonzero(np.abs(segment) < AUDIO_ZERO_EPS)) / n
    clip_mask = np.abs(segment) >= AUDIO_CLIP_THRESHOLD
    clip_ratio = float(np.count_nonzero(clip_mask)) / n
    clip_run_max = _longest_run(clip_mask)
    speech_ratio = compute_speech_ratio(segment, sample_rate)

    is_bad = (
        rms_db < AUDIO_RMS_DB_BAD_THRESHOLD
        or zero_ratio > AUDIO_ZERO_RATIO_BAD_THRESHOLD
        or clip_ratio > AUDIO_CLIP_RATIO_BAD_THRESHOLD
        or clip_run_max >= AUDIO_CLIP_RUN_BAD_THRESHOLD
        or speech_ratio < AUDIO_VAD_SPEECH_RATIO_BAD_THRESHOLD
    )

    return {
        "rms_db": round(rms_db, 4) if np.isfinite(rms_db) else rms_db,
        "zero_ratio": round(zero_ratio, 6),
        "clip_ratio": round(clip_ratio, 6),
        "clip_run_max": int(clip_run_max),
        "speech_ratio": round(speech_ratio, 6),
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
        metrics = compute_window_metrics(segment, sample_rate)
        rows.append({"window_id": f"{trial_id}_{idx}", **metrics})
    return rows


def extract_face_crop(frame_bgr: np.ndarray, detection) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    bbox = detection.location_data.relative_bounding_box
    x1 = max(int(bbox.xmin * w), 0)
    y1 = max(int(bbox.ymin * h), 0)
    x2 = min(int((bbox.xmin + bbox.width) * w), w)
    y2 = min(int((bbox.ymin + bbox.height) * h), h)
    return frame_bgr[y1:y2, x1:x2]


def compute_frame_video_metrics(
    frame_bgr: np.ndarray, face_detector
) -> tuple[float | None, float | None, int]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)
    face_detected = 1 if result.detections else 0
    if not face_detected:
        return None, None, face_detected

    face_crop = extract_face_crop(frame_bgr, result.detections[0])
    if face_crop.size == 0:
        return None, None, face_detected

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clip_ratio = float(
        np.count_nonzero((gray < VIDEO_GRAY_CLIP_LOW) | (gray > VIDEO_GRAY_CLIP_HIGH))
    ) / gray.size

    return blur, clip_ratio, face_detected


def compute_window_video_metrics(frames: list[np.ndarray], face_detector) -> dict:
    blurs = []
    clip_ratios = []
    face_hits = []
    for frame in frames:
        blur, clip_ratio, face_detected = compute_frame_video_metrics(frame, face_detector)
        if blur is not None:
            blurs.append(blur)
        if clip_ratio is not None:
            clip_ratios.append(clip_ratio)
        face_hits.append(face_detected)

    window_blur = float(np.mean(blurs)) if blurs else None
    window_clipping = float(np.mean(clip_ratios)) if clip_ratios else None
    face_detection_rate = float(np.mean(face_hits))

    is_bad = (
        face_detection_rate < VIDEO_FACE_DETECTION_RATE_BAD_THRESHOLD
        or window_blur is None
        or window_blur < VIDEO_BLUR_BAD_THRESHOLD
        or window_clipping is None
        or window_clipping > VIDEO_CLIPPING_BAD_THRESHOLD
    )

    return {
        "blur": round(window_blur, 4) if window_blur is not None else None,
        "clipping": round(window_clipping, 6) if window_clipping is not None else None,
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


def eeg_flat_line_per_channel(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nan_present = np.isnan(window).any(axis=0) | np.isinf(window).any(axis=0)
    channel_std = np.nanstd(window, axis=0)
    return channel_std, nan_present


def eeg_peak_to_peak_per_channel(window: np.ndarray) -> np.ndarray:
    return np.nanmax(window, axis=0) - np.nanmin(window, axis=0)


def eeg_rnsr_per_channel(window: np.ndarray, highpass_ba: tuple[np.ndarray, np.ndarray], fs: float) -> np.ndarray:
    b, a = highpass_ba
    window_hp1 = filtfilt(b, a, window, axis=0)
    freqs, psd = periodogram(window_hp1, fs=fs, axis=0)

    signal_mask = (freqs >= EEG_RNSR_SIGNAL_BAND_HZ[0]) & (freqs < EEG_RNSR_SIGNAL_BAND_HZ[1])
    noise_mask = (freqs >= EEG_RNSR_NOISE_BAND_HZ[0]) & (freqs <= EEG_RNSR_NOISE_BAND_HZ[1])
    p_signal = psd[signal_mask].sum(axis=0)
    p_noise = psd[noise_mask].sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p_signal > 0, p_noise / p_signal, np.inf)


def eeg_rnsr_channel_baseline(rnsr_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_channels = rnsr_matrix.shape[1]
    median_c = np.zeros(n_channels, dtype=np.float64)
    scale_c = np.zeros(n_channels, dtype=np.float64)
    for c in range(n_channels):
        finite_vals = rnsr_matrix[:, c][np.isfinite(rnsr_matrix[:, c])]
        if finite_vals.size:
            median_c[c] = np.median(finite_vals)
            scale_c[c] = 1.4826 * np.median(np.abs(finite_vals - median_c[c]))
    return median_c, scale_c


def eeg_rnsr_channel_bad(rnsr_matrix: np.ndarray, median_c: np.ndarray, scale_c: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(scale_c > 0, (rnsr_matrix - median_c) / scale_c, 0.0)
    return ~np.isfinite(rnsr_matrix) | (np.abs(zscore) > EEG_RNSR_ZSCORE_BAD_THRESHOLD)


def find_eeg_segment_array(mat: dict, object_name: str) -> np.ndarray:
    candidates = [v for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3]
    if len(candidates) != 1:
        raise ValueError(
            f"Nie znaleziono jednoznacznej tablicy 3D z danymi EEG w {object_name} "
            f"(znaleziono {len(candidates)} kandydatow)"
        )
    return candidates[0]


def compute_eeg_quality_windows(object_name: str, data: bytes) -> list[dict]:
    mat = scipy.io.loadmat(io.BytesIO(data))
    seg = find_eeg_segment_array(mat, object_name)

    n_samples, _n_channels, n_trials = seg.shape
    fs = n_samples / TRIAL_SECONDS
    window_len = int(round(WINDOW_SECONDS * fs))
    if window_len <= 0:
        raise ValueError(f"Nieprawidlowa dlugosc okna dla {object_name}")

    num_windows = min(n_samples // window_len, MAX_WINDOWS_PER_TRIAL)
    highpass_ba = butter(EEG_HIGHPASS_ORDER, EEG_HIGHPASS_HZ, btype="highpass", fs=fs)

    window_ids = []
    std_rows = []
    nan_rows = []
    ptp_rows = []
    rnsr_rows = []
    for trial_idx in range(n_trials):
        trial_id = f"{trial_idx + 1:03d}"
        for window_idx in range(num_windows):
            start = window_idx * window_len
            window = seg[start : start + window_len, :, trial_idx]

            std_c, nan_c = eeg_flat_line_per_channel(window)
            ptp_c = eeg_peak_to_peak_per_channel(window)
            rnsr_c = eeg_rnsr_per_channel(window, highpass_ba, fs)

            window_ids.append(f"{trial_id}_{window_idx}")
            std_rows.append(std_c)
            nan_rows.append(nan_c)
            ptp_rows.append(ptp_c)
            rnsr_rows.append(rnsr_c)

    std_matrix = np.array(std_rows, dtype=np.float64)
    nan_matrix = np.array(nan_rows, dtype=bool)
    ptp_matrix = np.array(ptp_rows, dtype=np.float64)
    rnsr_matrix = np.array(rnsr_rows, dtype=np.float64)

    flat_bad_matrix = nan_matrix | (std_matrix < EEG_FLAT_LINE_STD_BAD_THRESHOLD_UV)
    ptp_bad_matrix = ptp_matrix > EEG_PEAK_TO_PEAK_BAD_THRESHOLD_UV

    median_c, scale_c = eeg_rnsr_channel_baseline(rnsr_matrix)
    rnsr_bad_matrix = eeg_rnsr_channel_bad(rnsr_matrix, median_c, scale_c)

    channel_bad_matrix = flat_bad_matrix | ptp_bad_matrix | rnsr_bad_matrix
    window_bad = channel_bad_matrix.any(axis=1)

    flat_line_report = np.nanmedian(std_matrix, axis=1)
    ptp_report = np.nanmedian(ptp_matrix, axis=1)
    rnsr_report = np.median(rnsr_matrix, axis=1)

    rows = []
    for window_id, flat_v, rnsr_v, ptp_v, is_bad in zip(
        window_ids, flat_line_report, rnsr_report, ptp_report, window_bad
    ):
        rows.append(
            {
                "window_id": window_id,
                "flat_line": round(float(flat_v), 6),
                "rNSR": round(float(rnsr_v), 6) if np.isfinite(rnsr_v) else float(rnsr_v),
                "peak_to_peak": round(float(ptp_v), 6),
                "quality_flag": "BAD" if is_bad else "GOOD",
            }
        )
    return rows


AUDIO_REPORT_FIELDNAMES = [
    "window_id",
    "rms_db",
    "zero_ratio",
    "clip_ratio",
    "clip_run_max",
    "speech_ratio",
    "quality_flag",
]
VIDEO_REPORT_FIELDNAMES = ["window_id", "blur", "clipping", "face_detection_rate", "quality_flag"]
EEG_REPORT_FIELDNAMES = ["window_id", "flat_line", "rNSR", "peak_to_peak", "quality_flag"]


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


def build_eeg_quality_report(client: Minio, eid: str) -> list[dict]:
    rows = []
    for object_name in list_modality_files(client, eid, "eeg"):
        if object_name.endswith(EEG_LABEL_SUFFIX):
            continue
        try:
            response = client.get_object(BUCKET, object_name)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
            rows.extend(compute_eeg_quality_windows(object_name, data))
        except (S3Error, ValueError) as exc:
            print(f"[WARN] Pominieto {object_name}: {exc}")
    return rows


def save_eeg_quality_report(client: Minio, eid: str, rows: list[dict]) -> str:
    object_name = f"{QUALITY_FLAGS_PREFIX}/{eid}/eeg/{eid}_eeg_quality_flags.csv"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EEG_REPORT_FIELDNAMES)
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


def run_eeg_quality_flags(client: Minio, entities: list[str]) -> None:
    for eid in entities:
        rows = build_eeg_quality_report(client, eid)
        if not rows:
            print(f"[WARN] Brak danych eeg dla {eid}, pomijam zapis raportu.")
            continue
        saved_path = save_eeg_quality_report(client, eid, rows)
        n_bad = sum(1 for r in rows if r["quality_flag"] == "BAD")
        print(f"{eid}: zapisano {BUCKET}/{saved_path} ({len(rows)} okien, {n_bad} BAD)")


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

    if "eeg" in modalities:
        run_eeg_quality_flags(client, entities)


if __name__ == "__main__":
    main()
