import csv
import io
import os
import tempfile

import cv2
import numpy as np
import opensmile
import scipy.io
from minio import Minio
from minio.error import S3Error

from minio_common import (
    EEG_LABEL_SUFFIX,
    MAX_WINDOWS_PER_TRIAL,
    SOURCE_BUCKET,
    TRIAL_SECONDS,
    WINDOW_SECONDS,
    build_arg_parser,
    find_eeg_segment_array,
    get_minio_client,
    list_modality_files,
    parse_wav,
    resolve_entities,
    resolve_modalities,
    trial_number_from_object_name,
)

TARGET_BUCKET = "gold"
FEATURES_PREFIX = "feature_extraction_model"


def parse_args():
    parser = build_arg_parser("Ekstrakcja cech z plikow w MinIO.")
    return parser.parse_args()


_SMILE = None


def get_smile() -> opensmile.Smile:
    global _SMILE
    if _SMILE is None:
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


def extract_audio_features(segment: np.ndarray, sample_rate: int) -> dict:
    features = get_smile().process_signal(segment, sample_rate)
    return {name: round(float(value), 6) for name, value in features.iloc[0].items()}


def extract_video_features(frames: list[np.ndarray]) -> dict:
    # TODO
    return {}


def extract_eeg_features(window: np.ndarray, fs: float) -> dict:
    # TODO
    return {}


def compute_audio_feature_windows(object_name: str, data: bytes) -> list[dict]:
    samples, sample_rate = parse_wav(data)
    window_len = int(round(WINDOW_SECONDS * sample_rate))
    if window_len <= 0:
        raise ValueError(f"Nieprawidlowa dlugosc okna dla {object_name}")

    trial_id = trial_number_from_object_name(object_name)
    num_windows = min(len(samples) // window_len, MAX_WINDOWS_PER_TRIAL)

    rows = []
    for idx in range(num_windows):
        segment = samples[idx * window_len : (idx + 1) * window_len]
        features = extract_audio_features(segment, sample_rate)
        rows.append({"window_id": f"{trial_id}_{idx}", **features})
    return rows


def compute_video_feature_windows(object_name: str, video_path: str) -> list[dict]:
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

        while window_idx < MAX_WINDOWS_PER_TRIAL:
            ret, frame = cap.read()
            if not ret:
                break
            frames_in_window.append(frame)
            if len(frames_in_window) == window_len:
                features = extract_video_features(frames_in_window)
                rows.append({"window_id": f"{trial_id}_{window_idx}", **features})
                window_idx += 1
                frames_in_window = []
        return rows
    finally:
        cap.release()


def compute_eeg_feature_windows(object_name: str, data: bytes) -> list[dict]:
    mat = scipy.io.loadmat(io.BytesIO(data))
    seg = find_eeg_segment_array(mat, object_name)

    n_samples, _n_channels, n_trials = seg.shape
    fs = n_samples / TRIAL_SECONDS
    window_len = int(round(WINDOW_SECONDS * fs))
    if window_len <= 0:
        raise ValueError(f"Nieprawidlowa dlugosc okna dla {object_name}")

    num_windows = min(n_samples // window_len, MAX_WINDOWS_PER_TRIAL)

    rows = []
    for trial_idx in range(n_trials):
        trial_id = f"{trial_idx + 1:03d}"
        for window_idx in range(num_windows):
            start = window_idx * window_len
            window = seg[start : start + window_len, :, trial_idx]
            features = extract_eeg_features(window, fs)
            rows.append({"window_id": f"{trial_id}_{window_idx}", **features})
    return rows


def feature_fieldnames(rows: list[dict]) -> list[str]:
    feature_names: dict[str, None] = {}
    for row in rows:
        for key in row:
            if key != "window_id":
                feature_names[key] = None
    return ["window_id", *feature_names]


def save_feature_report(client: Minio, eid: str, modality: str, rows: list[dict]) -> str:
    object_name = f"{FEATURES_PREFIX}/{eid}/{modality}/{eid}_{modality}_features.csv"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=feature_fieldnames(rows))
    writer.writeheader()
    writer.writerows(rows)

    payload = buffer.getvalue().encode("utf-8")
    client.put_object(
        TARGET_BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="text/csv",
    )
    return object_name


def build_audio_feature_report(client: Minio, eid: str) -> list[dict]:
    rows = []
    for object_name in list_modality_files(client, eid, "audio"):
        try:
            response = client.get_object(SOURCE_BUCKET, object_name)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
            rows.extend(compute_audio_feature_windows(object_name, data))
        except (S3Error, ValueError) as exc:
            print(f"[WARN] Pominieto {object_name}: {exc}")
    return rows


def build_video_feature_report(client: Minio, eid: str) -> list[dict]:
    rows = []
    for object_name in list_modality_files(client, eid, "video"):
        suffix = os.path.splitext(object_name)[1] or ".mp4"
        try:
            response = client.get_object(SOURCE_BUCKET, object_name)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()

            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_file:
                tmp_file.write(data)
                tmp_file.flush()
                rows.extend(compute_video_feature_windows(object_name, tmp_file.name))
        except (S3Error, ValueError) as exc:
            print(f"[WARN] Pominieto {object_name}: {exc}")
    return rows


def build_eeg_feature_report(client: Minio, eid: str) -> list[dict]:
    rows = []
    for object_name in list_modality_files(client, eid, "eeg"):
        if object_name.endswith(EEG_LABEL_SUFFIX):
            continue
        try:
            response = client.get_object(SOURCE_BUCKET, object_name)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
            rows.extend(compute_eeg_feature_windows(object_name, data))
        except (S3Error, ValueError) as exc:
            print(f"[WARN] Pominieto {object_name}: {exc}")
    return rows


MODALITY_BUILDERS = {
    "audio": build_audio_feature_report,
    "video": build_video_feature_report,
    "eeg": build_eeg_feature_report,
}


def run_feature_extraction(client: Minio, modality: str, entities: list[str]) -> None:
    builder = MODALITY_BUILDERS[modality]
    for eid in entities:
        rows = builder(client, eid)
        if not rows:
            print(f"[WARN] Brak danych {modality} dla {eid}, pomijam zapis raportu.")
            continue
        saved_path = save_feature_report(client, eid, modality, rows)
        print(f"{eid}: zapisano {TARGET_BUCKET}/{saved_path} ({len(rows)} okien)")


def main() -> None:
    args = parse_args()
    modalities = resolve_modalities(args.modality)
    entities = resolve_entities(args.entity)

    print(f"Modalnosci: {list(modalities)}")
    print(f"Entities: {entities[0]}..{entities[-1]} ({len(entities)} szt.)")

    client = get_minio_client()

    for modality in modalities:
        run_feature_extraction(client, modality, entities)


if __name__ == "__main__":
    main()
