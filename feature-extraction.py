import csv
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import opensmile
import pandas as pd
import scipy.io
from minio import Minio
from minio.error import S3Error
from mne_features.feature_extraction import extract_features

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

EEG_MNE_FUNCS = ["pow_freq_bands", "hjorth_mobility", "hjorth_complexity", "app_entropy"]
EEG_MNE_FUNC_PARAMS = {
    "pow_freq_bands__freq_bands": np.array([0.5, 4.0, 8.0, 13.0, 30.0, 100.0]),
    "app_entropy__emb": 2,
}

OPENFACE_BIN = os.environ.get("OPENFACE_PATH") or shutil.which("FeatureExtraction")
OPENFACE_POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
OPENFACE_GAZE_COLS = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]
OPENFACE_AU_COLS = [
    f"AU{n:02d}_r" for n in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
] + [
    f"AU{n:02d}_c" for n in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
]
OPENFACE_FEATURE_COLS = OPENFACE_POSE_COLS + OPENFACE_GAZE_COLS + OPENFACE_AU_COLS


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


def run_openface(video_path: str) -> pd.DataFrame:
    if not OPENFACE_BIN:
        raise RuntimeError(
            "Nie znaleziono OpenFace FeatureExtraction (ustaw OPENFACE_PATH lub dodaj do PATH)"
        )

    out_dir = tempfile.mkdtemp()
    try:
        try:
            subprocess.run(
                [OPENFACE_BIN, "-f", video_path, "-out_dir", out_dir, "-aus", "-pose", "-gaze", "-quiet"],
                capture_output=True,
                check=True,
                cwd=str(Path(OPENFACE_BIN).parent),
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
            raise ValueError(f"OpenFace zakonczyl sie bledem: {stderr[:300] or exc}") from exc
        csv_files = list(Path(out_dir).glob("*.csv"))
        if not csv_files:
            raise ValueError("OpenFace nie zwrocil pliku CSV")
        df = pd.read_csv(csv_files[0])
        df.columns = [c.strip() for c in df.columns]
        return df
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def extract_video_features(window_frames: pd.DataFrame) -> dict:
    cols = [c for c in OPENFACE_FEATURE_COLS if c in window_frames.columns]
    means = window_frames[cols].mean()
    return {name: round(float(value), 6) for name, value in means.items()}


def extract_eeg_features(window: np.ndarray, fs: float) -> dict:
    data = window.T[np.newaxis, :, :].astype(np.float64)
    features = extract_features(
        data, fs, EEG_MNE_FUNCS, funcs_params=EEG_MNE_FUNC_PARAMS, return_as_df=True
    )
    row = features.iloc[0]
    return {f"{func}_{name}": round(float(value), 6) for (func, name), value in row.items()}


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
    finally:
        cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"Nieprawidlowy fps dla {object_name}")
    window_len = int(round(WINDOW_SECONDS * fps))
    if window_len <= 0:
        raise ValueError(f"Nieprawidlowa dlugosc okna dla {object_name}")

    df = run_openface(video_path)
    if "success" in df.columns:
        df = df[df["success"] == 1]
    if df.empty or "frame" not in df.columns:
        return []

    trial_id = trial_number_from_object_name(object_name)
    df = df.copy()
    df["window_idx"] = (df["frame"] - 1) // window_len

    rows = []
    for window_idx, window_frames in df.groupby("window_idx"):
        if window_idx >= MAX_WINDOWS_PER_TRIAL:
            continue
        features = extract_video_features(window_frames)
        rows.append({"window_id": f"{trial_id}_{int(window_idx)}", **features})
    return rows


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
