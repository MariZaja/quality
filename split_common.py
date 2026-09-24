import io

import pandas as pd
from minio import Minio
from minio.error import S3Error
from sklearn.model_selection import train_test_split

from minio_common import SOURCE_BUCKET

ANNOTATIONS_PREFIX = "05_annotations_model"
QUALITY_BUCKET = "gold"
QUALITY_PREFIX = "data_quality_model"
QUALITY_COLS = ("audio_quality", "video_quality", "eeg_quality")

TEST_SIZE = 0.2
RANDOM_STATE = 42

SPLIT_COL = "split"
TRAIN = "train"
TEST = "test"


def _get_csv(client: Minio, bucket: str, object_name: str) -> pd.DataFrame | None:
    try:
        response = client.get_object(bucket, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    try:
        return pd.read_csv(io.BytesIO(response.read()))
    finally:
        response.close()
        response.release_conn()


def trial_id(window_ids: pd.Series) -> pd.Series:
    # window_id ma postac "{trial_id}_{window_idx}".
    return window_ids.astype(str).str.split("_").str[0]


def build_entity_split(client: Minio, eid: str) -> pd.DataFrame | None:
    annotations = _get_csv(client, SOURCE_BUCKET, f"{ANNOTATIONS_PREFIX}/{eid}_annotations.csv")
    if annotations is None or annotations.empty:
        return None
    windows = annotations[["window_id", "emotion_class"]].dropna(subset=["emotion_class"])
    if windows.empty:
        return None

    quality = _get_csv(client, QUALITY_BUCKET, f"{QUALITY_PREFIX}/{eid}_data_quality.csv")
    if quality is not None and not quality.empty:
        windows = windows.merge(quality[["window_id", *QUALITY_COLS]], on="window_id", how="left")
        q = windows[list(QUALITY_COLS)]
        windows["_low"] = (q.notna() & (q != "GOOD")).any(axis=1)
    else:
        print(f"[WARN] {eid}: brak raportu jakosci, podzial stratyfikowany tylko po emocji.")
        windows["_low"] = False

    windows["_trial"] = trial_id(windows["window_id"])
    trials = windows.groupby("_trial").agg(
        emotion=("emotion_class", "first"),
        low=("_low", "any"),
    ).reset_index()

    strat_keys = [
        trials["emotion"].astype(str) + "_" + trials["low"].map({True: "LOW", False: "GOOD"}),
        trials["emotion"].astype(str),
        None,
    ]
    for strat in strat_keys:
        try:
            train_trials, _ = train_test_split(
                trials["_trial"], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
            )
            break
        except ValueError as exc:
            print(f"[WARN] {eid}: podzial stratyfikowany niemozliwy ({exc}), probuje slabszej stratyfikacji.")
    else:
        return None

    train_trials = set(train_trials)
    split = windows["_trial"].map(lambda t: TRAIN if t in train_trials else TEST)
    return pd.DataFrame({"window_id": windows["window_id"].values, SPLIT_COL: split.values})
