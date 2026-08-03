import io

import numpy as np
import pandas as pd
from minio import Minio
from minio.error import S3Error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from minio_common import (
    SOURCE_BUCKET,
    build_arg_parser,
    get_minio_client,
    resolve_entities,
    resolve_modalities,
)

FEATURES_BUCKET = "gold"
FEATURES_PREFIX = "feature_extraction_model"
ANNOTATIONS_PREFIX = "05_annotations_model"
TARGET_BUCKET = "gold"
TARGET_PREFIX = "lda_reduction_model"

N_COMPONENTS = 3
REPORT_FIELDNAMES = ["window_id", "lda_1", "lda_2", "lda_3", "emotion"]


def parse_args():
    parser = build_arg_parser("Redukcja wymiarow cech (LDA) dla plikow z feature_extraction_model.")
    return parser.parse_args()


def get_object_bytes(client: Minio, bucket: str, object_name: str) -> bytes:
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def load_features(client: Minio, eid: str, modality: str) -> pd.DataFrame | None:
    object_name = f"{FEATURES_PREFIX}/{eid}/{modality}/{eid}_{modality}_features.csv"
    try:
        data = get_object_bytes(client, FEATURES_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    return pd.read_csv(io.BytesIO(data))


def load_annotations(client: Minio, eid: str) -> pd.DataFrame | None:
    object_name = f"{ANNOTATIONS_PREFIX}/{eid}_annotations.csv"
    try:
        data = get_object_bytes(client, SOURCE_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    return pd.read_csv(io.BytesIO(data))[["window_id", "emotion_class"]]


def reduce_with_lda(
    features: pd.DataFrame, annotations: pd.DataFrame, eid: str, modality: str
) -> pd.DataFrame | None:
    merged = features.merge(annotations, on="window_id", how="inner")
    if merged.empty:
        print(f"[WARN] {eid}/{modality}: brak wspolnych window_id cech i etykiet, pomijam.")
        return None

    feature_cols = [c for c in features.columns if c != "window_id"]
    valid = merged[feature_cols].notna().all(axis=1)
    if not valid.all():
        print(f"[WARN] {eid}/{modality}: pomijam {(~valid).sum()} okien z brakujacymi wartosciami cech.")
    merged = merged.loc[valid].reset_index(drop=True)

    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["emotion_class"]

    n_classes = y.nunique()
    n_components = min(N_COMPONENTS, len(feature_cols), n_classes - 1)
    if n_components < 1:
        print(f"[WARN] {eid}/{modality}: za malo klas emocji ({n_classes}) do redukcji LDA, pomijam.")
        return None

    X_scaled = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    reduced = lda.fit_transform(X_scaled, y)

    if n_components < N_COMPONENTS:
        print(
            f"[WARN] {eid}/{modality}: dostepne tylko {n_components} skladowe LDA "
            f"(klas emocji: {n_classes}), pozostale kolumny wypelniono zerami."
        )
        reduced = np.hstack([reduced, np.zeros((reduced.shape[0], N_COMPONENTS - n_components))])

    result = pd.DataFrame(reduced, columns=[f"lda_{i + 1}" for i in range(N_COMPONENTS)])
    result.insert(0, "window_id", merged["window_id"])
    result["emotion"] = y.values
    return result


def save_lda_report(client: Minio, eid: str, modality: str, df: pd.DataFrame) -> str:
    object_name = f"{TARGET_PREFIX}/{eid}/{modality}/{eid}_{modality}_lda.csv"

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, columns=REPORT_FIELDNAMES)

    payload = buffer.getvalue().encode("utf-8")
    client.put_object(
        TARGET_BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="text/csv",
    )
    return object_name


def run_lda_reduction(client: Minio, modality: str, entities: list[str]) -> None:
    for eid in entities:
        features = load_features(client, eid, modality)
        if features is None or features.empty:
            print(f"[WARN] Brak cech {modality} dla {eid}, pomijam.")
            continue

        annotations = load_annotations(client, eid)
        if annotations is None or annotations.empty:
            print(f"[WARN] Brak etykiet emocji dla {eid}, pomijam.")
            continue

        result = reduce_with_lda(features, annotations, eid, modality)
        if result is None:
            continue

        saved_path = save_lda_report(client, eid, modality, result)
        print(f"{eid}/{modality}: zapisano {TARGET_BUCKET}/{saved_path} ({len(result)} okien)")


def main() -> None:
    args = parse_args()
    modalities = resolve_modalities(args.modality)
    entities = resolve_entities(args.entity)

    print(f"Modalnosci: {list(modalities)}")
    print(f"Entities: {entities[0]}..{entities[-1]} ({len(entities)} szt.)")

    client = get_minio_client()

    for modality in modalities:
        run_lda_reduction(client, modality, entities)


if __name__ == "__main__":
    main()
