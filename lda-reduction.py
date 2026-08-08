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
    resolve_modalities,
)

FEATURES_BUCKET = "gold"
FEATURES_PREFIX = "feature_extraction_model"
ANNOTATIONS_PREFIX = "05_annotations_model"
CLUSTERING_BUCKET = "gold"
CLUSTERING_PREFIX = "clustering_model"
TARGET_BUCKET = "gold"
TARGET_PREFIX = "lda_reduction_model"

N_COMPONENTS = 3
REPORT_FIELDNAMES = ["window_id", "lda_1", "lda_2", "lda_3", "emotion"]


def parse_args():
    parser = build_arg_parser(
        "Redukcja wymiarow cech (LDA) dla plikow z feature_extraction_model.",
        include_entity=False,
    )
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
    df = pd.read_csv(io.BytesIO(data))
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def load_annotations(client: Minio, eid: str) -> pd.DataFrame | None:
    object_name = f"{ANNOTATIONS_PREFIX}/{eid}_annotations.csv"
    try:
        data = get_object_bytes(client, SOURCE_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    return pd.read_csv(io.BytesIO(data))[["window_id", "emotion_class"]]


def load_cluster_assignments(client: Minio, modality: str) -> pd.DataFrame | None:
    object_name = f"{CLUSTERING_PREFIX}/{modality}/{modality}_clustering.csv"
    try:
        data = get_object_bytes(client, CLUSTERING_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    return pd.read_csv(io.BytesIO(data))[["entity", "cluster"]]


def merge_entity_data(
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
    merged = merged.loc[valid, ["window_id", *feature_cols, "emotion_class"]].reset_index(drop=True)
    if merged.empty:
        return None

    entity_col = pd.Series(eid, index=merged.index, name="entity")
    return pd.concat([entity_col, merged], axis=1)


def collect_entities_data(
    client: Minio, modality: str, entities: list[str]
) -> tuple[pd.DataFrame, list[str]] | tuple[None, None]:
    frames = []
    feature_cols = None
    for eid in entities:
        features = load_features(client, eid, modality)
        if features is None or features.empty:
            print(f"[WARN] Brak cech {modality} dla {eid}, pomijam.")
            continue

        annotations = load_annotations(client, eid)
        if annotations is None or annotations.empty:
            print(f"[WARN] Brak etykiet emocji dla {eid}, pomijam.")
            continue

        if feature_cols is None:
            feature_cols = [c for c in features.columns if c != "window_id"]

        merged = merge_entity_data(features, annotations, eid, modality)
        if merged is None:
            continue
        frames.append(merged)

    if not frames:
        return None, None
    return pd.concat(frames, ignore_index=True), feature_cols


def fit_cluster_lda(
    combined: pd.DataFrame, feature_cols: list[str], modality: str, cluster_id: int
) -> pd.DataFrame | None:
    X = combined[feature_cols].to_numpy(dtype=np.float32)
    y = combined["emotion_class"]

    n_classes = y.nunique()
    n_components = min(N_COMPONENTS, len(feature_cols), n_classes - 1)
    if n_components < 1:
        print(
            f"[WARN] {modality}/klaster {cluster_id}: za malo klas emocji ({n_classes}) "
            "do redukcji LDA, pomijam."
        )
        return None

    X_scaled = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    reduced = lda.fit_transform(X_scaled, y)

    if n_components < N_COMPONENTS:
        print(
            f"[WARN] {modality}/klaster {cluster_id}: dostepne tylko {n_components} skladowe LDA "
            f"(klas emocji: {n_classes}), pozostale kolumny wypelniono zerami."
        )
        reduced = np.hstack([reduced, np.zeros((reduced.shape[0], N_COMPONENTS - n_components))])

    result = pd.DataFrame(reduced, columns=[f"lda_{i + 1}" for i in range(N_COMPONENTS)])
    result.insert(0, "window_id", combined["window_id"].values)
    result.insert(0, "entity", combined["entity"].values)
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


def run_lda_reduction(client: Minio, modality: str) -> None:
    clusters = load_cluster_assignments(client, modality)
    if clusters is None or clusters.empty:
        print(f"[WARN] {modality}: brak przypisania do klastrow, pomijam.")
        return

    for cluster_id, cluster_group in clusters.groupby("cluster"):
        entities = cluster_group["entity"].tolist()
        combined, feature_cols = collect_entities_data(client, modality, entities)
        if combined is None:
            print(f"[WARN] {modality}/klaster {cluster_id}: brak danych do redukcji LDA, pomijam.")
            continue

        result = fit_cluster_lda(combined, feature_cols, modality, cluster_id)
        if result is None:
            continue

        for eid, group in result.groupby("entity", sort=False):
            saved_path = save_lda_report(client, eid, modality, group.drop(columns="entity"))
            print(
                f"{eid}/{modality} (klaster {cluster_id}): "
                f"zapisano {TARGET_BUCKET}/{saved_path} ({len(group)} okien)"
            )


def main() -> None:
    args = parse_args()
    modalities = resolve_modalities(args.modality)

    print(f"Modalnosci: {list(modalities)}")

    client = get_minio_client()

    for modality in modalities:
        run_lda_reduction(client, modality)


if __name__ == "__main__":
    main()
