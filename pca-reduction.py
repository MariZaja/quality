import io

import numpy as np
import pandas as pd
from minio import Minio
from minio.error import S3Error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from minio_common import (
    ENTITIES,
    build_arg_parser,
    get_minio_client,
    resolve_modalities,
)

FEATURES_BUCKET = "gold"
FEATURES_PREFIX = "feature_extraction_model"
TARGET_BUCKET = "gold"
TARGET_PREFIX = "pca_reduction_model"

N_COMPONENTS = 5


def parse_args():
    parser = build_arg_parser(
        "Redukcja wymiarow cech (PCA) dla plikow z feature_extraction_model, przed klasteryzacja.",
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

        if feature_cols is None:
            feature_cols = [c for c in features.columns if c != "window_id"]

        valid = features[feature_cols].notna().all(axis=1)
        if not valid.all():
            print(f"[WARN] {eid}/{modality}: pomijam {(~valid).sum()} okien z brakujacymi wartosciami cech.")
        features = features.loc[valid, ["window_id", *feature_cols]].reset_index(drop=True)
        if features.empty:
            continue

        entity_col = pd.Series(eid, index=features.index, name="entity")
        frames.append(pd.concat([entity_col, features], axis=1))

    if not frames:
        return None, None
    return pd.concat(frames, ignore_index=True), feature_cols


def fit_global_pca(combined: pd.DataFrame, feature_cols: list[str], modality: str) -> pd.DataFrame | None:
    X = combined[feature_cols].to_numpy(dtype=np.float32)

    n_components = min(N_COMPONENTS, len(feature_cols), X.shape[0] - 1)
    if n_components < N_COMPONENTS:
        print(
            f"[WARN] {modality}: za malo cech/probek na {N_COMPONENTS} skladowych PCA, "
            f"uzyto {n_components}."
        )
    if n_components < 1:
        print(f"[WARN] {modality}: za malo cech/probek do redukcji PCA, pomijam.")
        return None

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"{modality}: {n_components} skladowych PCA (wyjasniona wariancja: {explained:.4f})")

    result = pd.DataFrame(reduced, columns=[f"pca_{i + 1}" for i in range(n_components)])
    result.insert(0, "window_id", combined["window_id"].values)
    result.insert(0, "entity", combined["entity"].values)
    return result


def save_pca_report(client: Minio, eid: str, modality: str, df: pd.DataFrame) -> str:
    object_name = f"{TARGET_PREFIX}/{eid}/{modality}/{eid}_{modality}_pca.csv"

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    payload = buffer.getvalue().encode("utf-8")
    client.put_object(
        TARGET_BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="text/csv",
    )
    return object_name


def run_pca_reduction(client: Minio, modality: str, entities: list[str]) -> None:
    combined, feature_cols = collect_entities_data(client, modality, entities)
    if combined is None:
        print(f"[WARN] {modality}: brak danych do redukcji PCA, pomijam.")
        return

    result = fit_global_pca(combined, feature_cols, modality)
    if result is None:
        return

    for eid, group in result.groupby("entity", sort=False):
        saved_path = save_pca_report(client, eid, modality, group.drop(columns="entity"))
        print(f"{eid}/{modality}: zapisano {TARGET_BUCKET}/{saved_path} ({len(group)} okien)")


def main() -> None:
    args = parse_args()
    modalities = resolve_modalities(args.modality)
    entities = ENTITIES

    print(f"Modalnosci: {list(modalities)}")
    print(f"Entities: {entities[0]}..{entities[-1]} ({len(entities)} szt.)")

    client = get_minio_client()

    for modality in modalities:
        run_pca_reduction(client, modality, entities)


if __name__ == "__main__":
    main()
