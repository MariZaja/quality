import io
import math

import numpy as np
import pandas as pd
from minio import Minio
from minio.error import S3Error
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from minio_common import (
    build_arg_parser,
    get_minio_client,
    resolve_entities,
    resolve_modalities,
)

SOURCE_BUCKET = "gold"
SOURCE_PREFIX = "pca_reduction_model"
TARGET_BUCKET = "gold"
TARGET_PREFIX = "clustering_model"

N_BOOTSTRAP = 100
MIN_CLUSTER_SIZE = 3
MAX_CLUSTERS = 10


def parse_args():
    parser = build_arg_parser("Klasteryzacja uczestnikow na podstawie usrednionych cech PCA.")
    return parser.parse_args()


def get_object_bytes(client: Minio, bucket: str, object_name: str) -> bytes:
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def load_participant_pca_means(client: Minio, eid: str, modality: str) -> pd.Series | None:
    object_name = f"{SOURCE_PREFIX}/{eid}/{modality}/{eid}_{modality}_pca.csv"
    try:
        data = get_object_bytes(client, SOURCE_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None

    df = pd.read_csv(io.BytesIO(data))
    if df.empty:
        print(f"[WARN] {eid}/{modality}: pusty plik PCA, pomijam.")
        return None

    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    return df[pca_cols].mean()


def collect_participant_means(client: Minio, modality: str, entities: list[str]) -> pd.DataFrame:
    rows = []
    for eid in entities:
        means = load_participant_pca_means(client, eid, modality)
        if means is None:
            continue
        row = {"entity": eid, **means.to_dict()}
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ari_stability(X: np.ndarray, k: int, reference_labels: np.ndarray, n_bootstrap: int = N_BOOTSTRAP) -> float:
    n_samples = X.shape[0]
    rng = np.random.default_rng(42)

    ari_scores = []
    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, n_samples, n_samples)
        model = KMeans(n_clusters=k, random_state=42 + b, n_init=10).fit(X[boot_idx])
        boot_labels = model.predict(X)
        ari_scores.append(adjusted_rand_score(reference_labels, boot_labels))

    return float(np.mean(ari_scores))


def find_best_clustering(X: np.ndarray, entities: list[str], modality: str) -> tuple[np.ndarray, int, float, float]:
    n_samples = X.shape[0]
    # max_k = max(2, min(n_samples - 1, round(math.sqrt(n_samples / 2))))
    max_k = MAX_CLUSTERS

    best_labels, best_k, best_score, best_ari = None, None, -1.0, 0.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

        clusters = {}
        for eid, label in zip(entities, labels):
            clusters.setdefault(int(label), []).append(eid)
        clusters_str = "; ".join(
            f"{cid}: {', '.join(members)}" for cid, members in sorted(clusters.items())
        )

        smallest_cluster = min(len(members) for members in clusters.values())
        if smallest_cluster < MIN_CLUSTER_SIZE:
            print(
                f"{modality}: k={k}, pomijam (najmniejszy klaster ma {smallest_cluster} < "
                f"{MIN_CLUSTER_SIZE} elementow), klastry: {clusters_str}"
            )
            continue

        score = silhouette_score(X, labels)
        ari = bootstrap_ari_stability(X, k, labels)
        print(
            f"{modality}: k={k}, silhouette_score={score:.4f}, bootstrap_ari={ari:.4f}, klastry: {clusters_str}"
        )

        if score > best_score:
            best_labels, best_k, best_score, best_ari = labels, k, score, ari

    return best_labels, best_k, best_score, best_ari


def save_clustering_report(client: Minio, modality: str, df: pd.DataFrame, pca_cols: list[str]) -> str:
    object_name = f"{TARGET_PREFIX}/{modality}/{modality}_clustering.csv"

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, columns=["entity", *pca_cols, "cluster"])

    payload = buffer.getvalue().encode("utf-8")
    client.put_object(
        TARGET_BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="text/csv",
    )
    return object_name


def cluster_modality(client: Minio, modality: str, entities: list[str]) -> None:
    means_df = collect_participant_means(client, modality, entities)
    if len(means_df) < 3:
        print(f"[WARN] {modality}: za malo uczestnikow ({len(means_df)}) do klasteryzacji, pomijam.")
        return

    pca_cols = [c for c in means_df.columns if c != "entity"]
    X = StandardScaler().fit_transform(means_df[pca_cols].to_numpy(dtype=float))
    labels, best_k, best_score, best_ari = find_best_clustering(X, means_df["entity"].tolist(), modality)
    if labels is None:
        print(
            f"[WARN] {modality}: brak podzialu z klastrami >= {MIN_CLUSTER_SIZE} elementow, pomijam."
        )
        return

    means_df["cluster"] = labels
    print(
        f"{modality}: najlepsze k={best_k}, silhouette_score={best_score:.4f}, "
        f"bootstrap_ari={best_ari:.4f}"
    )

    saved_path = save_clustering_report(client, modality, means_df, pca_cols)
    print(f"{modality}: zapisano {TARGET_BUCKET}/{saved_path} ({len(means_df)} uczestnikow)")


def main() -> None:
    args = parse_args()
    modalities = resolve_modalities(args.modality)
    entities = resolve_entities(args.entity)

    print(f"Modalnosci: {list(modalities)}")
    print(f"Entities: {entities[0]}..{entities[-1]} ({len(entities)} szt.)")

    client = get_minio_client()

    for modality in modalities:
        cluster_modality(client, modality, entities)


if __name__ == "__main__":
    main()
