import argparse
import io

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

N_BOOTSTRAP_DEFAULT = 100
N_PERMUTATIONS_DEFAULT = 100
MIN_CLUSTER_SIZE = 3
MAX_CLUSTERS = 10
RANDOM_SEED = 42


def parse_args():
    parser = build_arg_parser(
        "Test permutacyjny dla klasteryzacji po PCA: niszczy strukture miedzyosobnicza "
        "(permutacja kazdej cechy niezaleznie miedzy uczestnikami) i porownuje realny "
        "bootstrap ARI z rozkladem ARI pod brakiem struktury, dla ustalonego d."
    )
    parser.add_argument(
        "--d", type=int, required=True,
        help="Liczba wymiarow PCA do uzycia (pierwsze d kolumn: pca_1..pca_d).",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Wymuszona liczba klastrow. Domyslnie: najlepsze k wg silhouette "
        f"(k=2..{MAX_CLUSTERS}, najmniejszy klaster >= {MIN_CLUSTER_SIZE}), tak jak w pca-clustering.py.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT,
        help=f"Liczba probek bootstrap przy kazdym pomiarze ARI. Domyslnie {N_BOOTSTRAP_DEFAULT}.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=N_PERMUTATIONS_DEFAULT,
        help=f"Liczba powtorzen permutacji (rozklad null). Domyslnie {N_PERMUTATIONS_DEFAULT}.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Ziarno losowosci. Domyslnie {RANDOM_SEED}.",
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Zapisz wynik testu do "
        f"{TARGET_BUCKET}/{TARGET_PREFIX}/{{modality}}/{{modality}}_permutation_test_d{{d}}.csv",
    )
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


def bootstrap_ari_stability(
    X: np.ndarray, k: int, reference_labels: np.ndarray, n_bootstrap: int, seed: int
) -> float:
    n_samples = X.shape[0]
    rng = np.random.default_rng(seed)

    ari_scores = []
    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, n_samples, n_samples)
        model = KMeans(n_clusters=k, random_state=seed + b, n_init=10).fit(X[boot_idx])
        boot_labels = model.predict(X)
        ari_scores.append(adjusted_rand_score(reference_labels, boot_labels))

    return float(np.mean(ari_scores))


def find_best_k(
    X: np.ndarray, seed: int, max_k: int = MAX_CLUSTERS, min_cluster_size: int = MIN_CLUSTER_SIZE
) -> tuple[np.ndarray | None, int | None, float]:
    best_labels, best_k, best_score = None, None, -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        if np.bincount(labels).min() < min_cluster_size:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_labels, best_k, best_score = labels, k, score
    return best_labels, best_k, best_score


def permute_features_independently(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X_perm = np.empty_like(X)
    for j in range(X.shape[1]):
        X_perm[:, j] = rng.permutation(X[:, j])
    return X_perm


def run_permutation_test(
    X: np.ndarray, k: int, n_bootstrap: int, n_permutations: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    null_scores = []
    for p in range(n_permutations):
        X_perm = permute_features_independently(X, rng)
        perm_labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_perm)
        ari = bootstrap_ari_stability(X_perm, k, perm_labels, n_bootstrap=n_bootstrap, seed=seed + p + 1)
        null_scores.append(ari)
        print(f"    permutacja {p + 1}/{n_permutations}: bootstrap_ari={ari:.4f}")
    return np.array(null_scores)


def save_report(
    client: Minio,
    modality: str,
    d: int,
    k: int,
    real_ari: float,
    null_scores: np.ndarray,
    p_value: float,
    percentile: float,
) -> str:
    object_name = f"{TARGET_PREFIX}/{modality}/{modality}_permutation_test_d{d}.csv"

    rows = [{"type": "real", "permutation": None, "d": d, "k": k, "ari": real_ari}]
    rows += [
        {"type": "null", "permutation": i + 1, "d": d, "k": k, "ari": float(score)}
        for i, score in enumerate(null_scores)
    ]
    df = pd.DataFrame(rows)
    df["null_mean"] = null_scores.mean()
    df["null_std"] = null_scores.std()
    df["p_value"] = p_value
    df["real_percentile"] = percentile

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


def process_modality(client: Minio, modality: str, entities: list[str], args: argparse.Namespace) -> None:
    means_df = collect_participant_means(client, modality, entities)
    if len(means_df) < 3:
        print(f"[WARN] {modality}: za malo uczestnikow ({len(means_df)}) do klasteryzacji, pomijam.")
        return

    pca_cols_all = [c for c in means_df.columns if c != "entity"]
    if args.d < 1 or args.d > len(pca_cols_all):
        print(
            f"[WARN] {modality}: zadano d={args.d}, dostepnych kolumn PCA={len(pca_cols_all)}, pomijam."
        )
        return
    pca_cols = pca_cols_all[: args.d]

    X = StandardScaler().fit_transform(means_df[pca_cols].to_numpy(dtype=float))

    if args.k is not None:
        k = args.k
        real_labels = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(X)
        score = silhouette_score(X, real_labels)
    else:
        real_labels, k, score = find_best_k(X, seed=args.seed)
        if real_labels is None:
            print(
                f"[WARN] {modality}: brak k w zakresie 2..{MAX_CLUSTERS} z najmniejszym "
                f"klastrem >= {MIN_CLUSTER_SIZE}, pomijam."
            )
            return

    real_ari = bootstrap_ari_stability(X, k, real_labels, n_bootstrap=args.n_bootstrap, seed=args.seed)
    print(
        f"\n{modality}: d={args.d}, n_uczestnikow={X.shape[0]}, k={k}, silhouette={score:.4f}, "
        f"realny bootstrap_ari={real_ari:.4f}"
    )

    print(
        f"{modality}: test permutacyjny -- {args.n_permutations} powtorzen, "
        f"n_bootstrap={args.n_bootstrap}, ten sam k={k}..."
    )
    null_scores = run_permutation_test(X, k, args.n_bootstrap, args.n_permutations, args.seed)

    b = int(np.sum(null_scores >= real_ari))
    p_value = (b + 1) / (args.n_permutations + 1)
    percentile = float((null_scores < real_ari).mean() * 100)

    print(f"\n{modality}: PODSUMOWANIE (d={args.d}, k={k})")
    print(f"  realny bootstrap ARI    = {real_ari:.4f}")
    print(
        f"  null (permutacja) ARI   : mean={null_scores.mean():.4f}, std={null_scores.std():.4f}, "
        f"min={null_scores.min():.4f}, max={null_scores.max():.4f}"
    )
    print(f"  realny ARI na {percentile:.1f} percentylu rozkladu null")
    print(f"  p-value (frakcja null >= real) = {p_value:.4f}")
    if p_value < 0.05:
        print("  -> realne ARI istotnie powyzej rozkladu null (struktura wyglada na realna).")
    else:
        print("  -> realne ARI miesci sie w zasiegu rozkladu null (brak jasnego dowodu na realna strukture).")

    if args.save_report:
        saved_path = save_report(client, modality, args.d, k, real_ari, null_scores, p_value, percentile)
        print(f"{modality}: zapisano {TARGET_BUCKET}/{saved_path}")


def main() -> None:
    args = parse_args()
    modalities = resolve_modalities(args.modality)
    entities = resolve_entities(args.entity)

    print(f"Modalnosci: {list(modalities)}")
    print(f"Entities: {entities[0]}..{entities[-1]} ({len(entities)} szt.)")
    print(f"d={args.d}, n_bootstrap={args.n_bootstrap}, n_permutations={args.n_permutations}, seed={args.seed}")

    client = get_minio_client()

    for modality in modalities:
        process_modality(client, modality, entities, args)


if __name__ == "__main__":
    main()
