import io

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from scipy.stats import norm as scipy_norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from minio import Minio
from minio.error import S3Error

from pgmpy.global_vars import config
config.set_backend("torch")

from pgmpy.models import FunctionalBayesianNetwork
from pgmpy.factors.hybrid import FunctionalCPD

from minio_common import (
    ENTITIES,
    SOURCE_BUCKET,
    get_minio_client,
    resolve_entities,
)

ANNOTATIONS_BUCKET = SOURCE_BUCKET
ANNOTATIONS_PREFIX = "05_annotations_model"
QUALITY_BUCKET = "gold"
QUALITY_PREFIX = "data_quality_model"
LDA_BUCKET = "gold"
LDA_PREFIX = "lda_reduction_model"
LDA_EXPERIMENTS = ("global", "entity", "cluster")

CLUSTERING_BUCKET = "gold"
CLUSTERING_PREFIX = "clustering_model"
CLUSTER_SOURCE_MODALITY = "video"

RESULTS_BUCKET = "gold"
RESULTS_PREFIX = "model"

N_LDA_COMPONENTS = 3


# -- Encodings --------------------------------------------------------------

E_STATES = ["Angry", "Sad", "Happy", "Calm"]
Q_STATES = ["BAD", "GOOD"]

E_BASE = "Calm"
Q_BASE = "GOOD"

E_ENC = {s: i for i, s in enumerate(E_STATES)}
Q_ENC = {s: i for i, s in enumerate(Q_STATES)}
E_DEC = {i: s for s, i in E_ENC.items()}
Q_DEC = {i: s for s, i in Q_ENC.items()}

BAD_IDX = Q_ENC["BAD"]

EMOTION_MAP = {"Anger": "Angry", "Sadness": "Sad", "Happiness": "Happy", "Calm": "Calm"}

MODALITIES = {
    "audio": {"q": "Q_audio", "quality_col": "audio_quality",
              "v_cols": [f"V_audio_{i}" for i in range(1, N_LDA_COMPONENTS + 1)]},
    "video": {"q": "Q_video", "quality_col": "video_quality",
              "v_cols": [f"V_video_{i}" for i in range(1, N_LDA_COMPONENTS + 1)]},
    "eeg":   {"q": "Q_eeg",   "quality_col": "eeg_quality",
              "v_cols": [f"V_eeg_{i}"   for i in range(1, N_LDA_COMPONENTS + 1)]},
}

N_E = len(E_STATES)   # 4
N_Q = len(Q_STATES)   # 2


# -- Data loading (MinIO) ----------------------------------------------------

def get_object_bytes(client: Minio, bucket: str, object_name: str) -> bytes:
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def load_annotations(client: Minio, eid: str) -> pd.DataFrame | None:
    object_name = f"{ANNOTATIONS_PREFIX}/{eid}_annotations.csv"
    try:
        data = get_object_bytes(client, ANNOTATIONS_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    return pd.read_csv(io.BytesIO(data))[["window_id", "emotion_class"]]


def load_quality(client: Minio, eid: str) -> pd.DataFrame | None:
    object_name = f"{QUALITY_PREFIX}/{eid}_data_quality.csv"
    try:
        data = get_object_bytes(client, QUALITY_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    return pd.read_csv(io.BytesIO(data))


def load_lda_features(client: Minio, eid: str, modality: str, lda_experiment: str) -> pd.DataFrame | None:
    object_name = f"{LDA_PREFIX}/{lda_experiment}/{eid}/{modality}/{eid}_{modality}_lda.csv"
    try:
        data = get_object_bytes(client, LDA_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    lda_cols = [f"lda_{i}" for i in range(1, N_LDA_COMPONENTS + 1)]
    return pd.read_csv(io.BytesIO(data))[["window_id", *lda_cols]]


def load_cluster_assignments(client: Minio) -> pd.DataFrame | None:
    object_name = (
        f"{CLUSTERING_PREFIX}/{CLUSTER_SOURCE_MODALITY}/{CLUSTER_SOURCE_MODALITY}_clustering.csv"
    )
    try:
        data = get_object_bytes(client, CLUSTERING_BUCKET, object_name)
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return None
    return pd.read_csv(io.BytesIO(data))[["entity", "cluster"]]


def load_entity_data(client: Minio, eid: str, lda_experiment: str) -> pd.DataFrame | None:
    annotations = load_annotations(client, eid)
    if annotations is None or annotations.empty:
        print(f"[WARN] Brak etykiet emocji dla {eid}, pomijam.")
        return None

    quality = load_quality(client, eid)
    if quality is None or quality.empty:
        print(f"[WARN] Brak raportu jakosci dla {eid}, pomijam.")
        return None

    base = annotations.merge(quality, on="window_id", how="left")
    base["E"] = base["emotion_class"].map(EMOTION_MAP)
    base = base.rename(columns={
        "audio_quality": "Q_audio",
        "video_quality": "Q_video",
        "eeg_quality":   "Q_eeg",
    })
    base = base[["window_id", "E", "Q_audio", "Q_video", "Q_eeg"]]

    for modality, cfg in MODALITIES.items():
        features = load_lda_features(client, eid, modality, lda_experiment)
        if features is None:
            continue
        rename = {f"lda_{i}": cfg["v_cols"][i - 1] for i in range(1, N_LDA_COMPONENTS + 1)}
        base = base.merge(features.rename(columns=rename), on="window_id", how="left")

    base.insert(0, "entity", eid)
    return base


def load_data(client: Minio, entities: list[str], lda_experiment: str) -> pd.DataFrame:
    frames = []
    for eid in entities:
        entity_df = load_entity_data(client, eid, lda_experiment)
        if entity_df is None:
            continue
        frames.append(entity_df)

    if not frames:
        raise RuntimeError(
            "Brak danych do treningu FBN -- sprawdz "
            f"{ANNOTATIONS_BUCKET}/{ANNOTATIONS_PREFIX}, {QUALITY_BUCKET}/{QUALITY_PREFIX}, "
            f"{LDA_BUCKET}/{LDA_PREFIX}/{lda_experiment}."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["E"]).reset_index(drop=True)
    return combined


# -- Przygotowanie DataFrame do fit() ----------------------------------------

def prepare_fit_df(df: pd.DataFrame, use_quality: bool) -> pd.DataFrame:
    """Koduje E i Q na int. Jesli use_quality=False, kolumny Q sa pomijane."""
    out = df.copy()
    out["E"] = out["E"].map(E_ENC)
    if use_quality:
        for cfg in MODALITIES.values():
            q_col = cfg["q"]
            out[q_col] = out[q_col].map(
                lambda x: Q_ENC[x] if (pd.notna(x) and x in Q_ENC) else np.nan
            )
    return out


def expand_reference_effect(
    raw: torch.Tensor,
    n_states: int,
    base_idx: int,
) -> torch.Tensor:
    """
    Zamienia wektor dlugosci n_states - 1 na pelny wektor efektow,
    w ktorym efekt klasy bazowej wynosi 0.
    """
    parts = []
    raw_i = 0

    for i in range(n_states):
        if i == base_idx:
            parts.append(torch.tensor(0.0, dtype=raw.dtype, device=raw.device))
        else:
            parts.append(raw[raw_i])
            raw_i += 1

    return torch.stack(parts)


def expand_reference_effect_np(
    raw: np.ndarray,
    n_states: int,
    base_idx: int,
) -> np.ndarray:
    full = np.zeros(n_states)
    raw_i = 0

    for i in range(n_states):
        if i == base_idx:
            full[i] = 0.0
        else:
            full[i] = raw[raw_i]
            raw_i += 1

    return full


# -- CPD -- wezel E (wspolny dla obu modeli) ---------------------------------

def cpd_fn_E(_parents):
    """Marginalny rozklad emocji P(E) -- uczony jako simplex."""
    e_probs = pyro.param(
        "E_probs",
        torch.ones(N_E) / N_E,
        constraint=constraints.simplex,
    )
    return dist.Categorical(probs=e_probs)


# -- CPD -- wezel Q (tylko w modelu z quality) -------------------------------

def make_cpd_fn_Q(q_name: str):
    """Marginalny rozklad jakosci P(Q) -- Q niezalezne od E."""
    def fn(_parents):
        q_probs = pyro.param(
            f"{q_name}_probs",
            torch.ones(N_Q) / N_Q,
            constraint=constraints.simplex,
        )
        return dist.Categorical(probs=q_probs)
    return fn


# -- CPD -- wezel V z quality: mu = beta0 + betaE[E] + betaQ[Q], sigma = base_sigma * penalty^(Q==BAD) --

def make_cpd_fn_V_with_quality(v_name: str, q_name: str):
    """
    V zalezy od E i Q. Rodzice: [E, Q_mod].

    Jakosc Q wchodzi do CPD na dwa sposoby:
      * mu    = beta0 + betaE[E] + betaQ[Q]                -- addytywny bias zalezny od jakosci,
      * sigma = base_sigma * (penalty ** (Q==BAD))          -- poziom szumu zalezny od jakosci.

    Karze (penalty) podlega tylko stan BAD -- GOOD ma sigma=base_sigma. penalty jest
    ograniczony do [1.0, 1.5]. Gdy model nauczy sie penalty > 1, likelihood zaszumionej
    modalnosci jest plaski -> slabo rusza posteriorem -> modalnosc dyskontuje sie
    automatycznie (quality-aware down-weighting).
    """
    def fn(parents):
        beta_0 = pyro.param(f"{v_name}_beta_0", torch.tensor(0.0))

        beta_E_raw = pyro.param(
            f"{v_name}_beta_E_raw",
            torch.zeros(N_E - 1),
        )

        beta_Q_raw = pyro.param(
            f"{v_name}_beta_Q_raw",
            torch.zeros(N_Q - 1),
        )

        base_sigma = pyro.param(
            f"{v_name}_sigma_base",
            torch.tensor(1.0),
            constraint=constraints.positive,
        )
        penalty = pyro.param(
            f"{v_name}_penalty",
            torch.tensor(1.2),
            constraint=constraints.interval(1.0, 1.5),
        )

        e_base_idx = E_ENC[E_BASE]
        q_base_idx = Q_ENC[Q_BASE]

        beta_E = expand_reference_effect(beta_E_raw, N_E, e_base_idx)
        beta_Q = expand_reference_effect(beta_Q_raw, N_Q, q_base_idx)

        e_idx = parents["E"].long()
        q_idx = parents[q_name].long()

        mu    = beta_0 + beta_E[e_idx] + beta_Q[q_idx]
        sigma = base_sigma * (penalty ** (q_idx == BAD_IDX).float())

        return dist.Normal(mu, sigma)
    return fn


# -- CPD -- wezel V bez quality: mu = beta0 + betaE[E] -----------------------

def make_cpd_fn_V_no_quality(v_name: str):
    """V zalezy tylko od E. Rodzice: [E]. sigma skalarne (brak wezla Q)."""
    def fn(parents):
        beta_0 = pyro.param(f"{v_name}_beta_0", torch.tensor(0.0))

        beta_E_raw = pyro.param(
            f"{v_name}_beta_E_raw",
            torch.zeros(N_E - 1),
        )

        sigma = pyro.param(
            f"{v_name}_sigma",
            torch.tensor(1.0),
            constraint=constraints.positive,
        )

        e_base_idx = E_ENC[E_BASE]
        beta_E = expand_reference_effect(beta_E_raw, N_E, e_base_idx)

        e_idx = parents["E"].long()
        mu = beta_0 + beta_E[e_idx]

        return dist.Normal(mu, sigma)
    return fn


# -- Budowanie modelu ---------------------------------------------------------

def build_modality_model(modality: str, use_quality: bool) -> FunctionalBayesianNetwork:
    """
    Buduje FBN dla jednej modalnosci:

    use_quality=True  -> E-->V_mod<--Q_mod
    use_quality=False -> E-->V_mod

    Modalnosci nie sa polaczone w grafie (audio/video/eeg sa warunkowo
    niezalezne przy danym E), wiec kazda jest uczona osobnym fit() na
    swoim wlasnym, maksymalnym podzbiorze wierszy -- patrz train().
    """
    cfg = MODALITIES[modality]
    q_col = cfg["q"]

    edges = []
    for v_name in cfg["v_cols"]:
        edges.append(("E", v_name))
        if use_quality:
            edges.append((q_col, v_name))

    model = FunctionalBayesianNetwork(edges)
    model.add_cpds(FunctionalCPD("E", fn=cpd_fn_E))

    if use_quality:
        model.add_cpds(FunctionalCPD(q_col, fn=make_cpd_fn_Q(q_col)))

    for v_name in cfg["v_cols"]:
        if use_quality:
            model.add_cpds(
                FunctionalCPD(
                    v_name,
                    fn=make_cpd_fn_V_with_quality(v_name, q_col),
                    parents=["E", q_col],
                )
            )
        else:
            model.add_cpds(
                FunctionalCPD(
                    v_name,
                    fn=make_cpd_fn_V_no_quality(v_name),
                    parents=["E"],
                )
            )

    assert model.check_model(), f"Graf FBN dla modalnosci '{modality}' jest niepoprawny!"
    return model


# -- Uczenie -------------------------------------------------------------------

def train(
    train_df: pd.DataFrame,
    use_quality: bool,
    num_steps: int = 5000,
    lr: float = 0.005,
    seed: int = 7,
) -> dict[str, torch.Tensor]:
    """
    Uczy CPD kazdej modalnosci osobnym fit() (patrz build_modality_model) --
    kazda modalnosc widzi wszystkie okna, dla ktorych MA dane, niezaleznie
    od tego, czy inne modalnosci maja braki w tych oknach.

    E_probs to MLE marginalnego P(E): znormalizowane liczebnosci klas w
    train_df. To dokladny (nie przyblizony) wzor dla kategorycznego
    rozkladu bez rodzicow, wiec SVI nie jest tu potrzebne.
    """
    fit_df_full = prepare_fit_df(train_df, use_quality)

    e_counts = train_df["E"].value_counts()
    e_probs = np.array([e_counts.get(e, 0) for e in E_STATES], dtype=float)
    e_probs /= e_probs.sum()

    params: dict[str, torch.Tensor] = {
        "E_probs": torch.tensor(e_probs, dtype=config.get_dtype()),
    }

    mode_label = "z quality (E->V<-Q)" if use_quality else "bez quality (E->V)"

    for modality, cfg in MODALITIES.items():
        q_col = cfg["q"]
        v_cols = cfg["v_cols"]
        drop_cols = ([q_col] + v_cols) if use_quality else v_cols

        mod_fit_df = fit_df_full.dropna(subset=drop_cols).reset_index(drop=True)
        print(f"Uczenie FBN [{modality}, {mode_label}] na {len(mod_fit_df)} oknach...")

        pyro.set_rng_seed(seed)
        pyro.clear_param_store()

        model = build_modality_model(modality, use_quality)
        mod_params = model.fit(
            mod_fit_df,
            estimator="SVI",
            optimizer=pyro.optim.Adam({"lr": lr}),
            num_steps=num_steps,
            seed=seed,
        )

        for k, v in mod_params.items():
            if k == "E_probs":
                continue  # zastapione MLE powyzej
            params[k] = v.detach().clone()

    return params


# -- Inferencja P(E | V, Q) ----------------------------------------------------

def predict_E(
    test_df: pd.DataFrame,
    params: dict[str, torch.Tensor],
    e_prior: np.ndarray,
    use_quality: bool,
) -> pd.DataFrame:
    records = []
    for _, row in test_df.iterrows():
        log_liks = np.zeros(len(E_STATES))

        for cfg in MODALITIES.values():
            q_col = cfg["q"]
            if pd.isna(row[q_col]):
                continue

            q = int(Q_ENC[row[q_col]]) if use_quality else None

            for v_name in cfg["v_cols"]:
                v_val = row[v_name]
                if pd.isna(v_val):
                    continue
                b0  = params[f"{v_name}_beta_0"].item()
                bE_raw = params[f"{v_name}_beta_E_raw"].detach().numpy()
                bE = expand_reference_effect_np(
                    bE_raw,
                    N_E,
                    E_ENC[E_BASE],
                )
                if use_quality:
                    bQ_raw = params[f"{v_name}_beta_Q_raw"].detach().numpy()
                    bQ = expand_reference_effect_np(
                        bQ_raw,
                        N_Q,
                        Q_ENC[Q_BASE],
                    )
                    mu_arr = b0 + bE + bQ[q]
                    # sigma zalezy od jakosci Q (stale wzgledem E)
                    base_sigma = params[f"{v_name}_sigma_base"].item()
                    penalty = params[f"{v_name}_penalty"].item()
                    sig = base_sigma * (penalty if q == BAD_IDX else 1.0)
                else:
                    mu_arr = b0 + bE
                    sig = params[f"{v_name}_sigma"].item()
                for e in range(len(E_STATES)):
                    log_liks[e] += scipy_norm.logpdf(v_val, mu_arr[e], sig)

        log_post = log_liks + np.log(e_prior + 1e-300)
        log_post -= log_post.max()
        posteriors = np.exp(log_post)
        posteriors /= posteriors.sum()

        records.append({
            "E_true": row["E"],
            "E_pred": E_DEC[int(np.argmax(posteriors))],
            **{f"P({E_STATES[e]})": f"{posteriors[e]:.3f}" for e in range(len(E_STATES))},
        })

    return pd.DataFrame(records)


# -- Inferencja P(E | V_mod[, Q_mod]) osobno dla kazdej modalnosci -------------

PER_WINDOW_PROB_COLUMNS = [
    f"{mod}_{e_name.lower()}" for mod in MODALITIES for e_name in E_STATES
]
PER_WINDOW_COLUMNS = ["window_id", "type"] + PER_WINDOW_PROB_COLUMNS


def predict_E_per_modality(
    df: pd.DataFrame,
    params: dict[str, torch.Tensor],
    e_prior: np.ndarray,
    use_quality: bool,
) -> pd.DataFrame:
    """P(E | V_mod[, Q_mod]) osobno dla kazdej modalnosci (bez laczenia dowodow) --
    per emocja, per okno. Brak etykiety jakosci / brak cech dla modalnosci -> NaN."""
    records = []
    for _, row in df.iterrows():
        rec = {"entity": row["entity"], "window_id": row["window_id"], "type": row["type"]}

        for mod, cfg in MODALITIES.items():
            q_col = cfg["q"]
            probs = np.full(len(E_STATES), np.nan)

            if pd.notna(row[q_col]):
                q = int(Q_ENC[row[q_col]]) if use_quality else None
                log_liks = np.zeros(len(E_STATES))
                any_v = False
                for v_name in cfg["v_cols"]:
                    v_val = row[v_name]
                    if pd.isna(v_val):
                        continue
                    any_v = True
                    b0  = params[f"{v_name}_beta_0"].item()
                    bE_raw = params[f"{v_name}_beta_E_raw"].detach().numpy()
                    bE = expand_reference_effect_np(bE_raw, N_E, E_ENC[E_BASE])
                    if use_quality:
                        bQ_raw = params[f"{v_name}_beta_Q_raw"].detach().numpy()
                        bQ = expand_reference_effect_np(bQ_raw, N_Q, Q_ENC[Q_BASE])
                        mu_arr = b0 + bE + bQ[q]
                        base_sigma = params[f"{v_name}_sigma_base"].item()
                        penalty = params[f"{v_name}_penalty"].item()
                        sig = base_sigma * (penalty if q == BAD_IDX else 1.0)
                    else:
                        mu_arr = b0 + bE
                        sig = params[f"{v_name}_sigma"].item()
                    for e in range(len(E_STATES)):
                        log_liks[e] += scipy_norm.logpdf(v_val, mu_arr[e], sig)

                if any_v:
                    log_post = log_liks + np.log(e_prior + 1e-300)
                    log_post -= log_post.max()
                    probs = np.exp(log_post)
                    probs /= probs.sum()

            for e_name, p in zip(E_STATES, probs):
                rec[f"{mod}_{e_name.lower()}"] = p

        records.append(rec)

    return pd.DataFrame(records)


# -- Diagnostyka sigma_base / penalty -----------------------------------------

def print_sigma_q(params: dict[str, torch.Tensor]) -> None:
    """Srednie sigma_base / penalty po cechach, per modalnosc.
    Sprawdza, czy model rzeczywiscie dyskontuje niska jakosc (penalty > 1)."""
    print("\nSrednie sigma_base / penalty (po cechach) -- oczekiwane: penalty > 1:")
    for mod, cfg in MODALITIES.items():
        base_mean = float(np.mean([
            params[f"{v}_sigma_base"].item()
            for v in cfg["v_cols"]
        ]))
        penalty_mean = float(np.mean([
            params[f"{v}_penalty"].item()
            for v in cfg["v_cols"]
        ]))
        print(f"  {mod:6s}: sigma_base={base_mean:.3f}  penalty={penalty_mean:.3f}  "
              f"BAD_sigma={base_mean * penalty_mean:.3f}")


ALL_V_COLS = [v for cfg in MODALITIES.values() for v in cfg["v_cols"]]


def sigma_beta_q_values(params: dict[str, torch.Tensor], use_quality: bool) -> dict[str, float]:
    """sigma (per poziom Q, wyprowadzone z sigma_base/penalty) i beta_Q dla kazdej cechy V
    i kazdego poziomu Q (do zapisu w raporcie).
    W modelu bez quality (use_quality=False) te wezly nie istnieja -> NaN."""
    values: dict[str, float] = {}
    for v in ALL_V_COLS:
        if use_quality:
            base_sigma = params[f"{v}_sigma_base"].item()
            penalty = params[f"{v}_penalty"].item()
            sigma_q = np.array([
                base_sigma * (penalty if qi == BAD_IDX else 1.0)
                for qi in range(N_Q)
            ])
            beta_Q_raw = params[f"{v}_beta_Q_raw"].detach().numpy()
            beta_Q = expand_reference_effect_np(beta_Q_raw, N_Q, Q_ENC[Q_BASE])
        for qi, q_state in enumerate(Q_STATES):
            values[f"sigma_q_{v}_{q_state}"] = float(sigma_q[qi]) if use_quality else np.nan
            values[f"beta_q_{v}_{q_state}"] = float(beta_Q[qi]) if use_quality else np.nan
    return values


# -- Zapis wynikow do MinIO ------------------------------------------------------

RESULTS_COLUMNS = (
    ["group", "dataset", "n", "accuracy",
     "precision_macro", "recall_macro", "f1_macro",
     "precision_weighted", "recall_weighted", "f1_weighted"]
    + [f"{metric}_{e_name}" for e_name in E_STATES for metric in ("precision", "recall", "f1")]
    + [f"{split_name}_n_{q}" for split_name in ("entity", "train", "test") for q in ("GOOD", "BAD")]
    + [f"sigma_q_{v}_{q}" for v in ALL_V_COLS for q in Q_STATES]
    + [f"beta_q_{v}_{q}" for v in ALL_V_COLS for q in Q_STATES]
)


def compute_average_rows(rows: list[dict]) -> list[dict]:
    """Usrednia metryki po grupach (entity/cluster), osobno dla kazdego datasetu (Test/Test1/Test2)."""
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in RESULTS_COLUMNS if c not in ("group", "dataset")]
    avg_rows = []
    for dataset, group_df in df.groupby("dataset", sort=False):
        avg = group_df[numeric_cols].mean(numeric_only=True).to_dict()
        avg["group"] = "AVERAGE"
        avg["dataset"] = dataset
        avg_rows.append(avg)
    return avg_rows


def save_results(client: Minio, lda_experiment: str, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)[RESULTS_COLUMNS]
    object_name = f"{RESULTS_PREFIX}/{lda_experiment}_nq_mini_models.csv"

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    payload = buffer.getvalue().encode("utf-8")
    client.put_object(
        RESULTS_BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="text/csv",
    )
    print(f"\nZapisano wyniki: {RESULTS_BUCKET}/{object_name}")
    return object_name


PER_WINDOW_RESULTS_PREFIX = f"{RESULTS_PREFIX}/results_nq_mini_models"


def save_per_window_results(client: Minio, per_window_df: pd.DataFrame) -> None:
    """Zapisuje per-okno prawdopodobienstwa emocji (per modalnosc) osobno dla kazdego
    entity: gold/model/results/{eid}.csv."""
    for eid, group in per_window_df.groupby("entity", sort=False):
        out = group[PER_WINDOW_COLUMNS]
        object_name = f"{PER_WINDOW_RESULTS_PREFIX}/{eid}.csv"

        buffer = io.StringIO()
        out.to_csv(buffer, index=False)
        payload = buffer.getvalue().encode("utf-8")
        client.put_object(
            RESULTS_BUCKET,
            object_name,
            data=io.BytesIO(payload),
            length=len(payload),
            content_type="text/csv",
        )
        print(f"Zapisano wyniki per-okno: {RESULTS_BUCKET}/{object_name}")


# -- Uruchomienie pelnego pipeline'u (load -> split -> train -> eval) dla jednej grupy entities ---

def run_pipeline(
    client: Minio,
    entities: list[str],
    group_label: str,
    use_quality: bool,
    lda_experiment: str,
    steps: int,
) -> list[dict]:
    print(f"\n{'=' * 70}\n=== {group_label} (entities: {', '.join(entities)}) ===\n{'=' * 70}")

    print("Wczytywanie danych z MinIO...")
    df_all = load_data(client, entities, lda_experiment)
    print(f"  Lacznie okien: {len(df_all)}")
    print(f"  Rozklad E: {df_all['E'].value_counts().to_dict()}")
    for cfg in MODALITIES.values():
        n = df_all[cfg["q"]].notna().sum()
        print(f"  {cfg['q']}: {n} okien z etykieta jakosci  "
              f"| {df_all[cfg['q']].value_counts(dropna=True).to_dict()}")

    # Stratyfikowany podzial na poziomie trialu (E x klasa jakosci) -- caly trial
    # trafia albo do train, albo do test (okna tego samego trialu nigdy nie sa rozdzielone).
    def _qual_class(row):
        for cfg in MODALITIES.values():
            q = row[cfg["q"]]
            if pd.notna(q) and q != "GOOD":
                return "LOW"
        return "GOOD"

    df_all["_qual_class"] = df_all.apply(_qual_class, axis=1)
    # window_id ma postac "{trial_id}_{window_idx}" -- trial_id + entity identyfikuje caly trial.
    df_all["_trial_key"] = df_all["entity"] + "_" + df_all["window_id"].str.split("_").str[0]

    trials = df_all.groupby("_trial_key").agg(
        E=("E", "first"),
        _trial_qual=("_qual_class", lambda s: "LOW" if (s != "GOOD").any() else "GOOD"),
    ).reset_index()
    trial_strat_key = trials["E"] + "_" + trials["_trial_qual"]

    try:
        train_keys, test_keys = train_test_split(
            trials["_trial_key"], test_size=0.2, random_state=42, stratify=trial_strat_key
        )
    except ValueError as exc:
        print(f"[WARN] Stratyfikowany podzial po trialach niemozliwy ({exc}), uzywam zwyklego podzialu.")
        train_keys, test_keys = train_test_split(trials["_trial_key"], test_size=0.2, random_state=42)

    train_keys, test_keys = set(train_keys), set(test_keys)
    df_all["type"] = np.where(df_all["_trial_key"].isin(train_keys), "train", "test")
    train_df  = df_all[df_all["_trial_key"].isin(train_keys)]
    test_full = df_all[df_all["_trial_key"].isin(test_keys)]

    # Liczba okien GOOD/BAD (po klasie jakosci calego okna) -- entity/train/test
    def _quality_counts(df: pd.DataFrame) -> dict[str, int]:
        vc = df["_qual_class"].value_counts()
        return {"GOOD": int(vc.get("GOOD", 0)), "BAD": int(vc.get("LOW", 0))}

    quality_window_counts = {
        f"{split_name}_n_{q}": count
        for split_name, split_df in (
            ("entity", df_all), ("train", train_df), ("test", test_full),
        )
        for q, count in _quality_counts(split_df).items()
    }
    print(f"  Okna GOOD/BAD -- entity: {quality_window_counts['entity_n_GOOD']}/{quality_window_counts['entity_n_BAD']}"
          f" | train: {quality_window_counts['train_n_GOOD']}/{quality_window_counts['train_n_BAD']}"
          f" | test: {quality_window_counts['test_n_GOOD']}/{quality_window_counts['test_n_BAD']}")

    drop_cols = ["_qual_class", "_trial_key"]
    train_df = train_df.drop(columns=drop_cols).reset_index(drop=True)
    test1_df = test_full[test_full["_qual_class"] == "GOOD"].drop(columns=drop_cols).reset_index(drop=True)
    test2_df = test_full[test_full["_qual_class"] == "LOW" ].drop(columns=drop_cols).reset_index(drop=True)
    test_df  = test_full.drop(columns=drop_cols).reset_index(drop=True)

    print(f"\n  Train: {len(train_df)} | Test: {len(test_df)} "
          f"(all-GOOD: {len(test1_df)}, lower-quality: {len(test2_df)})\n")

    # Ucz kazda modalnosc osobno (patrz train()); E_probs to MLE po train_df.
    params = train(train_df, use_quality, num_steps=steps)

    e_vals = params["E_probs"].tolist()
    print(f"\n  E_probs: [{', '.join(f'{E_STATES[i]}={v:.3f}' for i, v in enumerate(e_vals))}]")
    e_prior = params["E_probs"].detach().numpy()

    # Wyniki per-okno (P(E) osobno dla kazdej modalnosci) -- zapis per entity
    per_window_df = predict_E_per_modality(df_all, params, e_prior, use_quality)
    save_per_window_results(client, per_window_df)

    # Diagnostyka sigma_q -- tylko w modelu z quality
    if use_quality:
        print_sigma_q(params)

    # Ewaluacja
    def evaluate(df: pd.DataFrame, label: str) -> dict | None:
        if len(df) == 0:
            print(f"\n=== {label} -- brak probek ===")
            return None
        print(f"\n=== {label} ({len(df)} probek) ===")
        res = predict_E(df, params, e_prior, use_quality)
        acc = accuracy_score(res["E_true"], res["E_pred"])
        print(f"Accuracy: {acc:.3f}  ({int(acc * len(res))}/{len(res)})\n")
        print(classification_report(
            res["E_true"],
            res["E_pred"],
            labels=E_STATES,
            target_names=E_STATES,
            zero_division=0,)
        )

        report = classification_report(
            res["E_true"], res["E_pred"],
            labels=E_STATES, target_names=E_STATES,
            zero_division=0, output_dict=True,
        )
        row = {
            "group": group_label,
            "dataset": label,
            "n": len(res),
            "accuracy": acc,
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }
        for e_name in E_STATES:
            row[f"precision_{e_name}"] = report[e_name]["precision"]
            row[f"recall_{e_name}"] = report[e_name]["recall"]
            row[f"f1_{e_name}"] = report[e_name]["f1-score"]
        row.update(quality_window_counts)
        row.update(sigma_beta_q_values(params, use_quality))
        return row

    results = []
    for df, label in (
        (test_df,  "Test -- caly zbior"),
        (test1_df, "Test1 -- all-GOOD"),
        (test2_df, "Test2 -- lower-quality"),
    ):
        row = evaluate(df, label)
        if row is not None:
            results.append(row)

    return results


# -- Main ------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FBN klasyfikacja emocji")
    parser.add_argument(
        "--no-quality", action="store_true",
        help="Uzyj modelu bez wezlow Q (tylko E->V). Domyslnie: z quality (E->V<-Q).",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--entity", default=None,
        help="Entity od ktorego zaczac iteracje (np. e05). Domyslnie wszystkie entities.",
    )
    parser.add_argument(
        "--lda-experiment", choices=LDA_EXPERIMENTS, default="entity",
        help=(
            "Wariant danych LDA i sposobu uczenia: "
            "'global' -- dane z gold/lda_reduction_model/global, jeden model na wszystkich entities razem; "
            "'entity' -- dane z gold/lda_reduction_model/entity, osobny model per entity; "
            "'cluster' -- dane z gold/lda_reduction_model/cluster, osobny model per klaster entities. "
            "Domyslnie 'entity'."
        ),
    )
    args = parser.parse_args()
    use_quality = not args.no_quality

    print(f"Tryb: {'z quality (E->V<-Q)' if use_quality else 'bez quality (E->V)'}")
    print(f"LDA experiment: {args.lda_experiment}")

    client = get_minio_client()
    entities = resolve_entities(args.entity) if args.entity else list(ENTITIES)

    all_results: list[dict] = []

    if args.lda_experiment == "global":
        all_results += run_pipeline(client, entities, "global", use_quality, args.lda_experiment, args.steps)

    elif args.lda_experiment == "entity":
        for eid in entities:
            all_results += run_pipeline(client, [eid], eid, use_quality, args.lda_experiment, args.steps)
        all_results += compute_average_rows(all_results)

    elif args.lda_experiment == "cluster":
        clusters = load_cluster_assignments(client)
        if clusters is None or clusters.empty:
            raise RuntimeError(
                "Brak przypisania do klastrow "
                f"({CLUSTERING_BUCKET}/{CLUSTERING_PREFIX}/{CLUSTER_SOURCE_MODALITY}) -- "
                "nie mozna uruchomic trybu 'cluster'."
            )
        entities_set = set(entities)
        for cluster_id, cluster_group in clusters.groupby("cluster"):
            cluster_entities = [e for e in cluster_group["entity"].tolist() if e in entities_set]
            if not cluster_entities:
                continue
            all_results += run_pipeline(
                client, cluster_entities, f"cluster {cluster_id}",
                use_quality, args.lda_experiment, args.steps,
            )
        all_results += compute_average_rows(all_results)

    if all_results:
        save_results(client, args.lda_experiment, all_results)
    else:
        print("[WARN] Brak wynikow do zapisania.")
