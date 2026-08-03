import argparse
import csv
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from minio import Minio
from minio.error import S3Error

from minio_common import (
    ENTITIES,
    MAX_WINDOWS_PER_TRIAL,
    SOURCE_BUCKET,
    get_minio_client,
    resolve_entities,
)

AUXILIARY_PREFIX = "01_entity_resolution/eav/auxiliary"
META_DATA_OBJECT = f"{AUXILIARY_PREFIX}/metadata/meta_data.csv"
QUESTIONNAIRE_OBJECT = f"{AUXILIARY_PREFIX}/annotations/questionnaire.xlsx"

TARGET_PREFIX = "05_annotations_model"

REPORT_FIELDNAMES = [
    "window_id",
    "emotion_class",
    "valence_participant",
    "valence_experimenter",
    "arousal_participant",
    "arousal_experimenter",
    "avg_valence",
    "avg_arousal",
    "valence_arousal_emotion",
]

_ONE_HOT_COL_TO_CLASS: Dict[int, str] = {
    5: "Neutral", 6: "Neutral",
    7: "Sadness", 8: "Sadness",
    9: "Anger", 10: "Anger",
    11: "Happiness", 12: "Happiness",
    13: "Calm", 14: "Calm",
}

_QUESTIONNAIRE_CLASS_MAP: Dict[str, str] = {
    "Happy": "Happiness",
    "Calm": "Calm",
    "Angry": "Anger",
    "Sad": "Sadness",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buduje etykiety emocji (valence/arousal) dla datasetu EAV "
        "na podstawie questionnaire.xlsx."
    )
    parser.add_argument(
        "--entity",
        default=ENTITIES[0],
        help=f"Entity od ktorego zaczac iteracje (np. e05). Zakres: {ENTITIES[0]}..{ENTITIES[-1]}. "
        f"Domyslnie {ENTITIES[0]}.",
    )
    return parser.parse_args()


def get_object_bytes(client: Minio, bucket: str, object_name: str) -> bytes:
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def load_conversations(data: bytes) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(data), header=None, skiprows=2)

    conv_emotion: Dict[int, str] = {}
    trials_by_conv: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        idx = pd.to_numeric(row.iloc[3], errors="coerce")
        if pd.isna(idx):
            continue
        trial_id = int(idx)
        conv_idx = (trial_id - 1) // 10
        trials_by_conv.setdefault(conv_idx, []).append(trial_id)

        if conv_idx in conv_emotion:
            continue
        for col_pos, cls in _ONE_HOT_COL_TO_CLASS.items():
            val = pd.to_numeric(row.iloc[col_pos], errors="coerce")
            if pd.notna(val) and int(val) == 1 and cls != "Neutral":
                conv_emotion[conv_idx] = cls
                break

    class_count: Dict[str, int] = {}
    conversations: List[Dict[str, Any]] = []
    for conv_idx in sorted(trials_by_conv):
        cls = conv_emotion.get(conv_idx)
        if cls is None:
            print(f"[WARN] Konwersacja {conv_idx}: brak dominujacej klasy emocji, pomijam.")
            continue
        class_count[cls] = class_count.get(cls, 0) + 1
        conversations.append({
            "conv_idx": conv_idx,
            "emotion_class": cls,
            "occurrence": class_count[cls],
            "trial_ids": sorted(trials_by_conv[conv_idx]),
        })
    return conversations


def parse_questionnaire_sheet(df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, float]]:
    result: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row_idx in range(len(df)):
        cell = df.iloc[row_idx, 0]
        if not isinstance(cell, str):
            continue
        m = re.search(r"emotion\s+class\s*:?\s*(\w+)", cell, re.IGNORECASE)
        if not m:
            continue
        raw_class = m.group(1).strip().capitalize()
        emotion_class = _QUESTIONNAIRE_CLASS_MAP.get(raw_class, raw_class)
        for k in range(1, 6):
            data_idx = row_idx + 1 + k  # +1 pomija wiersz z naglowkami kolumn
            if data_idx >= len(df):
                break
            r = df.iloc[data_idx]
            try:
                val_part = float(r.iloc[1])
                aro_part = float(r.iloc[2])
            except (ValueError, TypeError, IndexError) as exc:
                print(f"[WARN] Blad parsowania oceny uczestnika [{emotion_class} k={k} row={data_idx}]: {exc}")
                continue

            result[(emotion_class, k)] = {
                "valence_participant": val_part,
                "arousal_participant": aro_part,
                "valence_experimenter": _parse_experimenter_score(r.iloc[5]),
                "arousal_experimenter": _parse_experimenter_score(r.iloc[6]),
            }
    return result


def _parse_experimenter_score(value: Any) -> Any:
    if pd.isna(value):
        return "-"
    try:
        return float(value)
    except (ValueError, TypeError):
        return "-"


def load_questionnaire(data: bytes) -> Dict[str, Dict[Tuple[str, int], Dict[str, float]]]:
    xl = pd.ExcelFile(io.BytesIO(data))
    result: Dict[str, Dict[Tuple[str, int], Dict[str, float]]] = {}
    for sheet_name in xl.sheet_names:
        if not re.match(r"subject_\d+", sheet_name, re.IGNORECASE):
            continue
        df = xl.parse(sheet_name, header=None)
        result[sheet_name] = parse_questionnaire_sheet(df)
    return result


def valence_arousal_to_emotion(valence: float, arousal: float) -> Optional[str]:
    if pd.isna(valence) or pd.isna(arousal):
        return None
    if valence >= 0 and arousal >= 0:
        return "Happiness"
    if valence <= 0 and arousal >= 0:
        return "Anger"
    if valence >= 0 and arousal <= 0:
        return "Calm"
    if valence <= 0 and arousal <= 0:
        return "Sadness"
    return None


def _average_with_participant(participant_value: float, experimenter_value: Any) -> float:
    if isinstance(experimenter_value, str):
        return participant_value
    return (participant_value + experimenter_value) / 2


def build_annotation_report(
    eid: str,
    conversations: List[Dict[str, Any]],
    questionnaire: Dict[str, Dict[Tuple[str, int], Dict[str, float]]],
) -> List[Dict]:
    m = re.match(r"e0*(\d+)$", eid, re.IGNORECASE)
    sheet_name = f"subject_{int(m.group(1))}"
    subject_scores = questionnaire.get(sheet_name)
    if subject_scores is None:
        print(f"[WARN] {eid}: brak arkusza '{sheet_name}' w kwestionariuszu, pomijam.")
        return []

    rows: List[Dict] = []
    for conv in conversations:
        entry = subject_scores.get((conv["emotion_class"], conv["occurrence"]))
        if entry is None:
            print(f"[WARN] {eid}: brak oceny dla ({conv['emotion_class']}, k={conv['occurrence']})")
            continue

        avg_valence = _average_with_participant(entry["valence_participant"], entry["valence_experimenter"])
        avg_arousal = _average_with_participant(entry["arousal_participant"], entry["arousal_experimenter"])
        va_emotion = valence_arousal_to_emotion(avg_valence, avg_arousal)

        for trial_id in conv["trial_ids"]:
            for window_idx in range(MAX_WINDOWS_PER_TRIAL):
                rows.append({
                    "window_id": f"{trial_id:03d}_{window_idx}",
                    "emotion_class": conv["emotion_class"],
                    "valence_participant": entry["valence_participant"],
                    "valence_experimenter": entry["valence_experimenter"],
                    "arousal_participant": entry["arousal_participant"],
                    "arousal_experimenter": entry["arousal_experimenter"],
                    "avg_valence": avg_valence,
                    "avg_arousal": avg_arousal,
                    "valence_arousal_emotion": va_emotion,
                })
    return rows


def save_annotation_report(client: Minio, eid: str, rows: List[Dict]) -> str:
    object_name = f"{TARGET_PREFIX}/{eid}_annotations.csv"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REPORT_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

    payload = buffer.getvalue().encode("utf-8")
    client.put_object(
        SOURCE_BUCKET,
        object_name,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="text/csv",
    )
    return object_name


def run_annotation_labels(client: Minio, entities: List[str]) -> None:
    try:
        meta_bytes = get_object_bytes(client, SOURCE_BUCKET, META_DATA_OBJECT)
    except S3Error as exc:
        print(f"[ERROR] Nie mozna wczytac {META_DATA_OBJECT}: {exc}")
        return
    conversations = load_conversations(meta_bytes)
    print(f"meta_data.csv: {len(conversations)} konwersacji")

    try:
        quest_bytes = get_object_bytes(client, SOURCE_BUCKET, QUESTIONNAIRE_OBJECT)
    except S3Error as exc:
        print(f"[ERROR] Nie mozna wczytac {QUESTIONNAIRE_OBJECT}: {exc}")
        return
    questionnaire = load_questionnaire(quest_bytes)
    print(f"questionnaire.xlsx: {len(questionnaire)} arkuszy uczestnikow")

    for eid in entities:
        rows = build_annotation_report(eid, conversations, questionnaire)
        if not rows:
            print(f"[WARN] Brak etykiet dla {eid}, pomijam zapis raportu.")
            continue
        saved_path = save_annotation_report(client, eid, rows)
        print(f"{eid}: zapisano {SOURCE_BUCKET}/{saved_path} ({len(rows)} okien)")


def main() -> None:
    args = parse_args()
    entities = resolve_entities(args.entity)

    print(f"Entities: {entities[0]}..{entities[-1]} ({len(entities)} szt.)")

    client = get_minio_client()
    run_annotation_labels(client, entities)


if __name__ == "__main__":
    main()
