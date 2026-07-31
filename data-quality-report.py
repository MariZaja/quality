import argparse
import csv
import io

from minio import Minio
from minio.error import S3Error

from minio_common import (
    ENTITIES,
    SOURCE_BUCKET,
    get_minio_client,
    resolve_entities,
)

QUALITY_FLAGS_PREFIX = "04_quality_flags_model"
TARGET_BUCKET = "gold"
TARGET_PREFIX = "data_quality_model"

MODALITIES = ("audio", "eeg", "video")
REPORT_FIELDNAMES = ["window_id", "audio_quality", "eeg_quality", "video_quality"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Buduje zbiorczy raport data quality (audio/eeg/video) na podstawie "
        "flag z 04_quality_flags_model."
    )
    parser.add_argument(
        "--entity",
        default=ENTITIES[0],
        help=f"Entity od ktorego zaczac iteracje (np. e05). Zakres: {ENTITIES[0]}..{ENTITIES[-1]}. "
        f"Domyslnie {ENTITIES[0]}.",
    )
    return parser.parse_args()


def window_sort_key(window_id: str) -> tuple[int, int]:
    trial_str, idx_str = window_id.split("_", 1)
    return int(trial_str), int(idx_str)


def load_quality_flags(client: Minio, eid: str, modality: str) -> dict[str, str]:
    object_name = f"{QUALITY_FLAGS_PREFIX}/{eid}/{modality}/{eid}_{modality}_quality_flags.csv"
    try:
        response = client.get_object(SOURCE_BUCKET, object_name)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        print(f"[WARN] Brak {object_name}: {exc}")
        return {}

    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    return {row["window_id"]: row["quality_flag"] for row in reader}


def build_data_quality_report(client: Minio, eid: str) -> list[dict]:
    flags_by_modality = {
        modality: load_quality_flags(client, eid, modality) for modality in MODALITIES
    }

    window_ids = set()
    for flags in flags_by_modality.values():
        window_ids.update(flags)

    rows = []
    for window_id in sorted(window_ids, key=window_sort_key):
        rows.append(
            {
                "window_id": window_id,
                "audio_quality": flags_by_modality["audio"].get(window_id, ""),
                "eeg_quality": flags_by_modality["eeg"].get(window_id, ""),
                "video_quality": flags_by_modality["video"].get(window_id, ""),
            }
        )
    return rows


def save_data_quality_report(client: Minio, eid: str, rows: list[dict]) -> str:
    object_name = f"{TARGET_PREFIX}/{eid}_data_quality.csv"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REPORT_FIELDNAMES)
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


def run_data_quality_report(client: Minio, entities: list[str]) -> None:
    for eid in entities:
        rows = build_data_quality_report(client, eid)
        if not rows:
            print(f"[WARN] Brak flag jakosci dla {eid}, pomijam zapis raportu.")
            continue
        saved_path = save_data_quality_report(client, eid, rows)
        print(f"{eid}: zapisano {TARGET_BUCKET}/{saved_path} ({len(rows)} okien)")


def main() -> None:
    args = parse_args()
    entities = resolve_entities(args.entity)

    print(f"Entities: {entities[0]}..{entities[-1]} ({len(entities)} szt.)")

    client = get_minio_client()
    run_data_quality_report(client, entities)


if __name__ == "__main__":
    main()
