import argparse
import os
import struct

import numpy as np
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

SOURCE_BUCKET = "silver"
ENTITY_RESOLUTION_PREFIX = "01_entity_resolution/eav/files"

ENTITIES = [f"e{n:02d}" for n in range(1, 43)]  # e01 .. e42
MODALITIES = ("audio", "video", "eeg")

WINDOW_SECONDS = 1.0
TRIAL_SECONDS = 20.0
MAX_WINDOWS_PER_TRIAL = int(TRIAL_SECONDS / WINDOW_SECONDS)

EEG_LABEL_SUFFIX = "_label.mat"


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def build_modality_prefix(eid: str, modality: str) -> str:
    return f"{ENTITY_RESOLUTION_PREFIX}/entity={eid}/modality={modality}/"


def resolve_modalities(modality_args: list[str]) -> tuple[str, ...]:
    requested = [m.strip().lower() for m in modality_args]
    if "all" in requested:
        return MODALITIES

    invalid = [m for m in requested if m not in MODALITIES]
    if invalid:
        raise ValueError(
            f"Nieznana modalnosc: {invalid}. Dozwolone: {list(MODALITIES)} lub 'all'."
        )
    return tuple(dict.fromkeys(requested))


def resolve_entities(start_entity: str) -> list[str]:
    start_entity = start_entity.strip().lower()
    if start_entity not in ENTITIES:
        raise ValueError(
            f"Nieznane entity: {start_entity}. Dozwolone: {ENTITIES[0]}..{ENTITIES[-1]}."
        )
    start_index = ENTITIES.index(start_entity)
    return ENTITIES[start_index:]


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--modality",
        nargs="+",
        default=["all"],
        help="Modalnosc/i do przetworzenia: 'all', 'audio', 'video', 'eeg' "
        "lub dowolna kombinacja podana jako kilka wartosci (np. --modality audio eeg). "
        "Domyslnie 'all'.",
    )
    parser.add_argument(
        "--entity",
        default=ENTITIES[0],
        help=f"Entity od ktorego zaczac iteracje (np. e05). Zakres: {ENTITIES[0]}..{ENTITIES[-1]}. "
        f"Domyslnie {ENTITIES[0]}.",
    )
    return parser


def list_modality_files(client: Minio, eid: str, modality: str) -> list[str]:
    prefix = build_modality_prefix(eid, modality)
    objects = client.list_objects(SOURCE_BUCKET, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]


def trial_number_from_object_name(object_name: str) -> str:
    basename = object_name.rsplit("/", 1)[-1]
    return basename.split("_", 1)[0]


def parse_wav(data: bytes) -> tuple[np.ndarray, int]:
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Nieprawidlowy naglowek WAV")

    pos = 12
    fmt = None
    audio_bytes = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + chunk_size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            audio_bytes = body
        pos += 8 + chunk_size + (chunk_size % 2)

    if fmt is None or audio_bytes is None:
        raise ValueError("Brak chunku fmt/data w pliku WAV")

    audio_format, channels, sample_rate, _, _, bits_per_sample = fmt
    if (audio_format, channels, bits_per_sample) != (3, 1, 32):
        raise ValueError(
            f"Nieobslugiwany format WAV: format={audio_format}, channels={channels}, bits={bits_per_sample}"
        )

    return np.frombuffer(audio_bytes, dtype="<f4"), sample_rate


def find_eeg_segment_array(mat: dict, object_name: str) -> np.ndarray:
    candidates = [v for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3]
    if len(candidates) != 1:
        raise ValueError(
            f"Nie znaleziono jednoznacznej tablicy 3D z danymi EEG w {object_name} "
            f"(znaleziono {len(candidates)} kandydatow)"
        )
    return candidates[0]
