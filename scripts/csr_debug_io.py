import json
from typing import Any, Dict, Iterable, List


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON payload in {path}: expected object or list.")


def load_dataset_records(data_path: str, limit: int) -> List[Dict[str, Any]]:
    if "@" in data_path:
        dataset_name, split = data_path.split("@", 1)
    else:
        dataset_name, split = data_path, "train"

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("The `datasets` package is required for --data. Use --input-json instead.") from exc

    dataset = load_dataset(dataset_name, split=split)
    records = []
    for idx, row in enumerate(dataset):
        if idx >= limit:
            break
        records.append(dict(row))
    return records


def iter_input_records(input_json: str = None, data: str = None, limit: int = 4) -> Iterable[Dict[str, Any]]:
    if input_json:
        yield from load_json_records(input_json)[:limit]
    elif data:
        yield from load_dataset_records(data, limit)
    else:
        return


def get_field(record: Dict[str, Any], key: str, default: str = "") -> str:
    value = record.get(key, default)
    return "" if value is None else str(value)
