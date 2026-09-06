"""Load benchmark cases from the shared, read-only dataset mount."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_DATASET_ROOT = Path.home() / "mnt" / "seraph-datasets"
BENCHMARKS = ("aime24", "aime25", "hmmt25", "gpqa_diamond")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _aime24(root: Path) -> tuple[list[dict], Path]:
    path = root / "AIME_2024" / "test-00000-of-00001.parquet"
    rows = pq.read_table(path, columns=["id", "problem", "solution"]).to_pylist()
    cases = []
    for row in rows:
        matches = re.findall(r"\\boxed\{([^{}]+)\}", row["solution"])
        if not matches:
            raise ValueError(f"AIME24 row {row['id']} has no boxed answer")
        cases.append({
            "id": str(row["id"]), "problem": row["problem"],
            "answer": matches[-1], "answer_type": "math",
        })
    return cases, path


def _math_jsonl(path: Path) -> tuple[list[dict], Path]:
    return [
        {
            "id": str(row["id"]),
            "problem": row.get("problem", row.get("question")),
            "answer": str(row["answer"]),
            "answer_type": "math",
        }
        for row in _read_jsonl(path)
    ], path


def _gpqa(root: Path, seed: int) -> tuple[list[dict], Path]:
    path = root / "data" / "gpqa" / "gpqa_diamond.csv"
    cases = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            choices = [(row["Correct Answer"], True)] + [
                (row[f"Incorrect Answer {i}"], False) for i in range(1, 4)
            ]
            random.Random(f"{seed}:{row['Record ID']}").shuffle(choices)
            answer = next("ABCD"[i] for i, (_, correct) in enumerate(choices) if correct)
            options = "\n".join(f"{'ABCD'[i]}. {text}" for i, (text, _) in enumerate(choices))
            cases.append({
                "id": row["Record ID"] or str(index),
                "problem": f"{row['Question']}\n\n{options}\n\nAnswer with only A, B, C, or D.",
                "answer": answer,
                "answer_type": "choice",
            })
    return cases, path


def load_benchmarks(
    names: list[str], limit_per_dataset: int,
    dataset_root: Path = DEFAULT_DATASET_ROOT, seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Load a deterministic prefix from each requested real benchmark."""
    if limit_per_dataset < 0:
        raise ValueError("limit_per_dataset must be non-negative (0 means all)")
    unknown = set(names) - set(BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
    loaders = {
        "aime24": lambda: _aime24(dataset_root),
        "aime25": lambda: _math_jsonl(dataset_root / "AIME_2025" / "test.jsonl"),
        "hmmt25": lambda: _math_jsonl(dataset_root / "HMMT_2025" / "hmmt_25.jsonl"),
        "gpqa_diamond": lambda: _gpqa(dataset_root, seed),
    }
    problems, provenance = [], []
    for name in names:
        rows, source = loaders[name]()
        selected = rows if limit_per_dataset == 0 else rows[:limit_per_dataset]
        provenance.append({
            "benchmark": name, "source": str(source),
            "available": len(rows), "loaded": len(selected),
        })
        problems.extend({**row, "benchmark": name} for row in selected)
    return problems, provenance
