"""Deterministic shard planning, validation, and benchmark result merging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

METHODS = ("think_at_n", "cons_at_n", "short_at_n", "long_at_n")


def stable_problem_seed(base_seed: int, benchmark: str, problem_id: str) -> int:
    material = f"{base_seed}\0{benchmark}\0{problem_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def plan_shards(problems: list[dict], calibrations: list[dict]) -> dict:
    """Greedily assign work to the worker with the earliest predicted finish."""
    if not calibrations:
        raise ValueError("at least one calibration is required")
    workers = []
    for item in calibrations:
        elapsed = float(item["elapsed_seconds"])
        count = int(item.get("problem_count", 1))
        if elapsed <= 0 or count <= 0:
            raise ValueError("calibration elapsed_seconds and problem_count must be positive")
        workers.append({
            "worker_id": item["worker_id"],
            "compute_profile": item["compute_profile"],
            "seconds_per_problem": elapsed / count,
            "problem_keys": [],
            "predicted_seconds": 0.0,
        })
    for problem in problems:
        worker = min(workers, key=lambda w: (w["predicted_seconds"], w["worker_id"]))
        worker["problem_keys"].append([problem["benchmark"], str(problem["id"])])
        worker["predicted_seconds"] += worker["seconds_per_problem"]
    return {
        "version": 1,
        "problem_count": len(problems),
        "problem_order": [[p["benchmark"], str(p["id"])] for p in problems],
        "workers": workers,
    }


def selected_problem_keys(manifest: dict, worker_id: str) -> set[tuple[str, str]]:
    matches = [w for w in manifest["workers"] if w["worker_id"] == worker_id]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain worker {worker_id!r} exactly once")
    return {(str(a), str(b)) for a, b in matches[0]["problem_keys"]}


def summarize(results: list[dict]) -> tuple[dict, dict]:
    summary = {m: {"correct": 0, "total_cost": 0} for m in METHODS}
    for result in results:
        for method in METHODS:
            summary[method]["correct"] += int(bool(result[method]["correct"]))
            summary[method]["total_cost"] += result[method]["cost"]
    benchmark_summary = {}
    for benchmark in dict.fromkeys(r["benchmark"] for r in results):
        rows = [r for r in results if r["benchmark"] == benchmark]
        benchmark_summary[benchmark] = {
            m: {
                "correct": sum(bool(r[m]["correct"]) for r in rows),
                "total": len(rows),
                "accuracy": sum(bool(r[m]["correct"]) for r in rows) / len(rows),
            }
            for m in METHODS
        }
    return summary, benchmark_summary


def merge_shards(manifest: dict, payloads: list[dict]) -> dict:
    assignments = [tuple(key) for w in manifest["workers"] for key in w["problem_keys"]]
    expected = [tuple(key) for key in manifest.get("problem_order", assignments)]
    if len(assignments) != len(set(assignments)):
        raise ValueError("manifest contains duplicate problem assignments")
    if set(expected) != set(assignments) or len(expected) != manifest["problem_count"]:
        raise ValueError("manifest coverage does not match problem_count")
    if not payloads:
        raise ValueError("no shard payloads supplied")
    reference = payloads[0]["config"]
    comparable = (
        "model", "model_id", "n", "eta", "prefix_length", "max_tokens",
        "early_stop", "seed", "score_continuation_dtr",
    )
    results = []
    for payload in payloads:
        if any(payload["config"].get(k) != reference.get(k) for k in comparable):
            raise ValueError("shard decoding/model configuration mismatch")
        results.extend(payload["results"])
    by_key = {}
    for result in results:
        key = (result["benchmark"], str(result["problem_id"]))
        if key in by_key:
            raise ValueError(f"duplicate shard result: {key}")
        expected_seed = stable_problem_seed(reference["seed"], *key)
        if result.get("seed") != expected_seed:
            raise ValueError(f"problem seed mismatch: {key}")
        by_key[key] = result
    missing, extra = set(expected) - set(by_key), set(by_key) - set(expected)
    if missing or extra:
        raise ValueError(f"shard coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    ordered = [by_key[key] for key in expected]
    summary, benchmark_summary = summarize(ordered)
    return {**payloads[0], "config": {**reference, "compute_profile": "distributed"},
            "summary": summary, "benchmark_summary": benchmark_summary, "results": ordered,
            "shards": [p["config"].get("worker_id") for p in payloads]}


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)
