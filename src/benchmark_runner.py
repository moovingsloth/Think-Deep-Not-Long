"""Think@n compute-efficiency benchmark (Experiment 2).

Compares Think@n against Cons@n / Short@n / Long@n on simple math.

Qwen3-4B-Thinking always opens a <think> block via the chat template
(enable_thinking is also passed for hybrid Qwen3/Qwen3.5 models).

Usage:
    uv run python src/benchmark_runner.py --model qwen4b --n 12 --num-problems 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.config.models import MODELS
from src.benchmark_datasets import BENCHMARKS, DEFAULT_DATASET_ROOT, load_benchmarks
from src.dtr_engine import DTREngine
from src.compute_profiles import PROFILES, resolve_compute_profile
from src.distributed_benchmark import load_json, selected_problem_keys, stable_problem_seed, summarize
from src.think_at_n import ThinkAtN
from src.answer_extraction import answers_match, extract_answer

TEST_PROBLEMS = [
    {"problem": "Calculate 12 * 12: ", "answer": "144"},
    {"problem": "What is the square root of 144? ", "answer": "12"},
    {"problem": "If x + 5 = 13, what is x? ", "answer": "8"},
    {"problem": "Calculate 15 + 27: ", "answer": "42"},
    {"problem": "What is 100 divided by 4? ", "answer": "25"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Think@n vs baselines (qwen4b thinking by default)"
    )
    parser.add_argument(
        "--model",
        default="qwen4b",
        choices=list(MODELS.keys()),
        help="Model short name (must be downloaded first)",
    )
    parser.add_argument("--n", type=int, default=12, help="Samples per problem")
    parser.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="Fraction of top-DTR samples to keep (0.5 = top 50%%)",
    )
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=50,
        help="Prefix tokens used to estimate DTR",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens for each continued sample (thinking traces need room)",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=3,
        help=f"How many of the {len(TEST_PROBLEMS)} built-in problems to run",
    )
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS)
    parser.add_argument(
        "--problems-per-dataset", type=int, default=0,
        help="Cases per benchmark (0 loads the complete dataset)",
    )
    parser.add_argument(
        "--dataset-root", default=None,
        help="Dataset root (default comes from the selected compute profile)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-profile", choices=["auto", *PROFILES], default="auto")
    parser.add_argument("--prefix-batch-size", type=int)
    parser.add_argument("--continuation-batch-size", type=int)
    parser.add_argument(
        "--score-continuation-dtr",
        action="store_true",
        help="Run expensive logit-lens scoring on continuation tokens (not needed for Think@n selection)",
    )
    parser.add_argument("--shard-manifest")
    parser.add_argument("--worker-id")
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue only top-η samples (default: on). --no-early-stop generates all n fully.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON path for results (default: results/benchmark_<model>_<stamp>.json)",
    )
    return parser.parse_args()


def _jsonable(result: dict) -> dict:
    out = {k: v for k, v in result.items() if k != "samples"}
    out["samples"] = [
        {k: v for k, v in sample.items() if k != "generated_ids"}
        for sample in result.get("samples", [])
    ]
    return out


def main() -> None:
    args = parse_args()
    profile = resolve_compute_profile(args.compute_profile)
    prefix_batch_size = args.prefix_batch_size or profile.prefix_batch_size
    continuation_batch_size = args.continuation_batch_size or profile.continuation_batch_size
    if args.prefix_batch_size is not None and args.prefix_batch_size < 1:
        raise ValueError("--prefix-batch-size must be positive")
    if args.continuation_batch_size is not None and args.continuation_batch_size < 1:
        raise ValueError("--continuation-batch-size must be positive")
    if bool(args.shard_manifest) != bool(args.worker_id):
        raise ValueError("--shard-manifest and --worker-id must be supplied together")
    if args.model not in MODELS:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODELS.keys())}")
    if not args.benchmarks and not 1 <= args.num_problems <= len(TEST_PROBLEMS):
        raise ValueError(f"--num-problems must be 1..{len(TEST_PROBLEMS)}")

    import torch

    print("=" * 80)
    print("Think@n benchmark")
    print("=" * 80)
    print("Configuration:")
    print(f"  model: {args.model} ({MODELS[args.model]['model_id']})")
    print("  thinking: enabled (chat template + enable_thinking=True)")
    print(f"  n={args.n}  eta={args.eta}  prefix={args.prefix_length}  max_tokens={args.max_tokens}")
    print(f"  early_stop={args.early_stop}  seed={args.seed}")
    if args.benchmarks:
        dataset_root = args.dataset_root or profile.dataset_root
        problems, provenance = load_benchmarks(
            args.benchmarks, args.problems_per_dataset, Path(dataset_root), args.seed
        )
        if args.shard_manifest:
            wanted = selected_problem_keys(load_json(args.shard_manifest), args.worker_id)
            problems = [p for p in problems if (p["benchmark"], str(p["id"])) in wanted]
            if len(problems) != len(wanted):
                raise ValueError("manifest references problems absent from the loaded datasets")
        print(f"  benchmarks: {', '.join(args.benchmarks)}")
        limit_label = "all" if args.problems_per_dataset == 0 else str(args.problems_per_dataset)
        print(f"  problems: {len(problems)} ({limit_label} per dataset)")
        for item in provenance:
            print(f"  loaded {item['loaded']}/{item['available']} {item['benchmark']} from {item['source']}")
    else:
        problems = TEST_PROBLEMS[: args.num_problems]
        provenance = [{"benchmark": "builtin", "loaded": len(problems), "available": len(TEST_PROBLEMS)}]
        print(f"  problems: {args.num_problems}/{len(TEST_PROBLEMS)}")
    print()

    engine = DTREngine(**MODELS[args.model])
    think = ThinkAtN(
        dtr_engine=engine,
        n=args.n,
        eta=args.eta,
        prefix_length=args.prefix_length,
        max_tokens=args.max_tokens,
        prefix_batch_size=prefix_batch_size,
        continuation_batch_size=continuation_batch_size,
        score_continuation_dtr=args.score_continuation_dtr,
    )

    results = []
    benchmark_started = time.monotonic()
    for i, case in enumerate(problems, start=1):
        print(f"\n{'#' * 80}")
        print(f"Problem {i}/{len(problems)}")
        print(f"{'#' * 80}")
        problem_seed = stable_problem_seed(args.seed, case.get("benchmark", "builtin"), case.get("id", str(i)))
        torch.manual_seed(problem_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(problem_seed)
        result = think.solve(
                problem=case["problem"],
                ground_truth=case["answer"],
                early_stop=args.early_stop,
            )
        result["benchmark"] = case.get("benchmark", "builtin")
        result["problem_id"] = case.get("id", str(i))
        result["seed"] = problem_seed
        results.append(result)

    methods = ["think_at_n", "cons_at_n", "short_at_n", "long_at_n"]
    summary, benchmark_summary = summarize(results)
    selected_rows, rejected_rows = [], []
    for result in results:
        ranked = sorted(result["samples"], key=lambda sample: sample["dtr"], reverse=True)
        selected_count = max(1, int(args.eta * len(ranked)))
        for selected, sample in enumerate(ranked):
            row = {
                "dtr": sample["dtr"],
                "correct": answers_match(extract_answer(sample["text"]), result["ground_truth"]),
            }
            (selected_rows if selected < selected_count else rejected_rows).append(row)
    def _mean(rows, key):
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0
    dtr_effect = {
        "selected_sample_count": len(selected_rows),
        "rejected_sample_count": len(rejected_rows),
        "selected_mean_dtr": _mean(selected_rows, "dtr"),
        "rejected_mean_dtr": _mean(rejected_rows, "dtr"),
        "selected_sample_accuracy": _mean(selected_rows, "correct"),
        "rejected_sample_accuracy": _mean(rejected_rows, "correct"),
        "think_vs_cons_accuracy_delta": (
            summary["think_at_n"]["correct"] - summary["cons_at_n"]["correct"]
        ) / len(results),
        "cost_saving_vs_cons": 1 - (
            summary["think_at_n"]["total_cost"] / summary["cons_at_n"]["total_cost"]
        ) if summary["cons_at_n"]["total_cost"] else 0.0,
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<25} {'Accuracy':<15} {'Total Cost':<15} {'Avg Cost'}")
    print("-" * 80)
    for method in methods:
        accuracy = summary[method]["correct"] / len(results)
        total_cost = summary[method]["total_cost"]
        avg_cost = total_cost / len(results)
        name = results[0][method]["method"]
        print(f"{name:<25} {accuracy:>6.1%}{'':<8} {total_cost:>10,}{'':<5} {avg_cost:>8,.0f}")

    think_total = summary["think_at_n"]["total_cost"]
    cons_total = summary["cons_at_n"]["total_cost"]
    print("-" * 80)
    if cons_total > 0:
        savings = (1 - think_total / cons_total) * 100
        print(f"Think@n cost savings vs Cons@n: {savings:.1f}%")
    print(
        f"Think@n accuracy: {summary['think_at_n']['correct'] / len(results):.1%}  "
        f"Cons@n accuracy: {summary['cons_at_n']['correct'] / len(results):.1%}"
    )
    print("=" * 80)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or os.path.join(
        _ROOT, "results", f"benchmark_{args.model}_n{args.n}_{stamp}.json"
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    payload = {
        "config": {
            "model": args.model,
            "model_id": MODELS[args.model]["model_id"],
            "thinking": True,
            "n": args.n,
            "eta": args.eta,
            "prefix_length": args.prefix_length,
            "max_tokens": args.max_tokens,
            "early_stop": args.early_stop,
            "seed": args.seed,
            "num_problems": args.num_problems,
            "benchmarks": args.benchmarks or ["builtin"],
            "problems_per_dataset": args.problems_per_dataset if args.benchmarks else None,
            "dataset_root": str(dataset_root) if args.benchmarks else None,
            "compute_profile": profile.name,
            "prefix_batch_size": prefix_batch_size,
            "continuation_batch_size": continuation_batch_size,
            "worker_id": args.worker_id,
            "score_continuation_dtr": args.score_continuation_dtr,
        },
        "dataset_provenance": provenance,
        "summary": summary,
        "benchmark_summary": benchmark_summary,
        "dtr_effect": dtr_effect,
        "results": [_jsonable(r) for r in results],
        "performance": {
            **engine.runtime_metrics,
            "prefix_tokens_per_second": engine.runtime_metrics["prefix_tokens"] / engine.runtime_metrics["prefix_seconds"] if engine.runtime_metrics["prefix_seconds"] else 0.0,
            "continuation_tokens_per_second": engine.runtime_metrics["continuation_tokens"] / engine.runtime_metrics["continuation_seconds"] if engine.runtime_metrics["continuation_seconds"] else 0.0,
            "elapsed_seconds": time.monotonic() - benchmark_started,
            "problem_count": len(results),
        },
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
