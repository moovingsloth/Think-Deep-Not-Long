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
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.config.models import MODELS
from src.dtr_engine import DTREngine
from src.think_at_n import ThinkAtN

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
    parser.add_argument("--seed", type=int, default=0)
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
    if args.model not in MODELS:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODELS.keys())}")
    if not 1 <= args.num_problems <= len(TEST_PROBLEMS):
        raise ValueError(f"--num-problems must be 1..{len(TEST_PROBLEMS)}")

    import torch

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("Think@n benchmark")
    print("=" * 80)
    print("Configuration:")
    print(f"  model: {args.model} ({MODELS[args.model]['model_id']})")
    print("  thinking: enabled (chat template + enable_thinking=True)")
    print(f"  n={args.n}  eta={args.eta}  prefix={args.prefix_length}  max_tokens={args.max_tokens}")
    print(f"  early_stop={args.early_stop}  seed={args.seed}")
    print(f"  problems: {args.num_problems}/{len(TEST_PROBLEMS)}")
    print()

    engine = DTREngine(**MODELS[args.model])
    think = ThinkAtN(
        dtr_engine=engine,
        n=args.n,
        eta=args.eta,
        prefix_length=args.prefix_length,
        max_tokens=args.max_tokens,
    )

    problems = TEST_PROBLEMS[: args.num_problems]
    results = []
    for i, case in enumerate(problems, start=1):
        print(f"\n{'#' * 80}")
        print(f"Problem {i}/{len(problems)}")
        print(f"{'#' * 80}")
        results.append(
            think.solve(
                problem=case["problem"],
                ground_truth=case["answer"],
                early_stop=args.early_stop,
            )
        )

    methods = ["think_at_n", "cons_at_n", "short_at_n", "long_at_n"]
    summary = {method: {"correct": 0, "total_cost": 0} for method in methods}
    for result in results:
        for method in methods:
            if result[method]["correct"]:
                summary[method]["correct"] += 1
            summary[method]["total_cost"] += result[method]["cost"]

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
        },
        "summary": summary,
        "results": [_jsonable(r) for r in results],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()