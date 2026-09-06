import argparse
import os
import sys

from dotenv import load_dotenv

# Add the Think-Deep-Not-Long root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.config.models import MODELS

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def _resolve_cache_dir(cache_dir: str) -> str:
    if os.path.isabs(cache_dir):
        return cache_dir
    return os.path.join(_project_root, cache_dir)


def _selected_names(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(MODELS.keys())
    names: list[str] = []
    names.extend(args.models)
    if args.model_flags:
        names.extend(args.model_flags)
    if not names:
        names = ["qwen4b", "qwen35"]
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model snapshots into the local cache"
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        choices=list(MODELS.keys()),
        help="Short names from src.config.models (default: qwen4b qwen35)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_flags",
        metavar="MODEL",
        choices=list(MODELS.keys()),
        help="Short name (repeatable; kept for compatibility)",
    )
    parser.add_argument("--all", action="store_true", help="Download every registered model")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print model ids and cache paths without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist in the cache",
    )
    args = parser.parse_args()

    for name in _selected_names(args):
        config = MODELS[name]
        model_id = config["model_id"]
        cache_dir = _resolve_cache_dir(config["cache_dir"])
        print(f"model={name} repo={model_id} cache_dir={cache_dir}")
        if args.dry_run:
            continue
        os.makedirs(cache_dir, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=HF_TOKEN,
            force_download=args.force,
        )
        print(f"stored {name} in {cache_dir}")


if __name__ == "__main__":
    main()
