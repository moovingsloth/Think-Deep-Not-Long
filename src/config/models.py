import os


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_DEFAULT_HF_HUB_CACHE = os.path.join(_REPO_ROOT, ".cache", "huggingface", "hub")

HF_HUB_CACHE = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "HF_HUB_CACHE",
            os.environ.get("HUGGINGFACE_HUB_CACHE", _DEFAULT_HF_HUB_CACHE),
        )
    )
)


# Model registry: short name -> {model_id, cache_dir}
MODELS = {
    "qwen.6b": {
        "model_id": "Qwen/Qwen3-0.6B",
        "cache_dir": HF_HUB_CACHE,
    },
    "deepseek": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "cache_dir": HF_HUB_CACHE,
    },
    "qwen4b": {
        "model_id": "Qwen/Qwen3-4B-Thinking-2507",
        "cache_dir": HF_HUB_CACHE,
    },
    "nanbeige": {
        "model_id": "Nanbeige/Nanbeige4-3B-Thinking-2511",
        "cache_dir": HF_HUB_CACHE,
    },
    "qwen35": {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "cache_dir": HF_HUB_CACHE,
    },
    "olmo3_7B": {
        "model_id": "allenai/Olmo-3-7B-Think",
        "cache_dir": HF_HUB_CACHE,
    },
    # "gemma3_4b": {
    #     "model_id": "google/gemma-3-4b-it",
    #     "cache_dir": "models/google/gemma3",
    # }
}