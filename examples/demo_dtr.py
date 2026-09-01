import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dtr_engine import DTREngine
from src.config.models import MODELS

def main(
    model_name: str = "qwen.6b",
    prompt: str = "Calculate 12 * 12: ",
    max_tokens: int = 25,
    do_sample: bool | None = None,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
    seed: int | None = None,
):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}")

    if seed is not None:
        import torch
        torch.manual_seed(seed)
    
    config = MODELS[model_name]
    engine = DTREngine(
        **config,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    
    print(f"Using model: {model_name}")
    print(f"Generating for: {prompt}")
    pieces = []
    for word, is_deep, dtr in engine.generate_with_dtr(
        prompt, max_tokens=max_tokens, do_sample=do_sample
    ):
        pieces.append(word)
        marker = "🧠" if is_deep else "  "
        shown = word.replace("\n", "\\n").replace("\t", "\\t")
        print(f"{marker} {shown!r:52} | DTR: {dtr:.2f}")
    print("\n--- decoded ---")
    print("".join(pieces))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        default="qwen.6b", 
        choices=list(MODELS.keys()), 
        help="Model to use (must be downloaded first)")
    prompt = "Calculate 12 * 12: "
    # prompt = """ Circle 𝜔1 with radius 6 centered at point 𝐴 is internally tangent at point 𝐵 to circle 𝜔2 with radius 15. Points 𝐶 and 𝐷lie on 𝜔2 such that 𝐵𝐶 is a diameter of 𝜔2 and 𝐵𝐶 ⊥ 𝐴𝐷. The rectangle 𝐸𝐹𝐺 𝐻 is inscribed in 𝜔1 such that 𝐸𝐹 ⊥ 𝐵𝐶, 𝐶 is closer to 𝐺 𝐻 than to 𝐸𝐹, and 𝐷 is closer to 𝐹𝐺 than to 𝐸𝐻, as shown. Triangles △𝐷𝐺𝐹 and △𝐶 𝐻𝐺 have equal areas. The area of rectangle 𝐸𝐹𝐺 𝐻 is 𝑚 𝑛 , where 𝑚 and 𝑛 are relatively prime positive integers. Find 𝑚 + 𝑛."""

    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--max-tokens", type=int, default=25)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Force greedy decoding (thinking models often loop without sampling)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty >1.0 downweights already generated tokens (default 1.0 = off)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="Ban repeating n-grams (default 0 = off; >0 can break arithmetic)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    main(
        model_name=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        do_sample=False if args.greedy else None,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed,
    )