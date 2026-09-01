import torch
import os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from src.model.model_utils import get_device, get_model_dtype

class DTREngine:
    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        cache_dir: str | None = None,
        g: float = 0.5, # Settling threshold (used in Jensen-Shannon Divergence)
        rho: float = 0.85, # Depth fraction (used in late regime start calculation)
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
    ):
        self.device = get_device()
        print(f"Device: {self.device}")
        # Default cache_dir derived from model_id if not provided
        if cache_dir is None:
            model_name = model_id.split("/")[-1]  # e.g. DeepSeek-R1-Distill-Llama-8B
            cache_dir = os.path.abspath(f"models/{model_name}")
        
        self.model_id = model_id
        self.cache_dir = os.path.abspath(cache_dir)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=self.cache_dir,
            local_files_only=True   # Only use local files, no network calls
        )
        self.dtype = get_model_dtype(self.device)
        print(f"Load dtype: {self.dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            # bf16 on CUDA (native for Qwen3-4B); fp16 on MPS for memory.
            # JSD math is still upcast to float32 in generate_step.
            dtype=self.dtype,
            cache_dir=self.cache_dir,
            local_files_only=True,   # Only use local files, no network calls
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        # While RMSNorm and attention layers behave the same in train/eval mode, 
        # some models may have dropout or other layers that behave differently. 
        # Best practice is to call .eval() for inference.
        self.model.eval()
        
        # Print model and tokenizer configs
        print(f"Model config: {self.model.model}")
        self.print_tokenizer_info()
        
        self.g = g
        self.rho = rho
        self.L = self.model.config.num_hidden_layers # Total layers (L) [cite: 103]
        self.eos_token_ids = self._collect_eos_ids()
        self._init_sampling_params(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    def print_tokenizer_info(self):
        """Print key tokenizer configs."""
        # info = {
        #     "name_or_path": self.tokenizer.name_or_path,
        #     "class": type(self.tokenizer).__name__,
        #     "vocab_size": self.tokenizer.vocab_size,
        #     "model_max_length": self.tokenizer.model_max_length,
        #     "is_fast": self.tokenizer.is_fast,
        #     "pad_token": self.tokenizer.pad_token,
        #     "pad_token_id": self.tokenizer.pad_token_id,
        #     "eos_token": self.tokenizer.eos_token,
        #     "eos_token_id": self.tokenizer.eos_token_id,
        #     "bos_token": self.tokenizer.bos_token,
        #     "bos_token_id": getattr(self.tokenizer, "bos_token_id", None),
        #     "unk_token": self.tokenizer.unk_token,
        #     "unk_token_id": getattr(self.tokenizer, "unk_token_id", None),
        # }
        # print("\nTokenizer Info:")
        # for k, v in info.items():
        #     print(f"  {k}: {v}")
        
        # For a config-like dump (attributes that are typically JSON-serializable)
        attrs = ["vocab_size", "model_max_length", "name_or_path", "pad_token", "eos_token", "bos_token", "unk_token"]
        for attr in attrs:
            if hasattr(self.tokenizer, attr):
                print(f"  {attr}: {getattr(self.tokenizer, attr)}")
        has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        print(f"  chat_template: {'set' if has_chat_template else None}")

    def _init_sampling_params(
        self,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
    ):
        """Read decoding hyperparameters from generation_config, with anti-loop defaults.

        Qwen3-Thinking's generation_config uses sampling (T=0.6, top_p=0.95, top_k=20).
        Greedy argmax on these models starts coherently then collapses into n-gram loops
        such as "the more the total is the more the total is ...".
        """
        gc = getattr(self.model, "generation_config", None)
        diff = {}
        if gc is not None and hasattr(gc, "to_diff_dict"):
            diff = gc.to_diff_dict()
        elif gc is not None:
            for key in ("temperature", "top_k", "top_p", "do_sample", "repetition_penalty", "no_repeat_ngram_size"):
                value = getattr(gc, key, None)
                if value is not None:
                    diff[key] = value

        self.temperature = float(diff.get("temperature", 0.6) or 0.6)
        self.top_k = int(diff.get("top_k", 20) or 0)
        self.top_p = float(diff.get("top_p", 0.95) or 1.0)
        self.do_sample_default = bool(diff.get("do_sample", True))
        self.repetition_penalty = float(
            repetition_penalty if repetition_penalty is not None else diff.get("repetition_penalty", 1.15)
        )
        self.no_repeat_ngram_size = int(
            no_repeat_ngram_size if no_repeat_ngram_size is not None else diff.get("no_repeat_ngram_size", 6)
        )
        print(
            f"  decoding: do_sample={self.do_sample_default} T={self.temperature} "
            f"top_p={self.top_p} top_k={self.top_k} "
            f"repetition_penalty={self.repetition_penalty} "
            f"no_repeat_ngram_size={self.no_repeat_ngram_size}"
        )

    @staticmethod
    def _cfg_value(generation_config, name, default):
        if generation_config is None:
            return default
        value = getattr(generation_config, name, default)
        return default if value is None else value

    def _collect_eos_ids(self) -> set[int]:
        ids: set[int] = set()
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
        gc = getattr(self.model, "generation_config", None)
        gc_eos = getattr(gc, "eos_token_id", None) if gc is not None else None
        if gc_eos is not None:
            ids.update(gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos])
        return {int(i) for i in ids}

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a user prompt, applying the chat template when the tokenizer has one.

        Qwen3-Thinking models require ChatML + `<think>` (via add_generation_prompt).
        Raw `tokenizer.encode(prompt)` is out of distribution and collapses to
        high-frequency loops such as repeating 'the'.
        """
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            try:
                encoded = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    enable_thinking=True,
                )
            except Exception as e:
                print(f"Warning: chat template failed ({e}); falling back to raw encode")
                encoded = None
            if encoded is not None:
                if hasattr(encoded, "input_ids"):
                    encoded = encoded.input_ids
                if not isinstance(encoded, torch.Tensor):
                    encoded = torch.tensor(encoded, dtype=torch.long)
                if encoded.dim() == 1:
                    encoded = encoded.unsqueeze(0)
                return encoded.to(self.device)
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def _select_next_token(
        self,
        logits: torch.Tensor,
        do_sample: bool = False,
        input_ids: torch.Tensor | None = None,
        prompt_len: int | None = None,
    ) -> torch.Tensor:
        """Pick the next token from last-position logits of shape [batch, vocab]."""
        if input_ids is not None:
            processors = LogitsProcessorList()
            penalty = getattr(self, "repetition_penalty", 1.0)
            if penalty and penalty != 1.0:
                processors.append(
                    RepetitionPenaltyLogitsProcessor(
                        penalty=float(penalty),
                        prompt_ignore_length=prompt_len,
                    )
                )
            ngram = getattr(self, "no_repeat_ngram_size", 0)
            if ngram and int(ngram) > 0:
                processors.append(NoRepeatNGramLogitsProcessor(int(ngram)))
            if do_sample:
                temperature = getattr(self, "temperature", 1.0)
                if temperature and float(temperature) > 0 and float(temperature) != 1.0:
                    processors.append(TemperatureLogitsWarper(float(temperature)))
                top_k = getattr(self, "top_k", 0)
                if top_k is not None and int(top_k) > 0:
                    processors.append(TopKLogitsWarper(int(top_k)))
                top_p = getattr(self, "top_p", 1.0)
                if top_p is not None and 0.0 < float(top_p) < 1.0:
                    processors.append(TopPLogitsWarper(float(top_p)))
            if processors:
                logits = processors(input_ids, logits)

        if not do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def calculate_jsd(self, p, q):
        """Eq 2: Jensen-Shannon Divergence [cite: 145]"""
        g = 0.5 # Default settling threshold - counts tokens that truly required sustained internal revision
        # if you set g too low (e.g., 0.25), the metric becomes "overly permissive". 
        # It starts counting simple, low-effort tokens as "thinking," 
        # which causes the correlation with accuracy to flatten out or disappear
        m = g * (p + q)

        # p_log = p.clamp(min=1e-10).log()
        # q_log = q.clamp(min=1e-10).log()
        m_log = m.clamp(min=1e-10).log()

        # Using kl_div for numerical stability on MPS
        return 0.5 * (F.kl_div(m_log, p, reduction='batchmean', log_target=False) + 
                      F.kl_div(m_log, q, reduction='batchmean', log_target=False))

    """
        h_final: normalized final hidden state.
        h_l: normalized intermediate hidden state.
        p_L, p_l: vocabulary distributions (Logit Lens).
            p_L is the final layer distribution, 
            p_l is the intermediate layer distribution.
        c_t: settling depth (first layer matching p_L).
        late_regime_start: start of late regime.
        is_deep: whether the token is a deep-thinking token.
        next_token: the next token to generate.
    """
    def generate_step(self, token_ids, do_sample: bool = False, prompt_len: int | None = None):
        with torch.no_grad():
            # Fixed: Pass output_hidden_states=True in forward call, not in from_pretrained
            outputs = self.model(token_ids, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states  # Tuple of (embedding, layer1, ..., layerL)
        # Use hidden_states[-1] with same norm+lm_head pipeline as intermediate layers (Logit Lens consistency)
        h_final = self.model.model.norm(hidden_states[-1][:, -1, :])
        # Upcast to float32 for the numerically sensitive JSD math
        lens_logits = self.model.lm_head(h_final).to(torch.float32)
        p_L = F.softmax(lens_logits, dim=-1)  # Final layer distribution [cite: 108]
        
        # Algorithm 1: Find settling depth c_t [cite: 113, 155]
        # hidden_states[0] = embedding, hidden_states[1..L] = after each transformer layer
        c_t = self.L
        # Iterating through the layers to find the settling depth
        for l in range(1, self.L + 1):
            # Project intermediate hidden state to vocabulary space (Logit Lens) [cite: 105, 106]
            h_l = self.model.model.norm(hidden_states[l][:, -1, :])
            p_l = F.softmax(self.model.lm_head(h_l).to(torch.float32), dim=-1)
            
            if self.calculate_jsd(p_L, p_l).item() <= self.g:
                c_t = l
                break
        
        # Check if token is "deep-thinking" (settles in the late regime) [cite: 157, 161]
        late_regime_start = int(self.rho * self.L) 
        # print(f"late_regime_start = {late_regime_start}")
        is_deep = c_t >= late_regime_start # Late regime definition 
        
        # Decode from the model's actual next-token logits; DTR still uses the logit lens
        if getattr(outputs, "logits", None) is not None:
            dec_logits = outputs.logits[:, -1, :].to(torch.float32)
        else:
            dec_logits = lens_logits
        next_token = self._select_next_token(
            dec_logits,
            do_sample=do_sample,
            input_ids=token_ids,
            prompt_len=prompt_len,
        )
        return next_token, is_deep

    def generate_with_dtr(self, prompt, max_tokens=50, do_sample: bool | None = None):
        """
        Generate tokens with DTR calculation (Algorithm 1).
        
        Args:
            prompt: Input text prompt (user content; chat template is applied)
            max_tokens: Maximum tokens to generate
            do_sample: If None, use the model's generation_config (Qwen thinking
                models sample). Greedy decoding often loops after a coherent start.
            
        Yields:
            Tuple of (token_str, is_deep, dtr) for each generated token
        """
        if do_sample is None:
            do_sample = getattr(self, "do_sample_default", True)
        token_ids = self._encode_prompt(prompt)
        prompt_len = token_ids.shape[1]
        deep_tokens = 0
        
        for t in range(1, max_tokens + 1):
            next_token, is_deep = self.generate_step(
                token_ids, do_sample=do_sample, prompt_len=prompt_len
            )
            if is_deep: deep_tokens += 1
            
            token_ids = torch.cat([token_ids, next_token], dim=1)
            dtr = deep_tokens / t # Eq 6: Deep-Thinking Ratio [cite: 164]
            
            yield self.tokenizer.decode(next_token[0]), is_deep, dtr
            if next_token.item() in self.eos_token_ids:
                break
    
    def estimate_dtr_from_prefix(
        self,
        prompt: str,
        prefix_length: int = 50,
        do_sample: bool = False,
    ) -> tuple[str, float, int, list[int]]:
        """
        Generate a prefix and estimate DTR from it (for Think@n early stopping).
        
        Args:
            prompt: Input text prompt
            prefix_length: Number of tokens to generate for DTR estimation
            do_sample: Whether to sample tokens (needed for diverse Think@n prefixes)
            
        Returns:
            Tuple of (generated_text, prefix_dtr, tokens_generated, generated_ids)
        """
        token_ids = self._encode_prompt(prompt)
        prompt_len = token_ids.shape[1]
        deep_tokens = 0
        generated_tokens = []
        
        for t in range(1, prefix_length + 1):
            next_token, is_deep = self.generate_step(
                token_ids, do_sample=do_sample, prompt_len=prompt_len
            )
            if is_deep:
                deep_tokens += 1
            
            generated_tokens.append(next_token.item())
            token_ids = torch.cat([token_ids, next_token], dim=1)
            
            if next_token.item() in self.eos_token_ids:
                break
        
        tokens_generated = len(generated_tokens)
        prefix_dtr = deep_tokens / tokens_generated if tokens_generated > 0 else 0.0
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text, prefix_dtr, tokens_generated, generated_tokens
    
    def generate_n_samples(
        self, 
        prompt: str, 
        n: int = 48, 
        prefix_length: int = 50, 
        max_tokens: int = 500,
        early_stop: bool = True,
        eta: float = 0.5,
        do_sample: bool = True,
    ) -> list[dict]:
        """
        Generate n samples with early stopping based on prefix DTR (for Think@n).
        
        Args:
            prompt: Input text prompt
            n: Number of samples to generate
            prefix_length: Number of tokens for DTR estimation
            max_tokens: Maximum tokens per sample (full generation)
            early_stop: Whether to stop low-DTR samples early
            eta: Fraction of top samples to continue (e.g., 0.5 = top 50%)
            do_sample: Sample prefixes/continuations (required for diverse n samples)
            
        Returns:
            List of sample dicts with keys: 'text', 'dtr', 'tokens', 'full_generation'
        """
        samples = []
        
        # Phase 1: Generate prefixes for all n samples
        print(f"Generating {n} prefixes ({prefix_length} tokens each)...")
        for i in range(n):
            prefix_text, prefix_dtr, tokens_gen, prefix_ids = self.estimate_dtr_from_prefix(
                prompt, prefix_length, do_sample=do_sample
            )
            samples.append({
                'text': prefix_text,
                'dtr': prefix_dtr,
                'tokens': tokens_gen,
                'generated_ids': prefix_ids,
                'full_generation': False,
                'sample_id': i
            })
        
        if not early_stop:
            # Continue all samples to completion
            print(f"Continuing all {n} samples to completion...")
            for sample in samples:
                full_text, final_dtr, total_tokens = self._continue_generation(
                    prompt, sample['text'], max_tokens,
                    prefix_ids=sample.get('generated_ids'),
                    do_sample=do_sample,
                )
                sample['text'] = full_text
                sample['dtr'] = final_dtr
                sample['tokens'] = total_tokens
                sample['full_generation'] = True
        else:
            # Phase 2: Rank by prefix DTR and continue only top eta%
            samples_sorted = sorted(samples, key=lambda x: x['dtr'], reverse=True)
            top_k = max(1, int(eta * n))
            
            print(f"Early stopping: continuing top {top_k}/{n} samples (η={eta})...")
            for i, sample in enumerate(samples_sorted):
                if i < top_k:
                    # Continue top samples
                    full_text, final_dtr, total_tokens = self._continue_generation(
                        prompt, sample['text'], max_tokens,
                        prefix_ids=sample.get('generated_ids'),
                        do_sample=do_sample,
                    )
                    sample['text'] = full_text
                    sample['dtr'] = final_dtr
                    sample['tokens'] = total_tokens
                    sample['full_generation'] = True
                else:
                    # Bottom samples stopped at prefix
                    sample['full_generation'] = False
        
        return samples
    
    def _continue_generation(
        self,
        prompt: str,
        prefix_text: str,
        max_tokens: int,
        prefix_ids: list[int] | None = None,
        do_sample: bool = True,
    ) -> tuple[str, float, int]:
        """
        Continue generation from a prefix to completion.
        
        Args:
            prompt: Original input prompt
            prefix_text: Already generated prefix text
            max_tokens: Maximum total tokens
            prefix_ids: Token ids of the prefix (avoids decode/encode drift)
            do_sample: Whether to sample continuation tokens
            
        Returns:
            Tuple of (full_text, final_dtr, total_tokens)
        """
        token_ids = self._encode_prompt(prompt)
        prompt_len = token_ids.shape[1]
        if prefix_ids:
            prefix_tensor = torch.tensor([prefix_ids], dtype=token_ids.dtype, device=self.device)
            token_ids = torch.cat([token_ids, prefix_tensor], dim=1)
        elif prefix_text:
            prefix_tensor = self.tokenizer.encode(
                prefix_text, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            token_ids = torch.cat([token_ids, prefix_tensor], dim=1)
        
        # Count deep tokens in what we generate from here
        deep_tokens = 0
        tokens_generated = 0
        generated_tokens = []
        
        for t in range(max_tokens):
            next_token, is_deep = self.generate_step(
                token_ids, do_sample=do_sample, prompt_len=prompt_len
            )
            if is_deep:
                deep_tokens += 1
            
            generated_tokens.append(next_token.item())
            tokens_generated += 1
            token_ids = torch.cat([token_ids, next_token], dim=1)
            
            if next_token.item() in self.eos_token_ids:
                break
        
        # Calculate final DTR over the continued portion
        continuation_dtr = deep_tokens / tokens_generated if tokens_generated > 0 else 0.0
        
        # Decode only the new tokens
        continuation_text = self.tokenizer.decode(generated_tokens)
        full_text = prefix_text + continuation_text
        
        # Total tokens = prefix + continuation
        total_tokens = token_ids.shape[1] - prompt_len
        
        return full_text, continuation_dtr, total_tokens

    def main():
        """Entry point for running DTREngine from CLI or as a module."""
        import argparse

        parser = argparse.ArgumentParser(description="DTR Engine - Deep Thinking Ratio")
        parser.add_argument(
            "--model-id",
            default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            help="HuggingFace model ID",
        )
        parser.add_argument(
            "--cache-dir",
            default=None,
            help="Local cache directory for model (default: models/<model-name>)",
        )
        parser.add_argument(
            "--prompt",
            default="Calculate 12 * 12: ",
            help="Prompt to generate",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=25,
            help="Max tokens to generate",
        )
        parser.add_argument("--g", type=float, default=0.5, help="Settling threshold")
        parser.add_argument("--rho", type=float, default=0.85, help="Depth fraction")
        parser.add_argument(
            "--greedy",
            action="store_true",
            help="Force greedy decoding (thinking models often loop without sampling)",
        )
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args()
        if args.seed is not None:
            torch.manual_seed(args.seed)

        engine = DTREngine(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            g=args.g,
            rho=args.rho,
        )
        print(f"Using model: {engine.model_id}, cache: {engine.cache_dir}")
        print(f"Generating for: {args.prompt}")
        for word, is_deep, dtr in engine.generate_with_dtr(
            args.prompt,
            max_tokens=args.max_tokens,
            do_sample=False if args.greedy else None,
        ):
            marker = "<<deep-think-token>>" if is_deep else "  "
            print(f"{marker} '{word:10}' | DTR: {dtr:.2f}")


    if __name__ == "__main__":
        main()