"""
Answer extraction utilities for Think@n.

Extracts numerical answers from model-generated text for evaluation.
Supports LaTeX \boxed{} notation and fallback patterns.
"""

import json
import re
from functools import lru_cache
from typing import Optional


ANSWER_TYPES = ("math", "choice", "plain")


def _last_boxed(text: str) -> Optional[str]:
    """Return the contents of the last balanced ``\\boxed{...}`` expression."""
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    content_start = start + len(r"\boxed{")
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def _extract_answer_from_span(text: str) -> Optional[str]:
    """Extract a numeric answer from a text span (boxed, then patterns, then last number)."""
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.replace('$', '').strip()

    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s+([+-]?\d+\.?\d*)',
        r'(?:final\s+)?answer:\s*([+-]?\d+\.?\d*)',
        r'equals?\s+([+-]?\d+\.?\d*)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Multiple-choice benchmarks such as GPQA request a single option letter.
    choice = re.search(r'(?:final\s+)?answer(?:\s+is|\s*:)?\s*\(?([A-D])\)?', text, re.IGNORECASE)
    if choice:
        return choice.group(1).upper()
    stripped = text.strip().rstrip(".")
    if re.fullmatch(r"\(?[A-Da-d]\)?", stripped):
        return stripped.strip("()").upper()

    numbers = re.findall(r'[+-]?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None


def _extract_choice(text: str) -> Optional[str]:
    """Extract a final multiple-choice letter without matching prose letters."""
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            value = json.loads(match.group(0)).get("answer")
        except (AttributeError, json.JSONDecodeError):
            continue
        if isinstance(value, str) and value.strip().upper() in {"A", "B", "C", "D"}:
            return value.strip().upper()
    choice = re.search(
        r"(?:final\s+)?answer(?:\s+is|\s*:)?\s*[\"']?\(?([A-D])\)?[\"']?",
        text,
        re.IGNORECASE,
    )
    if choice:
        return choice.group(1).upper()
    stripped = text.strip().rstrip(".").strip("()\"'")
    return stripped.upper() if re.fullmatch(r"[A-Da-d]", stripped) else None


def extract_answer(
    text: str,
    *,
    answer_type: str = "plain",
    require_think_end: bool = False,
) -> Optional[str]:
    """
    Extract answer from generated text.
    
    Tries multiple strategies in order:
    1. Post-</think> span (Qwen thinking models)
    2. LaTeX \boxed{...} notation (AIME style)
    3. "answer is X" patterns
    4. Last numerical value in text
    
    Args:
        text: Generated text containing the answer
        
    Returns:
        Extracted answer as string, or None if no answer found
        
    Examples:
        >>> extract_answer("Therefore \\boxed{144}")
        '144'
        >>> extract_answer("The answer is 12")
        '12'
        >>> extract_answer("We get x = 8")
        '8'
        >>> extract_answer("<think>12 then 13</think>\\\\n\\\\boxed{144}")
        '144'
    """
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"Unsupported answer type: {answer_type}")
    if not text:
        return None

    # Prefer the final answer after the thinking trace.
    if "</think>" in text:
        after = text.rsplit("</think>", 1)[-1].strip()
        if after:
            found = _extract_choice(after) if answer_type == "choice" else _extract_answer_from_span(after)
            if found is not None:
                return found

    if require_think_end:
        return None

    if answer_type == "choice":
        return _extract_choice(text)

    return _extract_answer_from_span(text)


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """
    Normalize answer for comparison.
    
    - Converts to lowercase
    - Strips whitespace
    - Removes trailing .0 from decimals (144.0 -> 144)
    - Handles None gracefully
    
    Args:
        answer: Answer string to normalize
        
    Returns:
        Normalized answer or None
        
    Examples:
        >>> normalize_answer("144.0")
        '144'
        >>> normalize_answer("  12  ")
        '12'
    """
    if answer is None:
        return None
    
    answer = str(answer).strip().lower()
    
    # Remove trailing .0 from decimals
    if '.' in answer:
        try:
            num = float(answer)
            if num == int(num):
                answer = str(int(num))
        except ValueError:
            pass
    
    return answer


@lru_cache(maxsize=4096)
def _parse_math(answer: str):
    """Parse a math answer with the benchmark's pinned verifier."""
    from math_verify import parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    wrapped = answer if "$" in answer or r"\boxed" in answer else f"${answer}$"
    return parse(
        wrapped,
        extraction_config=(
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ),
        fallback_mode="no_fallback",
    )


def answers_match(
    answer1: Optional[str],
    answer2: Optional[str],
    *,
    answer_type: str = "plain",
) -> bool:
    """
    Check if two answers are equivalent after normalization.
    
    Args:
        answer1: First answer
        answer2: Second answer
        
    Returns:
        True if answers match after normalization
        
    Examples:
        >>> answers_match("144", "144.0")
        True
        >>> answers_match("12", "13")
        False
    """
    if answer_type == "math":
        if answer1 is None or answer2 is None:
            return False
        try:
            from math_verify import verify

            parsed1 = _parse_math(str(answer1))
            parsed2 = _parse_math(str(answer2))
            return bool(parsed1 and parsed2 and verify(parsed2, parsed1))
        except (ImportError, RuntimeError, TypeError, ValueError):
            return False

    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    
    if norm1 is None or norm2 is None:
        return False
    
    if answer_type == "choice":
        return norm1.upper() == norm2.upper()
    return norm1 == norm2
