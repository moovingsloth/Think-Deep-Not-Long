"""
Answer extraction utilities for Think@n.

Extracts numerical answers from model-generated text for evaluation.
Supports LaTeX \boxed{} notation and fallback patterns.
"""

import re
from typing import Optional


def _extract_answer_from_span(text: str) -> Optional[str]:
    """Extract a numeric answer from a text span (boxed, then patterns, then last number)."""
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        answer = answer.replace('$', '').replace('\\', '').strip()
        return answer

    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s+([+-]?\d+\.?\d*)',
        r'(?:final\s+)?answer:\s*([+-]?\d+\.?\d*)',
        r'equals?\s+([+-]?\d+\.?\d*)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    numbers = re.findall(r'[+-]?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None


def extract_answer(text: str) -> Optional[str]:
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
    if not text:
        return None

    # Prefer the final answer after the thinking trace.
    if "</think>" in text:
        after = text.rsplit("</think>", 1)[-1].strip()
        if after:
            found = _extract_answer_from_span(after)
            if found is not None:
                return found

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


def answers_match(answer1: Optional[str], answer2: Optional[str]) -> bool:
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
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    
    if norm1 is None or norm2 is None:
        return False
    
    return norm1 == norm2
