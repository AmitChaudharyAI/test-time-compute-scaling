"""Deterministic MATH scoring and vote canonicalization."""

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from sympy import Basic

from run_phase1c import normalize_answer


PARSER_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]


def parse_answer(answer):
    if answer is None:
        return []
    normalized = normalize_answer(answer)
    if not normalized:
        return []
    return parse(f"${normalized}$", extraction_config=PARSER_CONFIG)


def math_equal(prediction, reference):
    """Compare only an extracted prediction and a canonical reference."""
    pred_parsed = parse_answer(prediction)
    ref_parsed = parse_answer(reference)
    if pred_parsed and ref_parsed:
        return bool(verify(ref_parsed, pred_parsed))
    # Deterministic fallback for literal answers Math-Verify cannot parse.
    return normalize_answer(prediction) == normalize_answer(reference)


def vote_key(answer):
    """Canonical key from the same parsing/normalization stack used to score."""
    parsed = parse_answer(answer)
    if parsed:
        primary = parsed[0]
        if isinstance(primary, Basic):
            return "sympy:" + str(primary)
        return "parsed:" + str(primary)
    normalized = normalize_answer(answer)
    return None if normalized is None else "literal:" + normalized
