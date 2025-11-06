"""
Exp1 Data Utilities

This package provides:
- Question type inference (schema.py)
- Answer normalization and preprocessing (preprocess.py)
- Question type analysis (analyze_by_type.py)
"""

from .schema import (
    QuestionType,
    COLOR_VOCAB,
    infer_question_type,
    build_candidates
)

from .preprocess import (
    normalize_answer,
    enrich_jsonl
)

__all__ = [
    'QuestionType',
    'COLOR_VOCAB',
    'infer_question_type',
    'build_candidates',
    'normalize_answer',
    'enrich_jsonl'
]

