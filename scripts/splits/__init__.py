"""
Splits package for stratified dataset creation.
"""

from .pipeline import stratify_sample, check_proportions
from .file_io import save_splits, load_indices, load_tsv
from .json_prompts import create_json, load_files, get_nlp_model, get_tf_model
from .norm_utils import create_norm_files, regenerate_splits_from_indices
from .norm_check import (
    check_norm_file,
    fix_norm_file,
    validate_and_fix_norm_files,
    get_norm_statistics,
    NormValidationError
)

__all__ = [
    # Core splitting
    'stratify_sample',
    'check_proportions',
    # File I/O
    'save_splits',
    'load_indices',
    'load_tsv',
    # JSON generation
    'create_json',
    'load_files',
    'get_nlp_model',
    'get_tf_model',
    # NORM utilities
    'create_norm_files',
    'regenerate_splits_from_indices',
    # NORM validation
    'check_norm_file',
    'fix_norm_file',
    'validate_and_fix_norm_files',
    'get_norm_statistics',
    'NormValidationError',
]