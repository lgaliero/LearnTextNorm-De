"""
Splits package for stratified dataset creation.
"""

from splits.pipeline import stratify_sample, check_proportions
from splits.file_io import save_splits, load_indices, load_tsv
from splits.json_prompts import create_json, load_files, get_nlp_model, get_tf_model
from splits.norm_utils import create_norm_files, regenerate_splits_from_indices

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
]