"""
Splits package for stratified dataset creation.
"""

from .split_core import stratify_sample, check_proportions
from .file_utils import save_splits, load_indices, load_tsv
from .prompt_json import create_json, load_files, get_nlp_model, get_tf_model
from .norm_utils import create_norm_files, regenerate_splits_from_indices

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