"""
File I/O utilities for dataset splits.
"""

import os
import pandas as pd
from typing import List


def save_splits(
    df_split: pd.DataFrame,
    split_name: str,
    output_dir: str,
    indices: List[int],
    df_full: pd.DataFrame,
    paths_config: dict = None
) -> None:
    """
    Save source, target, and indices files for a dataset split.
    
    Args:
        df_split: Split DataFrame
        split_name: Name of the split ('test', 'train', 'dev')
        output_dir: Output directory
        indices: List of indices in this split
        df_full: Full DataFrame (for metadata)
        paths_config: Optional dict with path mappings (TEST_SRC, TRAIN_SRC, etc.)
    """
    # Use paths from config if available, otherwise construct
    if paths_config and split_name in ['test', 'train', 'dev']:
        src_file = paths_config.get(f'{split_name.upper()}_SRC')
        tgt_file = paths_config.get(f'{split_name.upper()}_TGT')
    else:
        src_file = None
        tgt_file = None
    
    # Fallback to output_dir if paths not in config
    if not src_file:
        src_file = os.path.join(output_dir, f"{split_name}.src")
    if not tgt_file:
        tgt_file = os.path.join(output_dir, f"{split_name}.tgt")
    
    indices_file = os.path.join(output_dir, f"{split_name}_indices.tsv")

    # Create directories if they don't exist
    for path in [src_file, tgt_file, indices_file]:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    # Write source file
    with open(src_file, 'w', encoding='utf-8') as f:
        for src in df_split['src']:
            f.write(f"{src}\n")
    
    # Write target file
    with open(tgt_file, 'w', encoding='utf-8') as f:
        for tgt in df_split['tgt']:
            f.write(f"{tgt}\n")
    
    # Create indices TSV with full metadata
    df_indices = df_full.loc[indices].copy()
    df_indices.insert(0, 'DF_INDEX', indices)
    df_indices.to_csv(indices_file, sep='\t', index=False, encoding='utf-8')
    
    print(f"✓ Saved {split_name} source: {src_file}")
    print(f"✓ Saved {split_name} target: {tgt_file}")
    print(f"✓ Saved {split_name} indices: {indices_file}")


def load_indices(output_dir: str, split_name: str = "test", 
                indices_filename: str = None) -> set:
    """
    Load indices from existing split to avoid overlap.
    
    Args:
        output_dir: Output directory
        split_name: Name of the split
        indices_filename: Optional custom filename
        
    Returns:
        Set of indices
    """
    if indices_filename:
        indices_file = os.path.join(output_dir, indices_filename)
    else:
        indices_file = os.path.join(output_dir, f"{split_name}_indices.tsv")
    
    if os.path.exists(indices_file):
        df_indices = pd.read_csv(indices_file, sep='\t', encoding='utf-8')
        return set(df_indices['DF_INDEX'].tolist())
    return set()


def load_tsv(tsv_path: str) -> pd.DataFrame:
    """
    Load TSV file into DataFrame.
    
    Args:
        tsv_path: Path to TSV file
        
    Returns:
        DataFrame
        
    Raises:
        FileNotFoundError: If TSV file doesn't exist
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    return pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
