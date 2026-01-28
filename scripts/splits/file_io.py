"""
File I/O utilities for dataset splits.
"""

import os
import pandas as pd
from typing import List
from configs import Paths


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
    Uses paths from configs.py (Paths class) by default.
    
    Args:
        df_split: Split DataFrame
        split_name: Name of the split ('test', 'train', 'dev')
        output_dir: Output directory (used only if paths_config is None)
        indices: List of indices in this split
        df_full: Full DataFrame (for metadata)
        paths_config: Optional dict with path mappings (TEST_SRC, TRAIN_SRC, etc.)
    """
    # Determine file paths from configs or fallback
    if paths_config:
        src_file = paths_config.get(f'{split_name.upper()}_SRC')
        tgt_file = paths_config.get(f'{split_name.upper()}_TGT')
        indices_file = paths_config.get(f'{split_name.upper()}_IDXS')
    else:
        # Try to use Paths from configs
        src_file = getattr(Paths, f'{split_name.upper()}_SRC', None)
        tgt_file = getattr(Paths, f'{split_name.upper()}_TGT', None)
        indices_file = getattr(Paths, f'{split_name.upper()}_IDXS', None)
    
    # Final fallback to output_dir if nothing found
    if not src_file:
        src_file = os.path.join(output_dir, split_name, f"{split_name}.src")
    if not tgt_file:
        tgt_file = os.path.join(output_dir, split_name, f"{split_name}.tgt")
    if not indices_file:
        indices_file = os.path.join(output_dir, split_name, f"{split_name}_indices.tsv")

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


def load_indices(output_dir: str = None, split_name: str = "test", 
                indices_filename: str = None) -> set:
    """
    Load indices from existing split to avoid overlap.
    Uses paths from configs.py (Paths class) by default.
    
    Args:
        output_dir: Output directory (optional, uses Paths from configs if not provided)
        split_name: Name of the split ('test', 'train', 'dev')
        indices_filename: Optional custom filename
        
    Returns:
        Set of indices
    """
    if indices_filename:
        # Custom filename provided
        if output_dir:
            indices_file = os.path.join(output_dir, indices_filename)
        else:
            indices_file = indices_filename
    else:
        # Try to get path from Paths config
        indices_file = getattr(Paths, f'{split_name.upper()}_IDXS', None)
        
        # Fallback to constructing from output_dir
        if not indices_file and output_dir:
            indices_file = os.path.join(output_dir, split_name, f"{split_name}_indices.tsv")
        elif not indices_file:
            # Last resort: use default from Paths.SET_SPLITS
            indices_file = os.path.join(Paths.SET_SPLITS, split_name, f"{split_name}_indices.tsv")
    
    if os.path.exists(indices_file):
        df_indices = pd.read_csv(indices_file, sep='\t', encoding='utf-8')
        return set(df_indices['DF_INDEX'].tolist())
    return set()


def load_tsv(tsv_path: str = None) -> pd.DataFrame:
    """
    Load TSV file into DataFrame.
    Uses Paths.EXTRACT_TSV from configs.py if path not provided.
    
    Args:
        tsv_path: Path to TSV file (optional, uses Paths.EXTRACT_TSV if not provided)
        
    Returns:
        DataFrame
        
    Raises:
        FileNotFoundError: If TSV file doesn't exist
    """
    # Use default from configs if not provided
    if tsv_path is None:
        tsv_path = Paths.EXTRACT_TSV
    
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8', on_bad_lines='warn')
    # Filter out rows where text_type is numeric (data corruption)
    if 'text_type' in df.columns:
        df = df[~df['text_type'].astype(str).str.match(r'^\d+$', na=False)]
    return df

def load_files(output_dir: str = None, split_name: str = "train",
               src_path: str = None, tgt_path: str = None) -> tuple:
    """
    Load source and target files for a split.
    Uses paths from configs.py (Paths class) by default.
    
    Args:
        output_dir: Output directory (optional)
        split_name: Name of the split ('test', 'train', 'dev')
        src_path: Override path for source file
        tgt_path: Override path for target file
        
    Returns:
        Tuple of (source_lines, target_lines) as lists
    """
    # Determine source file path
    if src_path:
        src_file = src_path
    else:
        src_file = getattr(Paths, f'{split_name.upper()}_SRC', None)
        if not src_file and output_dir:
            src_file = os.path.join(output_dir, split_name, f"{split_name}.src")
    
    # Determine target file path
    if tgt_path:
        tgt_file = tgt_path
    else:
        tgt_file = getattr(Paths, f'{split_name.upper()}_TGT', None)
        if not tgt_file and output_dir:
            tgt_file = os.path.join(output_dir, split_name, f"{split_name}.tgt")
    
    # Load source
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    
    # Load target
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f]
    
    return src_lines, tgt_lines