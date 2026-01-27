"""
Logic for applying detected operations to update TSV DataFrames.
"""

import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path


def apply_operations_to_corpus(
    tsv_df: pd.DataFrame,
    norm_sentences: List[Tuple[str, str, int, int]],
    operations: List[Dict]
) -> pd.DataFrame:
    """
    Apply detected operations to update a corpus in the TSV DataFrame.
    
    Args:
        tsv_df: DataFrame rows for this corpus
        norm_sentences: List of (src, tgt, line_start, line_end) from NORM file
        operations: List of operation dictionaries
        
    Returns:
        Updated DataFrame for this corpus
    """
    new_rows = []
    
    for op in operations:
        if op['type'] == 'keep':
            # Keep original row unchanged
            tsv_idx = op['tsv_start']
            new_rows.append(tsv_df.iloc[tsv_idx].to_dict())
        
        elif op['type'] == 'edit':
            # Update target only
            tsv_idx = op['tsv_start']
            norm_idx = op['norm_start']
            
            row = tsv_df.iloc[tsv_idx].to_dict()
            norm_src, norm_tgt, _, _ = norm_sentences[norm_idx]
            
            row['src'] = norm_src
            row['tgt'] = norm_tgt
            new_rows.append(row)
        
        elif op['type'] == 'merge':
            # 2 TSV rows -> 1 NORM sentence
            tsv_start = op['tsv_start']
            norm_idx = op['norm_start']
            
            # Use metadata from first TSV row
            row = tsv_df.iloc[tsv_start].to_dict()
            norm_src, norm_tgt, _, _ = norm_sentences[norm_idx]
            
            row['src'] = norm_src
            row['tgt'] = norm_tgt
            new_rows.append(row)
        
        elif op['type'] == 'split':
            # 1 TSV row -> 2 NORM sentences
            tsv_idx = op['tsv_start']
            norm_start = op['norm_start']
            norm_end = op['norm_end']
            
            # Create multiple rows with same metadata
            original_row = tsv_df.iloc[tsv_idx].to_dict()
            
            for i in range(norm_start, norm_end):
                norm_src, norm_tgt, _, _ = norm_sentences[i]
                row = original_row.copy()
                row['src'] = norm_src
                row['tgt'] = norm_tgt
                new_rows.append(row)
        
        # 'delete' operations don't add any rows
    
    return pd.DataFrame(new_rows)


def get_norm_path_for_corpus(corpus_name: str, extract_dir: Path) -> str:
    """
    Construct NORM file path from corpus name.
    
    Args:
        corpus_name: Name of the corpus
        extract_dir: Directory containing NORM files
        
    Returns:
        Path to NORM file if found, None otherwise
    """
    # Try common suffixes
    for suffix in ['_full', '_edited', '_with_meta', '']:
        norm_file = extract_dir / f"{corpus_name}{suffix}.norm"
        if norm_file.exists():
            return str(norm_file)
    return None


def find_norm_files_in_directory(directory: str, exclude_tsv: bool = True) -> List[str]:
    """
    Find all .norm files in directory.
    
    Args:
        directory: Directory to search
        exclude_tsv: If True, skip .tsv files (default: True)
        
    Returns:
        Sorted list of .norm file paths
    """
    norm_files = []
    
    for file in Path(directory).iterdir():
        if file.is_file():
            # Skip TSV files if requested
            if exclude_tsv and file.suffix.lower() == '.tsv':
                continue
            
            # Include .norm files
            if file.suffix.lower() == '.norm':
                norm_files.append(str(file))
    
    return sorted(norm_files)
