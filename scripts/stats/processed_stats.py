"""
Processed corpus statistics computation.
Analyzes TSV files after extraction/filtering.
"""

import os
import pandas as pd
from typing import Dict


def compute_processed_stats(tsv_path: str, tokenize_func) -> Dict[str, Dict]:
    """
    Compute statistics from processed TSV file (after extraction/filtering).
    
    Args:
        tsv_path: Path to TSV file with columns: corpus, src, tgt, etc.
        tokenize_func: Tokenization function to use (e.g., tokenize_for_stats)
        
    Returns:
        Dict mapping corpus name to stats dict with keys:
        - sentences: Number of sentence pairs
        - tokens_src: Token count in source column
        - tokens_tgt: Token count in target column
        - text_types: Dict of text type distributions
        - corrected: Dict of correction status counts
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    # Load TSV
    df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8', on_bad_lines='warn')
    
    # Filter out corrupted rows (numeric text_type)
    if 'text_type' in df.columns:
        df = df[~df['text_type'].astype(str).str.match(r'^\d+$', na=False)]
    
    stats = {}
    
    for corpus_name, group in df.groupby('corpus'):
        # Count sentences
        n_sentences = len(group)
        
        # Count tokens in src and tgt columns
        tokens_src = sum(len(tokenize_func(str(sent))) for sent in group['src'])
        tokens_tgt = sum(len(tokenize_func(str(sent))) for sent in group['tgt'])
        
        # Text type distribution
        text_types = group['text_type'].value_counts().to_dict() if 'text_type' in group.columns else {}
        
        # Correction status
        corrected = group['corrected'].value_counts().to_dict() if 'corrected' in group.columns else {}
        
        stats[corpus_name] = {
            'sentences': n_sentences,
            'tokens_src': tokens_src,
            'tokens_tgt': tokens_tgt,
            'text_types': text_types,
            'corrected': corrected
        }
    
    return stats
