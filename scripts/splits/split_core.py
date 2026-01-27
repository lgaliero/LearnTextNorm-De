"""
Core stratified data splitting logic.
Maintains proportions of subcorpora, text types, and correction ratios.
"""

import random
import pandas as pd
from typing import Tuple, List, Set


def stratify_sample(
    df: pd.DataFrame,
    sample_size: float,
    split_name: str,
    excluded_indices: Set[int] = None
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Perform stratified sampling by iteratively picking random line numbers from strata.
    Maintains proportions by tracking how many samples each stratum needs.
    Excludes any indices already used in previous sets.
    
    Args:
        df: DataFrame with corpus data
        sample_size: Proportion of data to sample (0.0 to 1.0)
        split_name: Name of the split (for logging)
        excluded_indices: Set of indices to exclude from sampling
        
    Returns:
        Tuple of (sampled DataFrame, list of sampled indices)
    """
    if excluded_indices is None:
        excluded_indices = set()
    
    # Create stratification key
    df['strat_key'] = (
        df['corpus'].astype(str) + "_" + 
        df['text_type'].astype(str) + "_" + 
        df['corrected'].astype(str)
    )
    
    print(f"\n{'='*80}")
    print(f"STRATIFIED SAMPLING FOR {split_name.upper()} SET")
    print(f"{'='*80}")
    
    # Build stratum info: available line numbers (excluding already used) and target sample size
    strata_info = {}
    total_available = 0
    
    for strat_key, group in df.groupby('strat_key'):
        # Get line numbers for this stratum, EXCLUDING any already used indices
        available_line_numbers = [idx for idx in group.index if idx not in excluded_indices]
        
        if len(available_line_numbers) == 0:
            continue
        
        n_total = len(available_line_numbers)
        n_sample = max(1, int(n_total * sample_size))
        n_sample = min(n_sample, n_total)
        
        strata_info[strat_key] = {
            'available': available_line_numbers.copy(),
            'target': n_sample,
            'sampled': []
        }
        total_available += n_total
    
    print(f"Total available indices (excluding {len(excluded_indices)} already used): {total_available:,}")
    
    # Iteratively pick random lines from strata until all targets are met
    sampled_indices = []
    
    while True:
        # Find strata that still need samples
        active_strata = {k: v for k, v in strata_info.items() 
                        if len(v['sampled']) < v['target'] and len(v['available']) > 0}
        
        if not active_strata:
            break
        
        # Pick a random stratum from those that need samples
        strat_key = random.choice(list(active_strata.keys()))
        
        # Pick a random line number from this stratum's available pool
        line_num = random.choice(strata_info[strat_key]['available'])
        
        # Record the sample
        strata_info[strat_key]['sampled'].append(line_num)
        strata_info[strat_key]['available'].remove(line_num)
        sampled_indices.append(line_num)
    
    # Prepare sampling details for reporting
    sampling_details = []
    for strat_key, info in strata_info.items():
        parts = strat_key.split('_')
        if len(parts) >= 3:
            corpus = '_'.join(parts[:-2])
            text_type = parts[-2]
            corrected = parts[-1]
        else:
            corpus, text_type, corrected = strat_key, "unknown", "unknown"
        
        total_available = len(info['available']) + len(info['sampled'])
        sampling_details.append({
            'stratum': strat_key,
            'corpus': corpus,
            'text_type': text_type,
            'corrected': corrected,
            'available_lines': total_available,
            'sampled': len(info['sampled']),
            'percentage': f"{len(info['sampled'])/total_available*100:.2f}%",
            'sample_line_numbers': info['sampled'][:5]
        })
    
    # Create sampled dataframe (keep extraction order)
    df_sampled = df.loc[sampled_indices].copy()
    
    # Print sampling report
    print(f"\nSampled {len(sampled_indices):,} sentences from {len(strata_info)} strata")
    print("\nSampling breakdown:")
    print("-" * 80)
    
    for detail in sampling_details:
        print(f"\n{detail['stratum']}")
        print(f"  Corpus: {detail['corpus']}, Type: {detail['text_type']}, Corrected: {detail['corrected']}")
        print(f"  Sampled: {detail['sampled']}/{detail['available_lines']} ({detail['percentage']})")
        print(f"  Example indices: {detail['sample_line_numbers']}")
    
    return df_sampled, sampled_indices


def check_proportions(df_split: pd.DataFrame, df_full: pd.DataFrame, split_name: str) -> None:
    """
    Check and report proportions of different dimensions in the split.
    
    Args:
        df_split: Split DataFrame
        df_full: Full DataFrame
        split_name: Name of the split (for logging)
    """
    print(f"\n{'='*80}")
    print(f"PROPORTION CHECK: {split_name.upper()} SET")
    print(f"{'='*80}")
    
    # Check corpus proportions
    print("\nCorpus distribution:")
    corpus_full = df_full['corpus'].value_counts(normalize=True)
    corpus_split = df_split['corpus'].value_counts(normalize=True)
    
    for corpus in corpus_full.index:
        full_pct = corpus_full.get(corpus, 0) * 100
        split_pct = corpus_split.get(corpus, 0) * 100
        diff = split_pct - full_pct
        print(f"  {corpus:20s}: {split_pct:5.2f}% (full: {full_pct:5.2f}%, diff: {diff:+5.2f}%)")
    
    # Check text type proportions
    print("\nText type distribution:")
    texttype_full = df_full['text_type'].value_counts(normalize=True)
    texttype_split = df_split['text_type'].value_counts(normalize=True)
    
    for text_type in texttype_full.index:
        full_pct = texttype_full.get(text_type, 0) * 100
        split_pct = texttype_split.get(text_type, 0) * 100
        diff = split_pct - full_pct
        print(f"  {text_type:20s}: {split_pct:5.2f}% (full: {full_pct:5.2f}%, diff: {diff:+5.2f}%)")
    
    # Check correction ratio
    print("\nCorrection distribution:")
    corrected_full = df_full['corrected'].value_counts(normalize=True)
    corrected_split = df_split['corrected'].value_counts(normalize=True)
    
    for corrected in corrected_full.index:
        full_pct = corrected_full.get(corrected, 0) * 100
        split_pct = corrected_split.get(corrected, 0) * 100
        diff = split_pct - full_pct
        print(f"  {str(corrected):20s}: {split_pct:5.2f}% (full: {full_pct:5.2f}%, diff: {diff:+5.2f}%)")
