"""
Statistics display and formatting functions.
Pretty-prints statistics tables to console.
"""

from typing import Dict
from collections import defaultdict


def display_raw_stats(stats: Dict[str, Dict]) -> None:
    """
    Display raw XML statistics in formatted table.
    
    Args:
        stats: Dict mapping corpus name to raw statistics
    """
    print("\n" + "="*110)
    print("RAW CORPUS STATISTICS (Pre-Extraction from XML)")
    print("="*110)
    print(f"{'Corpus':<25} {'Files':<10} {'Sentences':<15} {'Tokens (SRC)':<18} {'Tokens (TGT)':<18}")
    print("-"*110)
    
    totals = {'files': 0, 'sentences': 0, 'tokens_src': 0, 'tokens_tgt': 0}
    
    for corpus_name in sorted(stats.keys()):
        s = stats[corpus_name]
        print(f"{corpus_name:<25} {s['files']:<10} {s['sentences']:<15,} "
              f"{s['tokens_src']:<18,} {s['tokens_tgt']:<18,}")
        
        for key in totals:
            totals[key] += s[key]
    
    print("-"*110)
    print(f"{'TOTAL':<25} {totals['files']:<10} {totals['sentences']:<15,} "
          f"{totals['tokens_src']:<18,} {totals['tokens_tgt']:<18,}")
    print("="*110)
    print("NOTE: Sentence counts are approximate (based on punctuation).")
    print("      Token counts from raw XML are rough estimates.\n")


def display_processed_stats(stats: Dict[str, Dict]) -> None:
    """
    Display processed TSV statistics in formatted table.
    
    Args:
        stats: Dict mapping corpus name to processed statistics
    """
    print("\n" + "="*110)
    print("PROCESSED CORPUS STATISTICS (Post-Extraction from TSV)")
    print("="*110)
    print(f"{'Corpus':<25} {'Sentences':<15} {'Tokens (SRC)':<18} {'Tokens (TGT)':<18} {'Corrected':<12}")
    print("-"*110)
    
    totals = {'sentences': 0, 'tokens_src': 0, 'tokens_tgt': 0}
    
    for corpus_name in sorted(stats.keys()):
        s = stats[corpus_name]
        corrected_pct = s['corrected'].get(True, 0) / s['sentences'] * 100 if s['sentences'] > 0 else 0
        
        print(f"{corpus_name:<25} {s['sentences']:<15,} "
              f"{s['tokens_src']:<18,} {s['tokens_tgt']:<18,} {corrected_pct:<11.1f}%")
        
        for key in totals:
            totals[key] += s[key]
    
    print("-"*110)
    print(f"{'TOTAL':<25} {totals['sentences']:<15,} "
          f"{totals['tokens_src']:<18,} {totals['tokens_tgt']:<18,}")
    print("="*110)
    
    # Show text type distribution
    print("\nText Type Distribution:")
    print("-"*80)
    text_type_totals = defaultdict(int)
    for s in stats.values():
        for tt, count in s.get('text_types', {}).items():
            text_type_totals[tt] += count
    
    total_sents = totals['sentences']
    for tt in sorted(text_type_totals.keys()):
        count = text_type_totals[tt]
        pct = count / total_sents * 100 if total_sents > 0 else 0
        print(f"  {tt:<30} {count:<10,} ({pct:5.2f}%)")
    
    # Show correction distribution
    print("\nCorrection Status:")
    print("-"*80)
    correction_totals = defaultdict(int)
    for s in stats.values():
        for corr, count in s.get('corrected', {}).items():
            correction_totals[corr] += count
    
    for corr in sorted(correction_totals.keys()):
        count = correction_totals[corr]
        pct = count / total_sents * 100 if total_sents > 0 else 0
        label = "With corrections" if corr else "No corrections"
        print(f"  {label:<30} {count:<10,} ({pct:5.2f}%)")
    print()


def display_comparison(raw_stats: Dict[str, Dict], processed_stats: Dict[str, Dict]) -> None:
    """
    Display side-by-side comparison of raw vs processed statistics.
    
    Args:
        raw_stats: Dict mapping corpus name to raw statistics
        processed_stats: Dict mapping corpus name to processed statistics
    """
    print("\n" + "="*130)
    print("COMPARISON: Raw XML vs Processed TSV")
    print("="*130)
    print(f"{'Corpus':<25} {'Raw Sents':<15} {'Proc Sents':<15} {'Retention':<12} "
          f"{'Raw Tokens':<15} {'Proc Tokens':<15}")
    print("-"*130)
    
    raw_totals = {'sentences': 0, 'tokens_src': 0}
    proc_totals = {'sentences': 0, 'tokens_src': 0}
    
    all_corpora = sorted(set(raw_stats.keys()) | set(processed_stats.keys()))
    
    for corpus_name in all_corpora:
        raw = raw_stats.get(corpus_name, {'sentences': 0, 'tokens_src': 0})
        proc = processed_stats.get(corpus_name, {'sentences': 0, 'tokens_src': 0})
        
        raw_sents = raw['sentences']
        proc_sents = proc['sentences']
        retention = proc_sents / raw_sents * 100 if raw_sents > 0 else 0
        
        raw_tokens = raw['tokens_src']
        proc_tokens = proc['tokens_src']
        
        print(f"{corpus_name:<25} {raw_sents:<15,} {proc_sents:<15,} {retention:<11.1f}% "
              f"{raw_tokens:<15,} {proc_tokens:<15,}")
        
        raw_totals['sentences'] += raw_sents
        raw_totals['tokens_src'] += raw_tokens
        proc_totals['sentences'] += proc_sents
        proc_totals['tokens_src'] += proc_tokens
    
    total_retention = proc_totals['sentences'] / raw_totals['sentences'] * 100 if raw_totals['sentences'] > 0 else 0
    
    print("-"*130)
    print(f"{'TOTAL':<25} {raw_totals['sentences']:<15,} {proc_totals['sentences']:<15,} "
          f"{total_retention:<11.1f}% {raw_totals['tokens_src']:<15,} {proc_totals['tokens_src']:<15,}")
    print("="*130)
    print(f"\nFiltering removed: {raw_totals['sentences'] - proc_totals['sentences']:,} sentences "
          f"({100 - total_retention:.1f}%)")
    print(f"Reasons: Foreign language, duplicates, too short (≤4 words), punctuation-only\n")
