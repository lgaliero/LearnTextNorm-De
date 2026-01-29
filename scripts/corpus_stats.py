"""
Corpus statistics computation module.
Supports three modes: raw (pre-extraction XML), processed (post-extraction TSV), or both.
"""

import os
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Literal
from collections import defaultdict
import xml.etree.ElementTree as ET


def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization for statistics (splits on whitespace and punctuation)."""
    return [t for t in re.findall(r'\w+|[^\w\s]', text) if t.strip()]


# ==============================================================================
# RAW XML STATISTICS (Pre-extraction)
# ==============================================================================

def count_sentences_in_xml(xml_path: str, corpus_type: str) -> int:
    """Count sentences in raw XML using sentence-ending punctuation."""
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            xml_content = f.read()
        
        # Simple XML parsing without full extraction pipeline
        root = ET.fromstring(xml_content)
        
        # Get all text content
        text = ''.join(root.itertext())
        
        # Count sentence-ending punctuation
        return len(re.findall(r'[.!?]+', text))
    
    except Exception as e:
        print(f"    ERROR reading {os.path.basename(xml_path)}: {e}")
        return 0


def count_tokens_in_xml(xml_path: str) -> Dict[str, int]:
    """Count tokens in raw XML (approximate - doesn't distinguish src/tgt well)."""
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            xml_content = f.read()
        
        root = ET.fromstring(xml_content)
        text = ''.join(root.itertext())
        tokens = tokenize_simple(text)
        
        # Return same count for src and tgt (raw XML doesn't separate cleanly)
        return {'src': len(tokens), 'tgt': len(tokens)}
    
    except Exception as e:
        print(f"    ERROR reading {os.path.basename(xml_path)}: {e}")
        return {'src': 0, 'tgt': 0}


def compute_raw_stats(corpus_configs: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute statistics from raw XML files (before extraction/filtering).
    
    Args:
        corpus_configs: Dict mapping corpus name to config with 'base_dir' key
        
    Returns:
        Dict mapping corpus name to stats dict with keys:
        - files: Number of XML files
        - sentences: Approximate sentence count
        - tokens_src: Token count (source)
        - tokens_tgt: Token count (target, often same as src for raw)
    """
    stats = {}
    
    for corpus_name, cfg in corpus_configs.items():
        base_dir = cfg.get("base_dir", cfg.get("xml_dir", ""))
        
        if not os.path.isdir(base_dir):
            print(f"  WARNING: Directory not found: {base_dir}")
            stats[corpus_name] = {
                'files': 0,
                'sentences': 0,
                'tokens_src': 0,
                'tokens_tgt': 0
            }
            continue
        
        # Find all XML files
        xml_files = []
        for root_dir, dirs, files in os.walk(base_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in sorted(files):
                if f.lower().endswith(".xml") and not f.endswith(".pretty"):
                    xml_files.append(os.path.join(root_dir, f))
        
        # Count statistics
        total_sentences = 0
        total_tokens_src = 0
        total_tokens_tgt = 0
        
        for xml_path in xml_files:
            sentences = count_sentences_in_xml(xml_path, corpus_name)
            tokens = count_tokens_in_xml(xml_path)
            
            total_sentences += sentences
            total_tokens_src += tokens['src']
            total_tokens_tgt += tokens['tgt']
        
        stats[corpus_name] = {
            'files': len(xml_files),
            'sentences': total_sentences,
            'tokens_src': total_tokens_src,
            'tokens_tgt': total_tokens_tgt
        }
    
    return stats


# ==============================================================================
# PROCESSED TSV STATISTICS (Post-extraction)
# ==============================================================================

def compute_processed_stats(tsv_path: str) -> Dict[str, Dict]:
    """
    Compute statistics from processed TSV file (after extraction/filtering).
    
    Args:
        tsv_path: Path to TSV file with columns: corpus, src, tgt, etc.
        
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
        tokens_src = sum(len(tokenize_simple(str(sent))) for sent in group['src'])
        tokens_tgt = sum(len(tokenize_simple(str(sent))) for sent in group['tgt'])
        
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


# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

def display_raw_stats(stats: Dict[str, Dict]) -> None:
    """Display raw XML statistics in formatted table."""
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
    """Display processed TSV statistics in formatted table."""
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
    """Display side-by-side comparison of raw vs processed statistics."""
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


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def compute_corpus_statistics(
    mode: Literal['raw', 'processed', 'both'] = 'both',
    corpus_configs: Optional[Dict[str, Dict]] = None,
    tsv_path: Optional[str] = None,
    output_json: Optional[str] = None
) -> Dict:
    """
    Compute corpus statistics in specified mode.
    
    Args:
        mode: One of 'raw' (XML), 'processed' (TSV), or 'both'
        corpus_configs: Dict of corpus configurations (required for 'raw' or 'both')
        tsv_path: Path to processed TSV file (required for 'processed' or 'both')
        output_json: Optional path to save statistics as JSON
        
    Returns:
        Dict containing statistics based on mode
    """
    results = {}
    
    if mode in ['raw', 'both']:
        if corpus_configs is None:
            raise ValueError("corpus_configs required for 'raw' or 'both' modes")
        
        print("\n[1/2] Computing raw XML statistics..." if mode == 'both' else "\nComputing raw XML statistics...")
        raw_stats = compute_raw_stats(corpus_configs)
        results['raw'] = raw_stats
        
        if mode == 'raw':
            display_raw_stats(raw_stats)
    
    if mode in ['processed', 'both']:
        if tsv_path is None:
            raise ValueError("tsv_path required for 'processed' or 'both' modes")
        
        print("\n[2/2] Computing processed TSV statistics..." if mode == 'both' else "\nComputing processed TSV statistics...")
        processed_stats = compute_processed_stats(tsv_path)
        results['processed'] = processed_stats
        
        if mode == 'processed':
            display_processed_stats(processed_stats)
    
    # Display results
    if mode == 'both':
        display_raw_stats(results['raw'])
        display_processed_stats(results['processed'])
        display_comparison(results['raw'], results['processed'])
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to: {output_json}")
    
    return results


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute corpus statistics in different modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Raw XML statistics only
  python corpus_stats.py --mode raw
  
  # Processed TSV statistics only
  python corpus_stats.py --mode processed --tsv ../master_files/all_corpora.tsv
  
  # Both with comparison
  python corpus_stats.py --mode both --tsv ../master_files/all_corpora.tsv
  
  # Save results to JSON
  python corpus_stats.py --mode both --tsv ../master_files/all_corpora.tsv --output stats.json
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['raw', 'processed', 'both'],
        default='both',
        help='Statistics mode (default: both)'
    )
    parser.add_argument(
        '--tsv',
        help='Path to processed TSV file (required for processed/both modes)'
    )
    parser.add_argument(
        '--output',
        help='Save statistics to JSON file'
    )
    
    args = parser.parse_args()
    
    # Load configs
    try:
        from configs import Paths, ExtractionParams
        corpus_configs = ExtractionParams.CORPORA
        default_tsv = Paths.EXTRACT_TSV
    except ImportError:
        print("WARNING: configs.py not found, using defaults")
        corpus_configs = {}
        default_tsv = None
    
    # Use provided TSV or default
    tsv_path = args.tsv or default_tsv
    
    # Validate inputs
    if args.mode in ['processed', 'both'] and not tsv_path:
        parser.error(f"--tsv required for mode '{args.mode}'")
    
    if args.mode in ['raw', 'both'] and not corpus_configs:
        parser.error(f"corpus_configs not found in configs.py (required for mode '{args.mode}')")
    
    # Run statistics
    compute_corpus_statistics(
        mode=args.mode,
        corpus_configs=corpus_configs if args.mode in ['raw', 'both'] else None,
        tsv_path=tsv_path if args.mode in ['processed', 'both'] else None,
        output_json=args.output
    )