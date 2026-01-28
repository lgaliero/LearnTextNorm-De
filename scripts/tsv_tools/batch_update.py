"""
High-level batch update orchestration for TSV files.
Coordinates parsing, diff detection, and applying updates.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict

from .norm_parser import parse_norm_file_simple
from .diff_core import detect_operations, calculate_operation_stats
from .apply_ops import apply_operations_to_corpus

def recalculate_sent_num(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate sent_num for each file after operations (merges, splits, etc).
    Groups by corpus and xml_file, then renumbers sentences sequentially.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Group by corpus and xml_file
    for (corpus, xml_file), group_idx in df.groupby(['corpus', 'xml_file']).groups.items():
        # Renumber sequentially starting from 1
        new_sent_nums = list(range(1, len(group_idx) + 1))
        df.loc[group_idx, 'sent_num'] = new_sent_nums
    
    return df

def batch_update_tsv(
    tsv_path: str,
    norm_files: List[str],
    output_path: str = None,
    log_edits: bool = True
) -> pd.DataFrame:
    """
    Update TSV file from multiple NORM files with automatic index adjustment.
    
    Args:
        tsv_path: Path to TSV file
        norm_files: List of NORM file paths
        output_path: Output path (overwrites tsv_path if None)
        log_edits: Whether to create detailed edit log
        
    Returns:
        Updated DataFrame
    """
    # Load TSV
    print(f"\n{'='*80}")
    print(f"BATCH TSV UPDATE")
    print(f"{'='*80}")
    print(f"\nLoading TSV: {tsv_path}")
    
    df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
    print(f"✓ Loaded {len(df):,} rows from TSV")
    
    # Prepare tracking
    all_edit_details = {}
    all_stats = {}
    updated_corpora = []
    
    # Process each NORM file
    for norm_path in norm_files:
        corpus_name = Path(norm_path).stem
        # Remove common suffixes
        for suffix in ['_full', '_edited', '_with_meta']:
            if corpus_name.endswith(suffix):
                corpus_name = corpus_name[:-len(suffix)]
                break
        
        print(f"\n{'='*80}")
        print(f"Processing: {corpus_name}")
        print(f"NORM file: {Path(norm_path).name}")
        print(f"{'='*80}")
        
        # Filter TSV rows for this corpus
        corpus_mask = df['corpus'] == corpus_name
        corpus_df = df[corpus_mask].copy()
        
        if len(corpus_df) == 0:
            print(f"⚠️  No rows found in TSV for corpus '{corpus_name}'. Skipping.")
            continue
        
        print(f"  TSV rows for this corpus: {len(corpus_df):,}")
        
        # Parse NORM file
        try:
            norm_sentences = parse_norm_file_simple(norm_path)
            print(f"  NORM sentences: {len(norm_sentences):,}")
        except Exception as e:
            print(f"❌ Error parsing NORM file: {e}")
            continue
        
        # Detect operations
        try:
            operations, edit_details = detect_operations(corpus_df, norm_sentences)
        except Exception as e:
            print(f"❌ Error detecting operations: {e}")
            continue
        
        # Calculate statistics
        stats = calculate_operation_stats(edit_details)
        all_stats[corpus_name] = stats
        all_edit_details[corpus_name] = edit_details
        
        print(f"\n  Operations detected:")
        print(f"    • Kept:    {stats['keep']:,}")
        print(f"    • Edited:  {stats['edit']:,}")
        print(f"    • Split:   {stats['split']:,}")
        print(f"    • Merged:  {stats['merge']:,}")
        print(f"    • Deleted: {stats['delete']:,}")
        
        # Apply operations
        try:
            updated_corpus_df = apply_operations_to_corpus(
                corpus_df, norm_sentences, operations
            )
            
            # Replace corpus rows in main dataframe
            df = df[~corpus_mask]  # Remove old rows
            df = pd.concat([df, updated_corpus_df], ignore_index=True)
            
            updated_corpora.append(corpus_name)
            print(f"  ✓ Updated {len(updated_corpus_df):,} rows")
            
        except Exception as e:
            print(f"❌ Error applying operations: {e}")
            continue
    
    # Save updated TSV
    # Save updated TSV
    if output_path is None:
        output_path = tsv_path
    
    # Recalculate sent_num to fix indices after merges/splits
    df = recalculate_sent_num(df)
    
    df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    print(f"\n{'='*80}")
    print(f"BATCH UPDATE COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Updated {len(updated_corpora)} corpora: {', '.join(updated_corpora)}")
    print(f"✓ Saved to: {output_path}")
    print(f"  Total rows: {len(df):,}")
    
    # Create edit log if requested
    if log_edits:
        log_path = output_path.replace('.tsv', '_edit_log.txt')
        _write_edit_log(log_path, all_edit_details, all_stats)
        print(f"📝 Edit log saved: {log_path}")
    
    return df


def _write_edit_log(log_path: str, all_edit_details: Dict, all_stats: Dict):
    """Write detailed edit log to file."""
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BATCH UPDATE EDIT LOG\n")
        f.write("="*80 + "\n\n")
        
        for corpus_name, details in all_edit_details.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"CORPUS: {corpus_name}\n")
            f.write(f"{'='*80}\n")
            
            stats = all_stats[corpus_name]
            f.write(f"\nSummary:\n")
            f.write(f"  • Kept:    {stats['keep']}\n")
            f.write(f"  • Edited:  {stats['edit']}\n")
            f.write(f"  • Split:   {stats['split']}\n")
            f.write(f"  • Merged:  {stats['merge']}\n")
            f.write(f"  • Deleted: {stats['delete']}\n")
            
            # Group by operation type
            for op_type in ['edit', 'split', 'merge', 'delete']:
                ops = [d for d in details if d['type'] == op_type]
                if ops:
                    f.write(f"\n{op_type.upper()} OPERATIONS ({len(ops)}):\n")
                    f.write("-"*80 + "\n")
                    for op in ops[:20]:  # Limit to 20 per type
                        f.write(f"\nPosition: {op['tsv_position']}\n")
                        f.write(f"File: {op['xml_file']} | Sent: {op['sent_num']}\n")
                        f.write(f"Original SRC: {op['original_src'][:100]}...\n")
                        f.write(f"Original TGT: {op['original_tgt'][:100]}...\n")
                        f.write(f"New SRC: {op['new_src'][:100]}...\n")
                        f.write(f"New TGT: {op['new_tgt'][:100]}...\n")
                    
                    if len(ops) > 20:
                        f.write(f"\n... and {len(ops) - 20} more {op_type} operations\n")
