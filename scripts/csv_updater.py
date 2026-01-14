"""
Batch update CSV file from multiple edited NORM files with automatic index adjustment.

This script intelligently detects splits, merges, and deletions by comparing
NORM sentences with CSV content, WITHOUT requiring metadata markers.

Usage:
    python update_csv_from_norm.py batch-update \
        --csv-file output/all_corpora.csv \
        --norm-files output/LEONIDE_full.norm output/Kolipsi_1_L2_full.norm
"""

import re
import argparse
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import glob
from difflib import SequenceMatcher

def parse_norm_file_simple(norm_path: str) -> List[Tuple[str, str]]:
    """Parse NORM file into list of (src_sentence, tgt_sentence) pairs."""
    sentences = []
    current_src = []
    current_tgt = []
    
    with open(norm_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            if line.startswith('#'):
                continue
            
            if not line.strip():
                if current_src or current_tgt:
                    src_sent = ' '.join(current_src).strip()
                    tgt_sent = ' '.join(current_tgt).strip()
                    sentences.append((src_sent, tgt_sent))
                    current_src = []
                    current_tgt = []
                continue
            
            parts = line.split('\t')
            
            if len(parts) == 1:
                word = parts[0].strip()
                if word:
                    current_src.append(word)
            elif len(parts) == 2:
                src_word = parts[0].strip()
                tgt_word = parts[1].strip()
                
                if src_word:
                    current_src.append(src_word)
                if tgt_word:
                    current_tgt.append(tgt_word)
    
    if current_src or current_tgt:
        src_sent = ' '.join(current_src).strip()
        tgt_sent = ' '.join(current_tgt).strip()
        sentences.append((src_sent, tgt_sent))
    
    return sentences


def normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy matching."""
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fuzzy_match_score(str1: str, str2: str) -> float:
    """Calculate fuzzy match score (0.0 to 1.0)."""
    norm1 = normalize_for_comparison(str1)
    norm2 = normalize_for_comparison(str2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def detect_operations(csv_sentences: List[Tuple[str, str]], 
                      norm_sentences: List[Tuple[str, str]],
                      threshold: float = 0.85) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect splits, merges, deletions, and edits by comparing CSV with NORM.
    
    Returns:
        operations: List of operation dicts
        edit_details: List of detailed edit information for reporting
    """
    operations = []
    edit_details = []
    csv_i = 0
    norm_i = 0
    
    print(f"\n  Analyzing differences (CSV: {len(csv_sentences)}, NORM: {len(norm_sentences)})...")
    
    while csv_i < len(csv_sentences) or norm_i < len(norm_sentences):
        if csv_i >= len(csv_sentences):
            # Remaining NORM sentences are additions (shouldn't happen normally)
            operations.append({
                'type': 'addition',
                'csv_start': csv_i,
                'csv_end': csv_i,
                'norm_start': norm_i,
                'norm_end': len(norm_sentences)
            })
            break
        
        if norm_i >= len(norm_sentences):
            # Remaining CSV sentences were deleted
            for i in range(csv_i, len(csv_sentences)):
                edit_details.append({
                    'type': 'delete',
                    'csv_position': i + 1,
                    'original_src': csv_sentences[i][0],  # FULL sentence
                    'original_tgt': csv_sentences[i][1],  # FULL sentence
                    'new_src': '',
                    'new_tgt': '',
                    'match_score': None
                })
            operations.append({
                'type': 'delete',
                'csv_start': csv_i,
                'csv_end': len(csv_sentences),
                'norm_start': norm_i,
                'norm_end': norm_i
            })
            break
        
        csv_src, csv_tgt = csv_sentences[csv_i]
        norm_src, norm_tgt = norm_sentences[norm_i]
        
        # Check for exact or near match (allowing for minor edits)
        match_score = fuzzy_match_score(csv_src, norm_src)
        
        if match_score >= threshold:
            # Simple keep or edit
            op_type = 'edit' if (csv_src != norm_src or csv_tgt != norm_tgt) else 'keep'
            
            # DEBUG: Print first 10 "edits" to see what's different
            if op_type == 'edit' and csv_i < 10:
                print(f"\n  DEBUG Edit #{csv_i + 1} (score: {match_score:.2%}):")
                print(f"    CSV SRC: {csv_src[:100]}")
                print(f"    NORM SRC: {norm_src[:100]}")
                print(f"    Same? {csv_src == norm_src}")
            
            if op_type == 'edit':
                edit_details.append({
                    'type': 'edit',
                    'csv_position': csv_i + 1,
                    'original_src': csv_src,  # FULL sentence
                    'original_tgt': csv_tgt,  # FULL sentence
                    'new_src': norm_src,      # FULL sentence
                    'new_tgt': norm_tgt,      # FULL sentence
                    'match_score': match_score
                })
            elif op_type == 'keep':
                # Also log kept sentences with their score
                edit_details.append({
                    'type': 'keep',
                    'csv_position': csv_i + 1,
                    'original_src': csv_src,
                    'original_tgt': csv_tgt,
                    'new_src': norm_src,
                    'new_tgt': norm_tgt,
                    'match_score': match_score
                })
            
            operations.append({
                'type': op_type,
                'csv_start': csv_i,
                'csv_end': csv_i + 1,
                'norm_start': norm_i,
                'norm_end': norm_i + 1
            })
            csv_i += 1
            norm_i += 1
        
        else:
            # Potential split, merge, or deletion
            split_detected = False
            
            # Look ahead in NORM to see if multiple sentences combine to match CSV
            combined_norm = norm_src
            for lookahead in range(1, min(5, len(norm_sentences) - norm_i)):
                combined_norm += ' ' + norm_sentences[norm_i + lookahead][0]
                combined_score = fuzzy_match_score(csv_src, combined_norm)
                
                if combined_score >= threshold:
                    # Split: 1 CSV → multiple NORM
                    split_parts = [norm_sentences[norm_i + j][0] for j in range(lookahead + 1)]  # FULL sentences
                    split_parts_display = '\n    '.join([f"[{j+1}] {part}" for j, part in enumerate(split_parts)])
                    
                    edit_details.append({
                        'type': 'split',
                        'csv_position': csv_i + 1,
                        'original_src': csv_src,  # FULL sentence
                        'original_tgt': csv_tgt,  # FULL sentence
                        'new_src': f"Split into {lookahead + 1} parts:\n    {split_parts_display}",
                        'new_tgt': '',
                        'match_score': combined_score
                    })
                    
                    operations.append({
                        'type': 'split',
                        'csv_start': csv_i,
                        'csv_end': csv_i + 1,
                        'norm_start': norm_i,
                        'norm_end': norm_i + lookahead + 1
                    })
                    csv_i += 1
                    norm_i += lookahead + 1
                    split_detected = True
                    break
            
            if split_detected:
                continue
            
            # Check if multiple CSV sentences were merged into one NORM sentence
            merge_detected = False
            combined_csv = csv_src
            for lookahead in range(1, min(5, len(csv_sentences) - csv_i)):
                combined_csv += ' ' + csv_sentences[csv_i + lookahead][0]
                combined_score = fuzzy_match_score(combined_csv, norm_src)
                
                if combined_score >= threshold:
                    # Merge: multiple CSV → 1 NORM
                    merged_parts = [csv_sentences[csv_i + j][0] for j in range(lookahead + 1)]  # FULL sentences
                    merged_parts_display = '\n    '.join([f"[{j+1}] {part}" for j, part in enumerate(merged_parts)])
                    
                    edit_details.append({
                        'type': 'merge',
                        'csv_position': csv_i + 1,
                        'original_src': f"Merged from {lookahead + 1} parts:\n    {merged_parts_display}",
                        'original_tgt': '',
                        'new_src': norm_src,  # FULL sentence
                        'new_tgt': norm_tgt,   # FULL sentence
                        'match_score': combined_score
                    })
                    
                    operations.append({
                        'type': 'merge',
                        'csv_start': csv_i,
                        'csv_end': csv_i + lookahead + 1,
                        'norm_start': norm_i,
                        'norm_end': norm_i + 1
                    })
                    csv_i += lookahead + 1
                    norm_i += 1
                    merge_detected = True
                    break
            
            if merge_detected:
                continue
            
            # Check if CSV sentence was deleted
            deletion_detected = False
            if csv_i + 1 < len(csv_sentences):
                next_csv_src = csv_sentences[csv_i + 1][0]
                next_match_score = fuzzy_match_score(next_csv_src, norm_src)
                
                if next_match_score >= threshold:
                    # Deletion detected
                    edit_details.append({
                        'type': 'delete',
                        'csv_position': csv_i + 1,
                        'original_src': csv_src,  # FULL sentence
                        'original_tgt': csv_tgt,  # FULL sentence
                        'new_src': '[DELETED]',
                        'new_tgt': '',
                        'match_score': match_score
                    })
                    
                    operations.append({
                        'type': 'delete',
                        'csv_start': csv_i,
                        'csv_end': csv_i + 1,
                        'norm_start': norm_i,
                        'norm_end': norm_i
                    })
                    csv_i += 1
                    deletion_detected = True
            
            if deletion_detected:
                continue
            
            # Fallback: treat as edit (misalignment)
            edit_details.append({
                'type': 'edit',
                'csv_position': csv_i + 1,
                'original_src': csv_src,  # FULL sentence
                'original_tgt': csv_tgt,  # FULL sentence
                'new_src': norm_src,      # FULL sentence
                'new_tgt': norm_tgt,       # FULL sentence
                'match_score': match_score
            })
            
            operations.append({
                'type': 'edit',
                'csv_start': csv_i,
                'csv_end': csv_i + 1,
                'norm_start': norm_i,
                'norm_end': norm_i + 1
            })
            csv_i += 1
            norm_i += 1
    
    return operations, edit_details


def apply_operations(df: pd.DataFrame, corpus_name: str, 
                     csv_sentences: List[Tuple[str, str]],
                     norm_sentences: List[Tuple[str, str]],
                     operations: List[Dict],
                     edit_details: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply detected operations to update DataFrame with proper index adjustment.
    """
    corpus_df = df[df['corpus'] == corpus_name].copy()
    corpus_indices = corpus_df.index.tolist()
    
    # Keep non-corpus rows
    df_other = df[df['corpus'] != corpus_name].copy()
    
    # Build new rows based on operations
    new_rows = []
    
    stats = {
        'keep': 0,
        'edit': 0,
        'split': 0,
        'merge': 0,
        'delete': 0
    }
    
    for op in operations:
        op_type = op['type']
        csv_start = op['csv_start']
        csv_end = op['csv_end']
        norm_start = op['norm_start']
        norm_end = op['norm_end']
        
        if op_type == 'delete':
            # Skip these CSV rows
            stats['delete'] += (csv_end - csv_start)
            continue
        
        # Get template row from CSV for metadata
        if csv_start < len(corpus_indices):
            template_idx = corpus_indices[csv_start]
            template_row = corpus_df.loc[template_idx].copy()
        else:
            # Shouldn't happen, but handle gracefully
            continue
        
        # Get NORM sentences for this operation
        norm_sents = norm_sentences[norm_start:norm_end]
        
        if op_type == 'keep':
            # No change
            norm_src, norm_tgt = norm_sents[0]
            new_row = template_row.copy()
            new_row['src'] = norm_src
            new_row['tgt'] = norm_tgt
            new_row['corrected'] = (norm_src.strip() != norm_tgt.strip())
            new_rows.append(new_row)
            stats['keep'] += 1
        
        elif op_type == 'edit':
            # Simple edit
            norm_src, norm_tgt = norm_sents[0]
            new_row = template_row.copy()
            new_row['src'] = norm_src
            new_row['tgt'] = norm_tgt
            new_row['corrected'] = (norm_src.strip() != norm_tgt.strip())
            new_rows.append(new_row)
            stats['edit'] += 1
        
        elif op_type == 'split':
            # 1 CSV → multiple NORM: create multiple rows
            for i, (norm_src, norm_tgt) in enumerate(norm_sents):
                new_row = template_row.copy()
                new_row['src'] = norm_src
                new_row['tgt'] = norm_tgt
                new_row['corrected'] = (norm_src.strip() != norm_tgt.strip())
                # Keep same sent_num for first, append suffix for rest
                if i > 0:
                    new_row['sent_num'] = f"{template_row['sent_num']}.{i+1}"
                new_rows.append(new_row)
            stats['split'] += 1
        
        elif op_type == 'merge':
            # Multiple CSV → 1 NORM: use first CSV row's metadata
            norm_src, norm_tgt = norm_sents[0]
            new_row = template_row.copy()
            new_row['src'] = norm_src
            new_row['tgt'] = norm_tgt
            new_row['corrected'] = (norm_src.strip() != norm_tgt.strip())
            new_rows.append(new_row)
            stats['merge'] += 1
    
    # Create new corpus dataframe
    new_corpus_df = pd.DataFrame(new_rows)
    
    # Renumber sent_num sequentially per file
    if len(new_corpus_df) > 0:
        for xml_file in new_corpus_df['xml_file'].unique():
            file_mask = new_corpus_df['xml_file'] == xml_file
            new_corpus_df.loc[file_mask, 'sent_num'] = range(1, file_mask.sum() + 1)
    
    # Combine with other corpora
    df_updated = pd.concat([df_other, new_corpus_df], ignore_index=True)
    
    # Preserve original corpus order from input CSV
    corpus_order = {corpus: i for i, corpus in enumerate(df['corpus'].unique())}
    df_updated['_corpus_order'] = df_updated['corpus'].map(corpus_order)
    df_updated = df_updated.sort_values(['_corpus_order', 'xml_file', 'sent_num']).drop('_corpus_order', axis=1).reset_index(drop=True)
        
    return df_updated, stats, edit_details


def infer_corpus_name(norm_filename: str) -> str:
    """Infer corpus name from NORM filename."""
    name = Path(norm_filename).stem
    name = name.replace('_full', '').replace('_edited', '').replace('_with_meta', '')
    return name


def batch_update_csv_smart(csv_path: str, norm_files: List[str], 
                            output_path: str = None,
                            threshold: float = 0.85,
                            log_edits: bool = True) -> pd.DataFrame:
    """
    Update CSV from multiple NORM files with intelligent split/merge/delete detection.
    
    Args:
        csv_path: Path to CSV file
        norm_files: List of NORM file paths
        output_path: Output CSV path (overwrites original if None)
        threshold: Fuzzy match threshold (0.0-1.0, default 0.85)
        log_edits: If True, write detailed edit log to file
    """
    print("\n" + "="*80)
    print("BATCH UPDATE WITH AUTOMATIC INDEX ADJUSTMENT")
    print("="*80)
    print(f"\nFuzzy match threshold: {threshold} (sentences matching >{threshold*100:.0f}% are considered same)")
    
    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"Total rows: {len(df)}")
    
    # Process each NORM file
    all_stats = {}
    all_edit_details = {}
    
    for norm_file in norm_files:
        norm_filename = Path(norm_file).name
        corpus_name = infer_corpus_name(norm_filename)
        
        print(f"\n{'═'*80}")
        print(f"Processing: {norm_filename} → {corpus_name}")
        print(f"{'═'*80}")
        
        # Parse NORM file
        norm_sentences = parse_norm_file_simple(norm_file)
        print(f"  NORM sentences: {len(norm_sentences)}")
        
        # Get CSV sentences for this corpus
        corpus_df = df[df['corpus'] == corpus_name].copy()
        print(f"  CSV rows:       {len(corpus_df)}")
        
        if len(corpus_df) == 0:
            print(f"  ⚠️  WARNING: No rows found for corpus '{corpus_name}'")
            print(f"  Available corpora: {df['corpus'].unique().tolist()}")
            continue
        
        # Extract CSV sentences
        csv_sentences = [(row['src'], row['tgt']) for _, row in corpus_df.iterrows()]
        
        # Detect operations
        operations, edit_details = detect_operations(csv_sentences, norm_sentences, threshold)
        
        # Apply operations and update DataFrame
        df, stats, edit_details = apply_operations(df, corpus_name, csv_sentences, 
                                                    norm_sentences, operations, edit_details)
        
        # Report statistics
        print(f"\n  Operations detected:")
        print(f"    • Kept unchanged: {stats['keep']}")
        print(f"    • Edited:         {stats['edit']}")
        print(f"    • Split:          {stats['split']} (1 → many)")
        print(f"    • Merged:         {stats['merge']} (many → 1)")
        print(f"    • Deleted:        {stats['delete']}")
        
        total_changes = stats['edit'] + stats['split'] + stats['merge'] + stats['delete']
        print(f"    Total changes: {total_changes}")
        
        all_stats[corpus_name] = stats
        all_edit_details[corpus_name] = edit_details
    
    # Save updated CSV
    if output_path is None:
        output_path = csv_path
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Write detailed edit log
    if log_edits:
        log_path = Path(output_path).parent / f"{Path(output_path).stem}_edit_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED EDIT LOG\n")
            f.write("="*80 + "\n\n")
            
            for corpus_name, edits in all_edit_details.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"CORPUS: {corpus_name}\n")
                f.write(f"{'='*80}\n\n")
                
                for edit in edits:
                    # Format match score if present
                    score_str = f" (match: {edit.get('match_score', 0):.2%})" if edit.get('match_score') is not None else ""
                    
                    if edit['type'] == 'keep':
                        f.write(f"[ROW {edit['csv_position']}] KEPT UNCHANGED{score_str}\n")
                        f.write(f"  SRC: {edit['original_src']}\n")
                        f.write(f"  TGT: {edit['original_tgt']}\n")
                        f.write(f"{'-'*80}\n\n")
                    
                    elif edit['type'] == 'edit':
                        f.write(f"[ROW {edit['csv_position']}] EDIT{score_str}\n")
                        f.write(f"  Original SRC: {edit['original_src']}\n")
                        f.write(f"  New SRC:      {edit['new_src']}\n")
                        f.write(f"  Original TGT: {edit['original_tgt']}\n")
                        f.write(f"  New TGT:      {edit['new_tgt']}\n")
                        f.write(f"{'-'*80}\n\n")
                    
                    elif edit['type'] == 'split':
                        f.write(f"[ROW {edit['csv_position']}] SPLIT{score_str}\n")
                        f.write(f"  Original: {edit['original_src']}\n")
                        f.write(f"  {edit['new_src']}\n")
                        f.write(f"{'-'*80}\n\n")
                    
                    elif edit['type'] == 'merge':
                        f.write(f"[ROW {edit['csv_position']}] MERGE{score_str}\n")
                        f.write(f"  {edit['original_src']}\n")
                        f.write(f"  Merged to: {edit['new_src']}\n")
                        f.write(f"{'-'*80}\n\n")
                    
                    elif edit['type'] == 'delete':
                        score_str_del = f" (match: {edit.get('match_score', 0):.2%})" if edit.get('match_score') is not None else ""
                        f.write(f"[ROW {edit['csv_position']}] DELETE{score_str_del}\n")
                        f.write(f"  Deleted: {edit['original_src']}\n")
                        f.write(f"{'-'*80}\n\n")
        
        print(f"\n✓ Detailed edit log saved to: {log_path}")
    
    # Final summary
    print(f"\n{'═'*80}")
    print("FINAL SUMMARY")
    print(f"{'═'*80}")
    
    for corpus_name, stats in all_stats.items():
        print(f"\n{corpus_name}:")
        print(f"  Kept:    {stats['keep']}")
        print(f"  Edited:  {stats['edit']}")
        print(f"  Split:   {stats['split']}")
        print(f"  Merged:  {stats['merge']}")
        print(f"  Deleted: {stats['delete']}")
    
    print(f"\n✓ Updated CSV saved to: {output_path}")
    print(f"✓ All sentence indices have been automatically adjusted!")
    
    return df


def find_norm_files_in_directory(directory: str, exclude_csv: bool = True) -> List[str]:
    """
    Find all .norm files in directory, excluding CSV files.
    
    Args:
        directory: Directory to search
        exclude_csv: If True, skip .csv files (default: True)
    """
    norm_files = []
    
    for file in Path(directory).iterdir():
        if file.is_file():
            # Skip CSV files
            if exclude_csv and file.suffix.lower() == '.csv':
                continue
            
            # Include .norm files
            if file.suffix.lower() == '.norm':
                norm_files.append(str(file))
    
    return sorted(norm_files)


def batch_update_with_directory(csv_path: str, norm_dir: str,
                                 output_path: str = None,
                                 threshold: float = 0.85,
                                 log_edits: bool = True):
    """Update CSV from all NORM files in directory (excluding CSV files)."""
    
    # Find all .norm files, excluding CSV
    norm_files = find_norm_files_in_directory(norm_dir, exclude_csv=True)
    
    if not norm_files:
        print(f"ERROR: No .norm files found in {norm_dir}")
        print(f"\nLooked for files with .norm extension")
        print(f"CSV files are automatically excluded")
        return
    
    print(f"Found {len(norm_files)} NORM files in {norm_dir}:")
    for f in norm_files:
        print(f"  • {Path(f).name}")
    
    return batch_update_csv_smart(csv_path, norm_files, output_path, threshold, log_edits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch update CSV with automatic index adjustment (no metadata required)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Batch update command - ALL files in directory (MAIN COMMAND)
    batch_parser = subparsers.add_parser('batch-update',
                                           help='Update CSV from ALL .norm files in directory')
    batch_parser.add_argument('--directory', required=True,
                               help='Directory containing both CSV and .norm files (e.g., output/extraction)')
    batch_parser.add_argument('--csv-name', default='all_corpora.csv',
                               help='Name of CSV file in directory (default: all_corpora.csv)')
    batch_parser.add_argument('--output', default=None,
                               help='Output CSV path (default: overwrites original in same directory)')
    batch_parser.add_argument('--threshold', type=float, default=0.85,
                               help='Fuzzy match threshold (0.0-1.0, default: 0.85)')
    batch_parser.add_argument('--no-log', action='store_true',
                               help='Disable detailed edit log file')
    
    # Update command - specific files only (advanced)
    update_parser = subparsers.add_parser('update', 
                                          help='Update CSV from specific NORM files only')
    update_parser.add_argument('--csv-file', required=True, help='Path to CSV file')
    update_parser.add_argument('--norm-files', nargs='+', required=True,
                              help='List of specific NORM files to process')
    update_parser.add_argument('--output', default=None,
                              help='Output CSV path (overwrites if not provided)')
    update_parser.add_argument('--threshold', type=float, default=0.85,
                              help='Fuzzy match threshold (0.0-1.0, default: 0.85)')
    update_parser.add_argument('--no-log', action='store_true',
                               help='Disable detailed edit log file')
    
    args = parser.parse_args()
    
    if args.command == 'batch-update':
        # Process ALL .norm files in directory
        directory = Path(args.directory)
        
        if not directory.exists():
            print(f"ERROR: Directory not found: {args.directory}")
            exit(1)
        
        # Find CSV file
        csv_path = directory / args.csv_name
        if not csv_path.exists():
            print(f"ERROR: CSV file not found: {csv_path}")
            print(f"\nSearched for: {args.csv_name}")
            print(f"In directory: {directory}")
            
            # List available CSV files
            csv_files = list(directory.glob("*.csv"))
            if csv_files:
                print(f"\nAvailable CSV files:")
                for f in csv_files:
                    print(f"  • {f.name}")
                print(f"\nTry: --csv-name {csv_files[0].name}")
            exit(1)
        
        # Output path defaults to same directory
        if args.output is None:
            output_path = str(csv_path)
        else:
            output_path = args.output
        
        batch_update_with_directory(
            csv_path=str(csv_path),
            norm_dir=str(directory),
            output_path=output_path,
            threshold=args.threshold,
            log_edits=not args.no_log
        )
    
    elif args.command == 'update':
        # Process specific NORM files only
        if not Path(args.csv_file).exists():
            print(f"ERROR: CSV file not found: {args.csv_file}")
            exit(1)
        
        missing = [f for f in args.norm_files if not Path(f).exists()]
        if missing:
            print(f"ERROR: NORM files not found:")
            for f in missing:
                print(f"  • {f}")
            exit(1)
        
        print(f"\nProcessing {len(args.norm_files)} specific NORM file(s)...")
        
        batch_update_csv_smart(
            csv_path=args.csv_file,
            norm_files=args.norm_files,
            output_path=args.output,
            threshold=args.threshold,
            log_edits=not args.no_log
        )
    
    else:
        parser.print_help()
        print("\n" + "="*80)
        print("EXAMPLES")
        print("="*80)
        print("\n1. Update from specific NORM files:")
        print(f"   python {Path(__file__).name} batch-update \\")
        print("       --csv-file output/all_corpora.csv \\")
        print("       --norm-files output/LEONIDE_full.norm output/Kolipsi_1_L2_full.norm")
        print("\n2. Update from all NORM files in directory:")
        print(f"   python {Path(__file__).name} batch-update-dir \\")
        print("       --csv-file output/all_corpora.csv \\")
        print("       --norm-dir output/ \\")
        print("       --pattern '*_full.norm'")
        print("\n3. Adjust fuzzy matching threshold (for aggressive changes):")
        print(f"   python {Path(__file__).name} batch-update \\")
        print("       --csv-file output/all_corpora.csv \\")
        print("       --norm-files output/LEONIDE_full.norm \\")
        print("       --threshold 0.70")
        print("="*80 + "\n")