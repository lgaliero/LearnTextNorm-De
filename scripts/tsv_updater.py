"""
Batch update TSV file from multiple edited NORM files with automatic index adjustment.

This script intelligently detects splits, merges, and deletions by comparing
NORM sentences with TSV content, WITHOUT requiring metadata markers.

Usage:
    python update_tsv_from_norm.py batch-update \
        --tsv-file output/all_corpora.tsv \
        --norm-files output/LEONIDE_full.norm output/Kolipsi_1_L2_full.norm
"""

import re
import argparse
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import glob
from difflib import SequenceMatcher

def parse_metadata_file(meta_path: str) -> Dict[str, Dict]:
    """
    Parse .norm.meta.txt file to get exact line mappings.
    
    Returns:
        Dict mapping corpus_name to list of metadata dicts with keys:
        - xml_file, sent_num, line_start, line_end, src, tgt
    """
    metadata = {}
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 7:
                continue
            
            corpus, xml_file, sent_num, line_start, line_end, src, tgt = parts
            
            if corpus not in metadata:
                metadata[corpus] = []
            
            metadata[corpus].append({
                'xml_file': xml_file,
                'sent_num': int(sent_num),
                'line_start': int(line_start),
                'line_end': int(line_end),
                'src': src,
                'tgt': tgt
            })
    
    return metadata

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

def parse_norm_file_with_metadata(norm_path: str, meta_path: str, corpus_name: str) -> List[Tuple[str, str, Dict]]:
    """
    Parse NORM file using metadata for exact sentence boundaries.
    
    Returns:
        List of (src_sentence, tgt_sentence, metadata_dict) tuples
    """
    # Parse metadata
    all_metadata = parse_metadata_file(meta_path)
    
    if corpus_name not in all_metadata:
        raise ValueError(f"Corpus '{corpus_name}' not found in metadata file")
    
    corpus_metadata = all_metadata[corpus_name]
    
    # Read entire NORM file into lines
    with open(norm_path, 'r', encoding='utf-8') as f:
        norm_lines = [line.rstrip('\n') for line in f]
    
    sentences = []
    
    for meta in corpus_metadata:
        line_start = meta['line_start'] - 1  # Convert to 0-indexed
        line_end = meta['line_end'] - 1      # Convert to 0-indexed
        
        # Extract lines for this sentence
        sent_lines = norm_lines[line_start:line_end]
        
        current_src = []
        current_tgt = []
        
        for line in sent_lines:
            if line.startswith('#') or not line.strip():
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
        
        src_sent = ' '.join(current_src).strip()
        tgt_sent = ' '.join(current_tgt).strip()
        
        sentences.append((src_sent, tgt_sent, meta))
    
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


def detect_operations_with_metadata(tsv_sentences: List[Tuple[str, str]], 
                                     norm_sentences: List[Tuple[str, str, Dict]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect operations by comparing TSV with NORM using metadata for exact alignment.
    
    Args:
        tsv_sentences: List of (src, tgt) from TSV
        norm_sentences: List of (src, tgt, metadata) from NORM with metadata
    
    Returns:
        operations: List of operation dicts
        edit_details: List of detailed edit information
    """
    operations = []
    edit_details = []
    
    print(f"\n  Analyzing differences using metadata (TSV: {len(tsv_sentences)}, NORM: {len(norm_sentences)})...")
    
    # Build mapping of (xml_file, sent_num) to TSV index
    tsv_map = {}
    for tsv_i, (src, tgt) in enumerate(tsv_sentences):
        # We need to get xml_file and sent_num from the TSV row
        # This will be passed from the caller
        pass
    
    # Since we need TSV metadata, we'll compare directly by position
    # and use the NORM metadata to detect splits/merges
    
    tsv_i = 0
    norm_i = 0
    
    while tsv_i < len(tsv_sentences) or norm_i < len(norm_sentences):
        if tsv_i >= len(tsv_sentences):
            # Remaining NORM sentences are additions (shouldn't happen)
            for i in range(norm_i, len(norm_sentences)):
                norm_src, norm_tgt, meta = norm_sentences[i]
                edit_details.append({
                    'type': 'addition',
                    'tsv_position': tsv_i + 1,
                    'original_src': '',
                    'original_tgt': '',
                    'new_src': norm_src,
                    'new_tgt': norm_tgt,
                    'xml_file': meta['xml_file'],
                    'sent_num': meta['sent_num']
                })
            break
        
        if norm_i >= len(norm_sentences):
            # Remaining TSV sentences were deleted
            for i in range(tsv_i, len(tsv_sentences)):
                tsv_src, tsv_tgt = tsv_sentences[i]
                edit_details.append({
                    'type': 'delete',
                    'tsv_position': i + 1,
                    'original_src': tsv_src,
                    'original_tgt': tsv_tgt,
                    'new_src': '',
                    'new_tgt': ''
                })
            operations.append({
                'type': 'delete',
                'tsv_start': tsv_i,
                'tsv_end': len(tsv_sentences),
                'norm_start': norm_i,
                'norm_end': norm_i
            })
            break
        
        tsv_src, tsv_tgt = tsv_sentences[tsv_i]
        norm_src, norm_tgt, meta = norm_sentences[norm_i]
        
        # Check if sentences match exactly
        if tsv_src == norm_src and tsv_tgt == norm_tgt:
            # Exact match - keep
            edit_details.append({
                'type': 'keep',
                'tsv_position': tsv_i + 1,
                'original_src': tsv_src,
                'original_tgt': tsv_tgt,
                'new_src': norm_src,
                'new_tgt': norm_tgt,
                'xml_file': meta['xml_file'],
                'sent_num': meta['sent_num']
            })
            operations.append({
                'type': 'keep',
                'tsv_start': tsv_i,
                'tsv_end': tsv_i + 1,
                'norm_start': norm_i,
                'norm_end': norm_i + 1
            })
            tsv_i += 1
            norm_i += 1
        
        elif tsv_src.strip() == norm_src.strip():
            # Source matches but target differs - simple edit
            edit_details.append({
                'type': 'edit',
                'tsv_position': tsv_i + 1,
                'original_src': tsv_src,
                'original_tgt': tsv_tgt,
                'new_src': norm_src,
                'new_tgt': norm_tgt,
                'xml_file': meta['xml_file'],
                'sent_num': meta['sent_num']
            })
            operations.append({
                'type': 'edit',
                'tsv_start': tsv_i,
                'tsv_end': tsv_i + 1,
                'norm_start': norm_i,
                'norm_end': norm_i + 1
            })
            tsv_i += 1
            norm_i += 1
        
        else:
            # Check for split: look for fractional sent_num (e.g., 5.2, 5.3)
            # This indicates sentence was split in NORM
            current_sent_num = meta['sent_num']
            
            # Check if next NORM sentence has same integer part but fractional
            split_detected = False
            if norm_i + 1 < len(norm_sentences):
                next_meta = norm_sentences[norm_i + 1][2]
                # Check if sent_num suggests a split (same file, next is X.2, X.3, etc.)
                if (meta['xml_file'] == next_meta['xml_file'] and 
                    isinstance(next_meta['sent_num'], str) and '.' in str(next_meta['sent_num'])):
                    
                    # Collect all parts of the split
                    split_parts = [(norm_src, norm_tgt, meta)]
                    temp_i = norm_i + 1
                    
                    base_num = str(current_sent_num).split('.')[0]
                    
                    while temp_i < len(norm_sentences):
                        temp_src, temp_tgt, temp_meta = norm_sentences[temp_i]
                        temp_num_str = str(temp_meta['sent_num'])
                        
                        if ('.' in temp_num_str and 
                            temp_num_str.startswith(base_num + '.') and
                            temp_meta['xml_file'] == meta['xml_file']):
                            split_parts.append((temp_src, temp_tgt, temp_meta))
                            temp_i += 1
                        else:
                            break
                    
                    # This is a split
                    split_parts_display = '\n    '.join([f"[{j+1}] {part[0]}" for j, part in enumerate(split_parts)])
                    
                    edit_details.append({
                        'type': 'split',
                        'tsv_position': tsv_i + 1,
                        'original_src': tsv_src,
                        'original_tgt': tsv_tgt,
                        'new_src': f"Split into {len(split_parts)} parts:\n    {split_parts_display}",
                        'new_tgt': '',
                        'xml_file': meta['xml_file'],
                        'sent_num': current_sent_num
                    })
                    
                    operations.append({
                        'type': 'split',
                        'tsv_start': tsv_i,
                        'tsv_end': tsv_i + 1,
                        'norm_start': norm_i,
                        'norm_end': temp_i
                    })
                    
                    tsv_i += 1
                    norm_i = temp_i
                    split_detected = True
            
            if split_detected:
                continue
            
            # Otherwise, treat as edit
            edit_details.append({
                'type': 'edit',
                'tsv_position': tsv_i + 1,
                'original_src': tsv_src,
                'original_tgt': tsv_tgt,
                'new_src': norm_src,
                'new_tgt': norm_tgt,
                'xml_file': meta['xml_file'],
                'sent_num': meta['sent_num']
            })
            
            operations.append({
                'type': 'edit',
                'tsv_start': tsv_i,
                'tsv_end': tsv_i + 1,
                'norm_start': norm_i,
                'norm_end': norm_i + 1
            })
            tsv_i += 1
            norm_i += 1
    
    return operations, edit_details

def apply_operations(df: pd.DataFrame, corpus_name: str, 
                     tsv_sentences: List[Tuple[str, str]],
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
        tsv_start = op['tsv_start']
        tsv_end = op['tsv_end']
        norm_start = op['norm_start']
        norm_end = op['norm_end']
        
        if op_type == 'delete':
            # Skip these TSV rows
            stats['delete'] += (tsv_end - tsv_start)
            continue
        
        # Get template row from TSV for metadata
        if tsv_start < len(corpus_indices):
            template_idx = corpus_indices[tsv_start]
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
            # 1 TSV → multiple NORM: create multiple rows
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
            # Multiple TSV → 1 NORM: use first TSV row's metadata
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
    
    # Preserve original corpus order from input TSV
    corpus_order = {corpus: i for i, corpus in enumerate(df['corpus'].unique())}
    df_updated['_corpus_order'] = df_updated['corpus'].map(corpus_order)
    df_updated = df_updated.sort_values(['_corpus_order', 'xml_file', 'sent_num']).drop('_corpus_order', axis=1).reset_index(drop=True)
        
    return df_updated, stats, edit_details


def infer_corpus_name(norm_filename: str) -> str:
    """Infer corpus name from NORM filename."""
    name = Path(norm_filename).stem
    name = name.replace('_full', '').replace('_edited', '').replace('_with_meta', '')
    return name


def batch_update_tsv_smart(tsv_path: str, norm_files: List[str], 
                            metadata_file: str,
                            output_path: str = None,
                            log_edits: bool = True) -> pd.DataFrame:
    """
    Update TSV from multiple NORM files using metadata file for exact alignment.
    
    Args:
        tsv_path: Path to TSV file
        norm_files: List of NORM file paths
        metadata_file: Path to all_corpora.norm.meta.txt
        output_path: Output TSV path (overwrites original if None)
        log_edits: If True, write detailed edit log to file
    """
    print("\n" + "="*80)
    print("BATCH UPDATE WITH METADATA-BASED ALIGNMENT")
    print("="*80)
    print(f"\nUsing metadata file: {metadata_file}")
    
    # Load TSV
    print(f"\nLoading TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, encoding='utf-8', sep='\t')
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
        
        # Parse NORM file with metadata
        try:
            norm_sentences = parse_norm_file_with_metadata(norm_file, metadata_file, corpus_name)
        except ValueError as e:
            print(f"  ⚠️  ERROR: {e}")
            continue
        
        print(f"  NORM sentences: {len(norm_sentences)}")
        
        # Get TSV sentences for this corpus
        corpus_df = df[df['corpus'] == corpus_name].copy()
        print(f"  TSV rows:       {len(corpus_df)}")
        
        if len(corpus_df) == 0:
            print(f"  ⚠️  WARNING: No rows found for corpus '{corpus_name}'")
            print(f"  Available corpora: {df['corpus'].unique().tolist()}")
            continue
        
        # Extract TSV sentences
        tsv_sentences = [(row['src'], row['tgt']) for _, row in corpus_df.iterrows()]
        
        # Detect operations using metadata
        operations, edit_details = detect_operations_with_metadata(tsv_sentences, norm_sentences)
        
        # Apply operations and update DataFrame
        df, stats, edit_details = apply_operations(df, corpus_name, tsv_sentences, 
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
    
    # Save updated TSV
    if output_path is None:
        output_path = tsv_path
    
    df.to_csv(output_path, index=False, encoding='utf-8', sep='\t')
    
    # Write detailed edit log (same as before)
    # ... [keep existing log writing code]
    
    return df

def find_norm_files_in_directory(directory: str, exclude_tsv: bool = True) -> List[str]:
    """
    Find all .norm files in directory, excluding TSV files.
    
    Args:
        directory: Directory to search
        exclude_tsv: If True, skip .tsv files (default: True)
    """
    norm_files = []
    
    for file in Path(directory).iterdir():
        if file.is_file():
            # Skip TSV files
            if exclude_tsv and file.suffix.lower() == '.tsv':
                continue
            
            # Include .norm files
            if file.suffix.lower() == '.norm':
                norm_files.append(str(file))
    
    return sorted(norm_files)


def batch_update_with_directory(tsv_path: str, norm_dir: str,
                                 output_path: str = None,
                                 threshold: float = 0.85,
                                 log_edits: bool = True):
    """Update TSV from all NORM files in directory (excluding TSV files)."""
    
    # Find all .norm files, excluding TSV
    norm_files = find_norm_files_in_directory(norm_dir, exclude_tsv=True)
    
    if not norm_files:
        print(f"ERROR: No .norm files found in {norm_dir}")
        print(f"\nLooked for files with .norm extension")
        print(f"TSV files are automatically excluded")
        return
    
    print(f"Found {len(norm_files)} NORM files in {norm_dir}:")
    for f in norm_files:
        print(f"  • {Path(f).name}")
    
    return batch_update_tsv_smart(tsv_path, norm_files, output_path, threshold, log_edits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch update TSV with automatic index adjustment (no metadata required)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Batch update command - ALL files in directory (MAIN COMMAND)
    batch_parser = subparsers.add_parser('batch-update',
                                           help='Update TSV from ALL .norm files in directory')
    batch_parser.add_argument('--directory', required=True,
                               help='Directory containing both TSV and .norm files (e.g., output/extraction)')
    batch_parser.add_argument('--tsv-name', default='all_corpora.tsv',
                               help='Name of TSV file in directory (default: all_corpora.tsv)')
    batch_parser.add_argument('--output', default=None,
                               help='Output TSV path (default: overwrites original in same directory)')
    batch_parser.add_argument('--threshold', type=float, default=0.85,
                               help='Fuzzy match threshold (0.0-1.0, default: 0.85)')
    batch_parser.add_argument('--no-log', action='store_true',
                               help='Disable detailed edit log file')
    
    # Update command - specific files only (advanced)
    update_parser = subparsers.add_parser('update', 
                                          help='Update TSV from specific NORM files only')
    update_parser.add_argument('--tsv-file', required=True, help='Path to TSV file')
    update_parser.add_argument('--norm-files', nargs='+', required=True,
                              help='List of specific NORM files to process')
    update_parser.add_argument('--output', default=None,
                              help='Output TSV path (overwrites if not provided)')
    update_parser.add_argument('--threshold', type=float, default=0.85,
                              help='Fuzzy match threshold (0.0-1.0, default: 0.85)')
    update_parser.add_argument('--no-log', action='store_true',
                               help='Disable detailed edit log file')
    
    args = parser.parse_args()
    
    if args.command == 'batch-update':
        directory = Path(args.directory)
        
        if not directory.exists():
            print(f"ERROR: Directory not found: {args.directory}")
            exit(1)
        
        # Find TSV file
        tsv_path = directory / args.tsv_name
        if not tsv_path.exists():
            print(f"ERROR: TSV file not found: {tsv_path}")
            exit(1)
        
        # Find metadata file
        meta_path = directory / args.metadata_name
        if not meta_path.exists():
            print(f"ERROR: Metadata file not found: {meta_path}")
            exit(1)
        
        # Find NORM files
        norm_files = find_norm_files_in_directory(str(directory), exclude_tsv=True)
        if not norm_files:
            print(f"ERROR: No .norm files found in {directory}")
            exit(1)
        
        # Output path
        if args.output is None:
            output_path = str(tsv_path)
        else:
            output_path = args.output
        
        batch_update_tsv_smart(
            tsv_path=str(tsv_path),
            norm_files=norm_files,
            metadata_file=str(meta_path),
            output_path=output_path,
            log_edits=not args.no_log
        )

    elif args.command == 'update':
        # Process specific NORM files only
        if not Path(args.tsv_file).exists():
            print(f"ERROR: TSV file not found: {args.tsv_file}")
            exit(1)
        
        missing = [f for f in args.norm_files if not Path(f).exists()]
        if missing:
            print(f"ERROR: NORM files not found:")
            for f in missing:
                print(f"  • {f}")
            exit(1)
        
        print(f"\nProcessing {len(args.norm_files)} specific NORM file(s)...")
        
        batch_update_tsv_smart(
            tsv_path=args.tsv_file,
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
        print("       --tsv-file output/all_corpora.tsv \\")
        print("       --norm-files output/LEONIDE_full.norm output/Kolipsi_1_L2_full.norm")
        print("\n2. Update from all NORM files in directory:")
        print(f"   python {Path(__file__).name} batch-update-dir \\")
        print("       --tsv-file output/all_corpora.tsv \\")
        print("       --norm-dir output/ \\")
        print("       --pattern '*_full.norm'")
        print("\n3. Adjust fuzzy matching threshold (for aggressive changes):")
        print(f"   python {Path(__file__).name} batch-update \\")
        print("       --tsv-file output/all_corpora.tsv \\")
        print("       --norm-files output/LEONIDE_full.norm \\")
        print("       --threshold 0.70")
        print("="*80 + "\n")