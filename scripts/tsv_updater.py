"""
Batch update TSV file from multiple edited NORM files with automatic index adjustment.

This script intelligently detects splits, merges, and deletions by comparing
NORM sentences with TSV content, WITHOUT requiring metadata markers.

Usage:
    python tsv_updater.py batch-update \
        --tsv-file output/all_corpora.tsv \
        --norm-files output/LEONIDE_full.norm output/Kolipsi_1_L2_full.norm
"""

import re
import argparse
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from configs import Paths, ExtractionParams
import glob
from difflib import SequenceMatcher

#cancellare?
def get_norm_path_for_corpus(corpus_name: str) -> str:
    """Construct NORM file path from corpus name using configs."""
    # Try common suffixes
    for suffix in ['_full', '_edited', '_with_meta', '']:
        norm_file = Paths.EXTRACT_DIR / f"{corpus_name}{suffix}.norm"
        if norm_file.exists():
            return str(norm_file)
    return None

def normalize_text(text: str) -> str:
    """Normalize text for comparison: strip, lowercase, collapse whitespace."""
    return ' '.join(text.strip().lower().split())

def texts_similar(text1: str, text2: str, threshold: float = 0.95) -> bool:
    """Check if two texts are similar using ratio comparison."""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if norm1 == norm2:
        return True
    
    # Use SequenceMatcher for fuzzy matching
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold
    
def parse_norm_file_simple(norm_path: str) -> List[Tuple[str, str, int, int]]:
    """
    Parse NORM file into list of (src_sentence, tgt_sentence, line_start, line_end) tuples.
    
    Args:
        norm_path: Path to NORM file
    
    Returns:
        List of (src, tgt, line_start, line_end) where line_end is the blank line
    """
    sentences = []
    current_src = []
    current_tgt = []
    sent_start_line = 1
    current_line = 1
    
    with open(norm_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            if line.startswith('#'):
                current_line += 1
                continue
            
            if not line.strip():
                # Blank line = sentence boundary
                if current_src or current_tgt:
                    src_sent = ' '.join(current_src).strip()
                    tgt_sent = ' '.join(current_tgt).strip()
                    
                    # *** ADD THESE TWO LINES ***
                    # Remove spaces before punctuation for correct German typography
                    src_sent = re.sub(r'\s+([.,!?;:)\]])', r'\1', src_sent)
                    tgt_sent = re.sub(r'\s+([.,!?;:)\]])', r'\1', tgt_sent)
                    
                    sent_end_line = current_line  # Blank line is the end
                    sentences.append((src_sent, tgt_sent, sent_start_line, sent_end_line))
                    current_src = []
                    current_tgt = []
                    sent_start_line = current_line + 1  # Next sentence starts after blank
                current_line += 1
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
            
            current_line += 1
    
    # Handle last sentence if file doesn't end with blank line
    if current_src or current_tgt:
        src_sent = ' '.join(current_src).strip()
        tgt_sent = ' '.join(current_tgt).strip()
        
        # *** ADD THESE TWO LINES ***
        # Remove spaces before punctuation for correct German typography
        src_sent = re.sub(r'\s+([.,!?;:)\]])', r'\1', src_sent)
        tgt_sent = re.sub(r'\s+([.,!?;:)\]])', r'\1', tgt_sent)
        
        sent_end_line = current_line  # Last line of file
        sentences.append((src_sent, tgt_sent, sent_start_line, sent_end_line))
    
    return sentences
    
def detect_operations(tsv_df: pd.DataFrame, 
                     norm_sentences: List[Tuple[str, str, int, int]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect operations by comparing TSV DataFrame with NORM sentences.
    
    Args:
        tsv_df: DataFrame rows for this corpus (contains all metadata)
        norm_sentences: List of (src, tgt, line_start, line_end) from NORM file
    
    Returns:
        operations: List of operation dicts
        edit_details: List of detailed edit information
    """
    operations = []
    edit_details = []
    
    print(f"\n  Analyzing differences (TSV: {len(tsv_df)}, NORM: {len(norm_sentences)})...")
    
    # ADD DIAGNOSTIC: Check first few mismatches
    mismatches = []
    for i in range(min(10, len(tsv_df), len(norm_sentences))):
        tsv_src = tsv_df.iloc[i]['src']
        norm_src = norm_sentences[i][0]  # First element is src
        if not texts_similar(tsv_src, norm_src):
            mismatches.append((i, tsv_src[:50], norm_src[:50]))
    
    if mismatches:
        print(f"  ⚠️  First {len(mismatches)} mismatches detected:")
        for idx, tsv, norm in mismatches[:3]:
            print(f"    Row {idx+1}:")
            print(f"      TSV:  '{tsv}...'")
            print(f"      NORM: '{norm}...'")

    tsv_i = 0
    norm_i = 0
    
    while tsv_i < len(tsv_df) or norm_i < len(norm_sentences):
        if tsv_i >= len(tsv_df):
            break
        
        if norm_i >= len(norm_sentences):
            # Remaining TSV sentences were deleted
            operations.append({
                'type': 'delete',
                'tsv_start': tsv_i,
                'tsv_end': len(tsv_df)
            })
            for i in range(tsv_i, len(tsv_df)):
                row = tsv_df.iloc[i]
                edit_details.append({
                    'type': 'delete',
                    'tsv_position': i + 1,
                    'original_src': row['src'],
                    'original_tgt': row['tgt'],
                    'new_src': '',
                    'new_tgt': '',
                    'xml_file': row['xml_file'],
                    'sent_num': row['sent_num']
                })
            break
        
        row = tsv_df.iloc[tsv_i]
        tsv_src = row['src']
        tsv_tgt = row['tgt']
        norm_src, norm_tgt, line_start, line_end = norm_sentences[norm_i]  # Unpack all 4 values
        
        # Check if sentences match exactly
        if texts_similar(tsv_src, norm_src) and texts_similar(tsv_tgt, norm_tgt):
            # Exact match - keep
            edit_details.append({
                'type': 'keep',
                'tsv_position': tsv_i + 1,
                'original_src': tsv_src,
                'original_tgt': tsv_tgt,
                'new_src': norm_src,
                'new_tgt': norm_tgt,
                'xml_file': row['xml_file'],
                'sent_num': row['sent_num']
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
        
        elif texts_similar(tsv_src, norm_src):
            # Source matches but target differs - simple edit
            edit_details.append({
                'type': 'edit',
                'tsv_position': tsv_i + 1,
                'original_src': tsv_src,
                'original_tgt': tsv_tgt,
                'new_src': norm_src,
                'new_tgt': norm_tgt,
                'xml_file': row['xml_file'],
                'sent_num': row['sent_num']
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
            # Check for split: TSV source contains multiple NORM sources
            combined_norm_src = norm_src
            split_count = 1
            temp_i = norm_i + 1
            
            # Try to match by combining consecutive NORM sentences
            while temp_i < len(norm_sentences) and split_count < 5:
                next_norm_src = norm_sentences[temp_i][0]  # Get src only
                combined_norm_src = combined_norm_src + " " + next_norm_src
                
                if texts_similar(tsv_src, combined_norm_src):
                    # Found a split: 1 TSV → multiple NORM
                    split_parts = norm_sentences[norm_i:temp_i + 1]
                    split_parts_display = '\n    '.join([f"[{j+1}] {part[0]}" for j, part in enumerate(split_parts)])
                    
                    edit_details.append({
                        'type': 'split',
                        'tsv_position': tsv_i + 1,
                        'original_src': tsv_src,
                        'original_tgt': tsv_tgt,
                        'new_src': f"Split into {len(split_parts)} parts:\n    {split_parts_display}",
                        'new_tgt': '',
                        'xml_file': row['xml_file'],
                        'sent_num': row['sent_num']
                    })
                    
                    operations.append({
                        'type': 'split',
                        'tsv_start': tsv_i,
                        'tsv_end': tsv_i + 1,
                        'norm_start': norm_i,
                        'norm_end': temp_i + 1
                    })
                    
                    tsv_i += 1
                    norm_i = temp_i + 1
                    break
                
                temp_i += 1
                split_count += 1
            else:
                # Check for merge: multiple TSV → 1 NORM
                combined_tsv_src = tsv_src
                merge_count = 1
                temp_i = tsv_i + 1
                
                while temp_i < len(tsv_df) and merge_count < 5:
                    next_row = tsv_df.iloc[temp_i]
                    next_tsv_src = next_row['src']
                    combined_tsv_src = combined_tsv_src + " " + next_tsv_src
                    
                    if texts_similar(combined_tsv_src, norm_src):
                        # Found a merge: multiple TSV → 1 NORM
                        merged_rows = [tsv_df.iloc[i] for i in range(tsv_i, temp_i + 1)]
                        merged_src_display = '\n    '.join([f"[{j+1}] {r['src']}" for j, r in enumerate(merged_rows)])
                        
                        edit_details.append({
                            'type': 'merge',
                            'tsv_position': f"{tsv_i + 1}-{temp_i + 1}",
                            'original_src': f"Merged from {merge_count} parts:\n    {merged_src_display}",
                            'original_tgt': '',
                            'new_src': norm_src,
                            'new_tgt': norm_tgt,
                            'xml_file': row['xml_file'],
                            'sent_num': row['sent_num']
                        })
                        
                        operations.append({
                            'type': 'merge',
                            'tsv_start': tsv_i,
                            'tsv_end': temp_i + 1,
                            'norm_start': norm_i,
                            'norm_end': norm_i + 1
                        })
                        
                        tsv_i = temp_i + 1
                        norm_i += 1
                        break
                    
                    temp_i += 1
                    merge_count += 1
                else:
                    # No split or merge detected - treat as simple edit
                    edit_details.append({
                        'type': 'edit',
                        'tsv_position': tsv_i + 1,
                        'original_src': tsv_src,
                        'original_tgt': tsv_tgt,
                        'new_src': norm_src,
                        'new_tgt': norm_tgt,
                        'xml_file': row['xml_file'],
                        'sent_num': row['sent_num']
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
                     norm_sentences: List[Tuple[str, str, int, int]],
                     operations: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply detected operations to update DataFrame with proper index adjustment and line numbers.
    """
    corpus_df = df[df['corpus'] == corpus_name].copy()
    
    # Keep non-corpus rows
    df_other = df[df['corpus'] != corpus_name].copy()
    
    # Build new rows based on operations
    new_rows = []
    edit_details = []
    
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
        norm_start = op.get('norm_start', 0)
        norm_end = op.get('norm_end', 0)
        
        if op_type == 'delete':
            # Record deletions
            for i in range(tsv_start, tsv_end):
                row = corpus_df.iloc[i]
                edit_details.append({
                    'type': 'delete',
                    'tsv_position': i + 1,
                    'original_src': row['src'],
                    'original_tgt': row['tgt'],
                    'new_src': '',
                    'new_tgt': '',
                    'xml_file': row['xml_file'],
                    'sent_num': row['sent_num']
                })
            stats['delete'] += (tsv_end - tsv_start)
            continue

        # Get template row from TSV for metadata
        template_row = corpus_df.iloc[tsv_start].copy()
        
        # Get NORM sentences for this operation
        norm_sents = norm_sentences[norm_start:norm_end]
        
        if op_type == 'keep':
            # No change - update line numbers
            norm_src, norm_tgt, line_start, line_end = norm_sents[0]
            new_row = template_row.copy()
            new_row['src'] = norm_src
            new_row['tgt'] = norm_tgt
            new_row['corrected'] = (norm_src.strip() != norm_tgt.strip())
            new_row['line_start'] = line_start
            new_row['line_end'] = line_end
            new_rows.append(new_row)
            
            edit_details.append({
                'type': 'keep',
                'tsv_position': tsv_start + 1,
                'original_src': norm_src,
                'original_tgt': norm_tgt,
                'new_src': norm_src,
                'new_tgt': norm_tgt,
                'xml_file': template_row['xml_file'],
                'sent_num': template_row['sent_num']
            })
            stats['keep'] += 1
        
        elif op_type == 'edit':
            # Simple edit - update line numbers
            norm_src, norm_tgt, line_start, line_end = norm_sents[0]
            new_row = template_row.copy()
            new_row['src'] = norm_src
            new_row['tgt'] = norm_tgt
            new_row['corrected'] = (norm_src.strip() != norm_tgt.strip())
            new_row['line_start'] = line_start
            new_row['line_end'] = line_end
            new_rows.append(new_row)
            
            edit_details.append({
                'type': 'edit',
                'tsv_position': tsv_start + 1,
                'original_src': template_row['src'],
                'original_tgt': template_row['tgt'],
                'new_src': norm_src,
                'new_tgt': norm_tgt,
                'xml_file': template_row['xml_file'],
                'sent_num': template_row['sent_num']
            })
            stats['edit'] += 1
        
        elif op_type == 'split':
            # 1 TSV → multiple NORM: create multiple rows with their own line numbers
            split_parts_display = []
            for i, (norm_src, norm_tgt, line_start, line_end) in enumerate(norm_sents):
                new_row = template_row.copy()
                new_row['src'] = norm_src
                new_row['tgt'] = norm_tgt
                new_row['corrected'] = (norm_src.strip() != norm_tgt.strip())
                new_row['line_start'] = line_start
                new_row['line_end'] = line_end
                if i > 0:
                    new_row['sent_num'] = f"{template_row['sent_num']}.{i+1}"
                new_rows.append(new_row)
                split_parts_display.append(f"[{i+1}] {norm_src}")
            
            edit_details.append({
                'type': 'split',
                'tsv_position': tsv_start + 1,
                'original_src': template_row['src'],
                'original_tgt': template_row['tgt'],
                'new_src': f"Split into {len(norm_sents)} parts:\n    " + '\n    '.join(split_parts_display),
                'new_tgt': '',
                'xml_file': template_row['xml_file'],
                'sent_num': template_row['sent_num']
            })
            stats['split'] += 1
        
        elif op_type == 'merge':
            # Multiple TSV → 1 NORM: use first TSV row's metadata, NORM line numbers
            norm_src, norm_tgt, line_start, line_end = norm_sents[0]
            new_row = template_row.copy()
            new_row['src'] = norm_src
            new_row['tgt'] = norm_tgt
            new_row['corrected'] = (norm_src.strip() != norm_tgt.strip())
            new_row['line_start'] = line_start
            new_row['line_end'] = line_end
            new_rows.append(new_row)
            
            merged_rows = [corpus_df.iloc[i] for i in range(tsv_start, tsv_end)]
            merged_src_display = '\n    '.join([f"[{j+1}] {r['src']}" for j, r in enumerate(merged_rows)])
            
            edit_details.append({
                'type': 'merge',
                'tsv_position': f"{tsv_start + 1}-{tsv_end}",
                'original_src': f"Merged from {tsv_end - tsv_start} parts:\n    {merged_src_display}",
                'original_tgt': '',
                'new_src': norm_src,
                'new_tgt': norm_tgt,
                'xml_file': template_row['xml_file'],
                'sent_num': template_row['sent_num']
            })
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

# to remove:
def infer_corpus_name(norm_filename: str) -> str:
    """Infer corpus name from NORM filename."""
    name = Path(norm_filename).stem
    name = name.replace('_full', '').replace('_edited', '').replace('_with_meta', '')
    return name

def batch_update_tsv(tsv_path: str, norm_files: List[str], 
                     output_path: str = None,
                     log_edits: bool = True) -> pd.DataFrame:
    """
    Update TSV from multiple NORM files using TSV as metadata source.
    
    Args:
        tsv_path: Path to TSV file (contains all metadata)
        norm_files: List of NORM file paths
        output_path: Output TSV path (overwrites original if None)
        log_edits: If True, write detailed edit log to file
    """
    print("\n" + "="*80)
    print("BATCH UPDATE USING TSV METADATA")
    print("="*80)
    
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
        
        # Parse NORM file
        norm_sentences = parse_norm_file_simple(norm_file)
        print(f"  NORM sentences: {len(norm_sentences)}")
        
        # Get TSV rows for this corpus
        corpus_df = df[df['corpus'] == corpus_name].copy()
        print(f"  TSV rows:       {len(corpus_df)}")
        
        if len(corpus_df) == 0:
            print(f"  ⚠️  WARNING: No rows found for corpus '{corpus_name}'")
            print(f"  Available corpora: {df['corpus'].unique().tolist()}")
            continue
        
        # Detect operations
        operations, edit_details = detect_operations(corpus_df, norm_sentences)
        
        # Apply operations and update DataFrame
        df, stats, edit_details = apply_operations(df, corpus_name, norm_sentences, 
                                                    operations)
        
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
    
    # Convert line numbers to integers (handle NaN values)
    if 'line_start' in df.columns:
        df['line_start'] = df['line_start'].fillna(0).astype('Int64')  # Use nullable Int64
    if 'line_end' in df.columns:
        df['line_end'] = df['line_end'].fillna(0).astype('Int64')  # Use nullable Int64
    
    df.to_csv(output_path, index=False, encoding='utf-8', sep='\t')
    print(f"\n✅ Updated TSV saved: {output_path}")
    
    # Write detailed edit log
    if log_edits:
        log_path = output_path.replace('.tsv', '_edit_log.txt')
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
        
        print(f"📝 Edit log saved: {log_path}")
    
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
    
    return batch_update_tsv(tsv_path, norm_files, output_path, log_edits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch update TSV with automatic index adjustment (no metadata required)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Batch update command - ALL files in directory (MAIN COMMAND)
    batch_parser = subparsers.add_parser('batch-update',
                                        help='Update TSV from ALL .norm files in directory')
    batch_parser.add_argument('--directory', 
                            default=str(Paths.EXTRACT_DIR),  # ADD THIS
                            help=f'Directory with TSV and .norm files (default: {Paths.EXTRACT_DIR})')
    batch_parser.add_argument('--tsv-name', default='all_corpora.tsv',
                            help='Name of TSV file in directory (default: all_corpora.tsv)')
    batch_parser.add_argument('--output', default=None,
                               help='Output TSV path (default: overwrites original in same directory)')
    batch_parser.add_argument('--no-log', action='store_true',
                               help='Disable detailed edit log file')
    
    # Update command - specific files only (advanced)
    update_parser = subparsers.add_parser('update', 
                                      help='Update TSV from specific corpora by name')
    update_parser.add_argument('--tsv-file', 
                                default=str(Paths.EXTRACT_DIR),
                                help=f'Path to TSV file (default: {Paths.EXTRACT_DIR}/all_corpora.tsv)')
    update_parser.add_argument('--corpora', nargs='*',  # CHANGED from --norm-files
                                default=None,
                                help='Corpus names (e.g., LEONIDE Kolipsi_1_L2). Leave empty to process all.')
    update_parser.add_argument('--output', default=None,
                            help='Output TSV path (overwrites if not provided)')
    update_parser.add_argument('--no-log', action='store_true',
                            help='Disable detailed edit log file')
    
    args = parser.parse_args()
    
    if args.command == 'batch-update':
        directory = Path(args.directory) if args.directory else Paths.EXTRACT_DIR
        
        if not directory.exists():
            print(f"ERROR: Directory not found: {args.directory}")
            exit(1)
        
        # Find TSV file
        tsv_path = directory / args.tsv_name
        if not tsv_path.exists():
            print(f"ERROR: TSV file not found: {tsv_path}")
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
        
        batch_update_tsv(
            tsv_path=str(tsv_path),
            norm_files=norm_files,
            output_path=output_path,
            log_edits=not args.no_log
        )


    elif args.command == 'update':
        tsv_file = args.tsv_file
        
        # Convert corpus names to NORM file paths
        if not args.corpora:
            # Auto-discover all NORM files in EXTRACT_DIR
            norm_files = find_norm_files_in_directory(str(Paths.EXTRACT_DIR), exclude_tsv=True)
            if not norm_files:
                print(f"ERROR: No .norm files found in {Paths.EXTRACT_DIR}")
                exit(1)
            print(f"Auto-discovered {len(norm_files)} NORM files")
        else:
            # Get NORM paths for specified corpora
            norm_files = []
            missing = []
            
            for corpus_name in args.corpora:
                norm_path = get_norm_path_for_corpus(corpus_name)
                if norm_path:
                    norm_files.append(norm_path)
                    print(f"✓ {corpus_name} → {Path(norm_path).name}")
                else:
                    missing.append(corpus_name)
            
            if missing:
                print(f"\n❌ ERROR: NORM files not found for:")
                for corpus in missing:
                    print(f"  • {corpus}")
                print(f"\nSearched in: {Paths.EXTRACT_DIR}")
                exit(1)
        
        if not Path(tsv_file).exists():
            print(f"ERROR: TSV file not found: {tsv_file}")
            exit(1)
        
        batch_update_tsv(
            tsv_path=tsv_file,
            norm_files=norm_files,
            output_path=args.output,
            log_edits=not args.no_log
        )
    
    else:
        print("\n1. Update from all NORM files in directory:")
        print(f"   python {Path(__file__).name} batch-update \\")
        print("       --directory output/extraction \\")
        print("       --tsv-name all_corpora.tsv")
        print("\n2. Update from specific corpora:")
        print(f"   python {Path(__file__).name} update --corpora LEONIDE Kolipsi_1_L2")
        print("       --tsv-file output/all_corpora.tsv \\")
        print("       --norm-files output/LEONIDE_full.norm output/Kolipsi_1_L2_full.norm")
        print("="*80 + "\n")