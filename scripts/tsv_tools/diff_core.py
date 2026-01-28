"""
Core diff and alignment logic for comparing TSV and NORM files.
Detects splits, merges, edits, and deletions.
"""

import pandas as pd
from difflib import SequenceMatcher
from typing import List, Tuple, Dict

from .norm_parser import normalize_text


def texts_similar(text1: str, text2: str, threshold: float = 0.95) -> bool:
    """
    Check if two texts are similar using ratio comparison.
    
    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        True if texts are similar enough
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if norm1 == norm2:
        return True
    
    # Use SequenceMatcher for fuzzy matching
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


def find_next_match(tsv_df, tsv_i, norm_sentences, norm_i, lookahead=10):
    """
    Look ahead to find the next matching pair to re-sync alignment.
    
    Args:
        tsv_df: TSV DataFrame
        tsv_i: Current TSV index
        norm_sentences: NORM sentences list
        norm_i: Current NORM index
        lookahead: How many sentences ahead to search
        
    Returns:
        Tuple of (tsv_offset, norm_offset) or (None, None) if no match found
    """
    # Search within lookahead window
    for t_offset in range(lookahead):
        for n_offset in range(lookahead):
            t_idx = tsv_i + t_offset
            n_idx = norm_i + n_offset
            
            if t_idx >= len(tsv_df) or n_idx >= len(norm_sentences):
                continue
            
            tsv_src = tsv_df.iloc[t_idx]['src']
            norm_src = norm_sentences[n_idx][0]
            
            if texts_similar(tsv_src, norm_src):
                return (t_offset, n_offset)
    
    return (None, None)


def detect_operations(
    tsv_df: pd.DataFrame,
    norm_sentences: List[Tuple[str, str, int, int]]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect operations by comparing TSV DataFrame with NORM sentences.
    
    Args:
        tsv_df: DataFrame rows for this corpus (contains all metadata)
        norm_sentences: List of (src, tgt, line_start, line_end) from NORM file
    
    Returns:
        Tuple of (operations, edit_details)
        - operations: List of operation dicts describing changes
        - edit_details: List of detailed edit information
    """
    operations = []
    edit_details = []
    
    print(f"\n  Analyzing differences (TSV: {len(tsv_df)}, NORM: {len(norm_sentences)})...")
    
    # Diagnostic: Check first few mismatches
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
        norm_src, norm_tgt, line_start, line_end = norm_sentences[norm_i]
        
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
        
        elif texts_similar(tsv_src, norm_src, threshold=0.85):
            # Source is similar but not exact - likely has word insertions/deletions
            # Treat as edit
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
            # Try to detect split/merge
            # Check if next TSV matches current NORM (possible merge: 2 TSV -> 1 NORM)
            if tsv_i + 1 < len(tsv_df):
                next_row = tsv_df.iloc[tsv_i + 1]
                combined_src = tsv_src + " " + next_row['src']
                
                if texts_similar(combined_src, norm_src):
                    # Merge detected - 2 TSV -> 1 NORM
                    edit_details.append({
                        'type': 'merge',
                        'tsv_position': tsv_i + 1,
                        'original_src': combined_src,
                        'original_tgt': tsv_tgt + " " + next_row['tgt'],
                        'new_src': norm_src,
                        'new_tgt': norm_tgt,
                        'xml_file': row['xml_file'],
                        'sent_num': row['sent_num']
                    })
                    operations.append({
                        'type': 'merge',
                        'tsv_start': tsv_i,
                        'tsv_end': tsv_i + 2,
                        'norm_start': norm_i,
                        'norm_end': norm_i + 1
                    })
                    tsv_i += 2
                    norm_i += 1
                    continue
            
            # Check if next NORM matches current TSV (possible split: 1 TSV -> 2 NORM)
            if norm_i + 1 < len(norm_sentences):
                next_norm = norm_sentences[norm_i + 1]
                combined_norm = norm_src + " " + next_norm[0]
                
                if texts_similar(tsv_src, combined_norm):
                    # Split detected - 1 TSV -> 2 NORM
                    edit_details.append({
                        'type': 'split',
                        'tsv_position': tsv_i + 1,
                        'original_src': tsv_src,
                        'original_tgt': tsv_tgt,
                        'new_src': combined_norm,
                        'new_tgt': norm_tgt + " " + next_norm[1],
                        'xml_file': row['xml_file'],
                        'sent_num': row['sent_num']
                    })
                    operations.append({
                        'type': 'split',
                        'tsv_start': tsv_i,
                        'tsv_end': tsv_i + 1,
                        'norm_start': norm_i,
                        'norm_end': norm_i + 2
                    })
                    tsv_i += 1
                    norm_i += 2
                    continue
            
            # NEW: Try to re-sync by looking ahead
            t_offset, n_offset = find_next_match(tsv_df, tsv_i, norm_sentences, norm_i, lookahead=5)
            
            if t_offset is not None and n_offset is not None:
                # Found a match ahead - mark intervening sentences as mismatched
                print(f"  🔄 Re-syncing at TSV {tsv_i + t_offset}, NORM {norm_i + n_offset}")
                
                # Mark TSV rows before match as deleted
                for t in range(t_offset):
                    if tsv_i + t < len(tsv_df):
                        del_row = tsv_df.iloc[tsv_i + t]
                        edit_details.append({
                            'type': 'delete',
                            'tsv_position': tsv_i + t + 1,
                            'original_src': del_row['src'],
                            'original_tgt': del_row['tgt'],
                            'new_src': '',
                            'new_tgt': '',
                            'xml_file': del_row['xml_file'],
                            'sent_num': del_row['sent_num']
                        })
                        operations.append({
                            'type': 'delete',
                            'tsv_start': tsv_i + t,
                            'tsv_end': tsv_i + t + 1
                        })
                
                # Skip ahead to the matching position
                tsv_i += t_offset
                norm_i += n_offset

            else:
                # No match found nearby - treat as insertion/deletion
                # Check if this is likely an insertion (NORM has extra content)
                # by checking if next TSV matches current NORM
                is_insertion = False
                if tsv_i + 1 < len(tsv_df):
                    next_tsv_src = tsv_df.iloc[tsv_i + 1]['src']
                    if texts_similar(next_tsv_src, norm_src):
                        # TSV[i+1] matches NORM[i] - NORM has insertion before this
                        is_insertion = True
                
                if is_insertion:
                    # Don't mark as deletion - just advance norm_i to skip the insertion
                    norm_i += 1
                else:
                    # Mark as deletion and advance both
                    edit_details.append({
                        'type': 'delete',
                        'tsv_position': tsv_i + 1,
                        'original_src': tsv_src,
                        'original_tgt': tsv_tgt,
                        'new_src': '',
                        'new_tgt': '',
                        'xml_file': row['xml_file'],
                        'sent_num': row['sent_num']
                    })
                    operations.append({
                        'type': 'delete',
                        'tsv_start': tsv_i,
                        'tsv_end': tsv_i + 1
                    })
                    tsv_i += 1
                    norm_i += 1

    
    return operations, edit_details


def calculate_operation_stats(edit_details: List[Dict]) -> Dict[str, int]:
    """
    Calculate statistics from edit details.
    
    Args:
        edit_details: List of edit operation dictionaries
        
    Returns:
        Dictionary with counts for each operation type
    """
    stats = {
        'keep': 0,
        'edit': 0,
        'split': 0,
        'merge': 0,
        'delete': 0
    }
    
    for detail in edit_details:
        op_type = detail['type']
        if op_type in stats:
            stats[op_type] += 1
    
    return stats