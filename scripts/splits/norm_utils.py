"""
NORM file creation utilities.
Generates verticalized .norm files for train, dev, and test splits.
"""

import os
import pandas as pd
from typing import Dict, List, Optional


def create_norm_files(
    output_dir: str,
    tsv_path: str,
    extract_dir: Optional[str] = None,
    splits: Optional[List[str]] = None
) -> None:
    """
    Generate verticalized .norm files for train, dev, and test splits using TSV metadata.
    
    Each .norm file contains word-level alignments extracted from the original corpus
    NORM files, preserving the vertical format with blank line separators.
    
    Args:
        output_dir: Directory containing the split indices files
        tsv_path: Path to the corpus TSV (contains all metadata)
        extract_dir: Directory containing the original corpus .norm files (required)
        splits: List of split names to process (default: ['train', 'dev', 'test'])
    """
    if splits is None:
        splits = ['train', 'dev', 'test']
    
    if extract_dir is None:
        raise ValueError("extract_dir parameter is required - must specify directory with corpus .norm files")
    
    print("\n" + "=" * 80)
    print("GENERATING .norm FILES FOR SPLITS")
    print("=" * 80)
    
    # Load the full dataframe (this IS our metadata)
    df = pd.read_csv(tsv_path, encoding="utf-8", sep="\t", on_bad_lines='warn')
    print(f"✓ Loaded corpus TSV with {len(df)} sentences as metadata source")
    
    # Process each split
    for split_name in splits:
        indices_file = os.path.join(output_dir, f"{split_name}_indices.tsv")
        
        if not os.path.exists(indices_file):
            print(f"\n⚠️  Skipping {split_name}: indices file not found")
            continue
        
        # Load indices from TSV
        df_indices = pd.read_csv(indices_file, sep='\t', encoding='utf-8')
        indices = df_indices['DF_INDEX'].tolist()
        
        print(f"\n--- Processing {split_name} set ({len(indices)} sentences) ---")

        # Output file (combined source and target in same file)
        norm_output = os.path.join(output_dir, f"{split_name}.norm")
        
        # Cache for loaded .norm files
        norm_cache = {}
        
        output_lines = []
        sentences_processed = 0
        
        for idx_position, df_index in enumerate(indices):
            if (idx_position + 1) % 100 == 0:
                print(f"  Processed {idx_position + 1}/{len(indices)} sentences...")
            
            # Get metadata from dataframe
            row = df.loc[df_index]
            corpus_name = row['corpus']
            xml_file = row['xml_file']
            sent_num = row['sent_num']
            src_sentence = row['src']
            tgt_sentence = row['tgt']
            line_start = int(row['line_start'])
            line_end = int(row['line_end'])
            
            # Load .norm file if not cached
            norm_file_path = os.path.join(extract_dir, f"{corpus_name}.norm")
            
            if corpus_name not in norm_cache:
                if not os.path.exists(norm_file_path):
                    print(f"\n⚠️  Error: {norm_file_path} not found")
                    continue
                
                with open(norm_file_path, 'r', encoding='utf-8') as f:
                    norm_cache[corpus_name] = [line.rstrip('\n') for line in f]
            
            norm_lines = norm_cache[corpus_name]
            
            # Convert from 1-indexed file line numbers to 0-indexed array indices
            sentence_lines = norm_lines[line_start - 1:line_end]
            
            # Preserve all lines including blank lines
            # Add all lines from sentence_lines as-is (includes blank separator at end)
            output_lines.extend(sentence_lines)
            
            sentences_processed += 1
        
        # Write output file
        with open(norm_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
            # Add final newline
            if output_lines:
                f.write('\n')
        
        print(f"✓ Saved {split_name}.norm: {norm_output}")
        print(f"  ({sentences_processed} sentences, {len(output_lines)} lines)")
    
    print("\n✅ All .norm files generated successfully!")


def regenerate_splits_from_indices(
    output_dir: str,
    tsv_path: str,
    paths_config: Optional[Dict] = None
) -> None:
    """
    Regenerate .src, .tgt files from existing indices WITHOUT creating new splits.
    Preserves the existing random splits by reading from indices files.
    
    Args:
        output_dir: Directory containing the split indices files (fallback)
        tsv_path: Path to the corpus TSV (source of truth for sentences)
        paths_config: Optional dict with path mappings (TEST_SRC, TRAIN_SRC, etc.)
    """
    print("\n" + "=" * 80)
    print("REGENERATING SPLIT FILES FROM EXISTING INDICES")
    print("=" * 80)
    print("⚠️  This will OVERWRITE .src and .tgt files but keep the same splits")
    
    # Import here to avoid circular dependency
    from .file_utils import save_splits
    
    # Load the full corpus TSV
    df = pd.read_csv(tsv_path, encoding="utf-8", sep="\t", on_bad_lines='warn')
    print(f"✓ Loaded corpus TSV with {len(df)} sentences\n")
    
    # Process each split
    for split_name in ['test', 'train', 'dev']:
        # Get source file path from config or default
        if paths_config:
            if split_name == "test":
                src_file = paths_config.get('TEST_SRC')
            elif split_name == "train":
                src_file = paths_config.get('TRAIN_SRC')
            elif split_name == "dev":
                src_file = paths_config.get('DEV_SRC')
            else:
                src_file = None
        else:
            src_file = None
        
        if not src_file:
            src_file = os.path.join(output_dir, f"{split_name}.src")
        
        # Derive indices file location from .src file directory
        src_dir = os.path.dirname(src_file)
        indices_file = os.path.join(src_dir, f"{split_name}_indices.tsv")
        
        # Fallback to output_dir if not found
        if not os.path.exists(indices_file):
            indices_file = os.path.join(output_dir, f"{split_name}_indices.tsv")
        
        if not os.path.exists(indices_file):
            print(f"⚠️  Skipping {split_name}: indices file not found")
            print(f"     Looked in: {src_dir}")
            print(f"     And in: {output_dir}")
            continue
        
        print(f"--- Regenerating {split_name} set ---")
        print(f"  Reading indices from: {indices_file}")
        
        # Load indices from TSV
        df_indices = pd.read_csv(indices_file, sep='\t', encoding='utf-8')
        indices = df_indices['DF_INDEX'].tolist()
        
        # Get the actual data for these indices from main TSV
        df_split = df.loc[indices].copy()
        
        print(f"  Loaded {len(indices)} indices")
        print(f"  Extracting sentences from main TSV...")
        
        # Use save_splits to write .src and .tgt files
        save_splits(df_split, split_name, output_dir, indices, df, paths_config)
        
        print(f"  ✓ Regenerated {split_name}.src and {split_name}.tgt\n")
    
    print("✅ All split files regenerated successfully!")
    print("   The splits remain identical (same indices used)")