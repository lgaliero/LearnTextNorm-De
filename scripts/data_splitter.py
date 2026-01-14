#!/usr/bin/env python3
"""
Create stratified test/train/dev sets from the corpus.
Maintains proportions of: subcorpora, text types, and correction ratios.
Uses line-number-based random sampling for reproducibility.
Saves all set sentences and tracks indices to prevent overlap.
"""

import os
from configs import Paths, DataSplits
import pandas as pd
import random
from typing import Dict, Tuple, Optional, List, Set

def load_existing_indices(output_dir: str) -> set:
    """Load indices from existing test set to avoid overlap."""
    test_indices_file = os.path.join(output_dir, "test_indices.txt")
    if os.path.exists(test_indices_file):
        with open(test_indices_file, 'r') as f:
            return set(int(line.strip()) for line in f if line.strip())
    return set()

def save_split_files(df_split: pd.DataFrame, split_name: str, 
                     output_dir: str, indices: List[int]) -> None:
    """Save source, target, and indices files for a dataset split."""
    src_file = os.path.join(output_dir, f"{split_name}.src")
    tgt_file = os.path.join(output_dir, f"{split_name}.tgt")
    indices_file = os.path.join(output_dir, f"{split_name}_indices.txt")
    
    with open(src_file, 'w', encoding='utf-8') as f:
        for src in df_split['src']:
            f.write(f"{src}\n")
    
    with open(tgt_file, 'w', encoding='utf-8') as f:
        for tgt in df_split['tgt']:
            f.write(f"{tgt}\n")
    
    with open(indices_file, 'w', encoding='utf-8') as f:
        for idx in indices:
            f.write(f"{idx}\n")
    
    print(f"✓ Saved {split_name} source: {src_file}")
    print(f"✓ Saved {split_name} target: {tgt_file}")
    print(f"✓ Saved {split_name} indices: {indices_file}")

def stratified_sample_by_line_numbers(df: pd.DataFrame, sample_size: float, 
                                     split_name: str, 
                                     excluded_indices: Set[int] = None) -> Tuple[pd.DataFrame, List[int]]:
    """
    Perform stratified sampling using line numbers as the random pool.
    For each stratum, collect all valid line numbers and randomly select from them.
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
    
    sampled_indices = []
    sampling_details = []
    
    for strat_key, group in df.groupby('strat_key'):
        # Get line numbers (indices) for this stratum, excluding already used ones
        available_line_numbers = [idx for idx in group.index if idx not in excluded_indices]
        
        if len(available_line_numbers) == 0:
            continue
        
        n_total = len(available_line_numbers)
        n_sample = max(1, int(n_total * sample_size))
        
        # Ensure we don't sample more than available
        n_sample = min(n_sample, n_total)
        
        # Random sampling from available line numbers
        sampled_lines = random.sample(available_line_numbers, n_sample)
        sampled_indices.extend(sampled_lines)
        
        # Parse strat_key for reporting
        parts = strat_key.split('_')
        if len(parts) >= 3:
            corpus = '_'.join(parts[:-2])
            text_type = parts[-2]
            corrected = parts[-1]
        else:
            corpus, text_type, corrected = strat_key, "unknown", "unknown"
        
        sampling_details.append({
            'stratum': strat_key,
            'corpus': corpus,
            'text_type': text_type,
            'corrected': corrected,
            'available_lines': n_total,
            'sampled': n_sample,
            'percentage': f"{n_sample/n_total*100:.2f}%",
            'sample_line_numbers': sampled_lines[:5]  # Show first 5 as example
        })
    
    
    # Create sampled dataframe
    df_sampled = df.loc[sampled_indices].copy()
    df_sampled = df_sampled.drop(columns=['strat_key'])
    
    # Print sampling details
    print(f"\nTotal strata: {len(sampling_details)}")
    print(f"Total sampled: {len(sampled_indices):,} line numbers")
    print("\nSampling breakdown (showing first 10 strata):")
    for detail in sampling_details[:10]:
        print(f"  {detail['corpus']} | {detail['text_type']} | {detail['corrected']}: "
              f"{detail['sampled']}/{detail['available_lines']} lines "
              f"(e.g., lines {detail['sample_line_numbers']}...)")
    
    if len(sampling_details) > 10:
        print(f"  ... and {len(sampling_details) - 10} more strata")
    
    # Save detailed sampling report
    report_df = pd.DataFrame(sampling_details)
    # Remove the sample_line_numbers column for the CSV (too verbose)
    report_df_save = report_df.drop(columns=['sample_line_numbers'])
    
    return df_sampled, sampled_indices

def print_proportion_verification(df_full: pd.DataFrame, df_split: pd.DataFrame, 
                                 split_name: str) -> None:
    """Print verification of proportions maintained in split."""
    print("\n" + "=" * 80)
    print(f"PROPORTION VERIFICATION - {split_name.upper()}")
    print("=" * 80)
    
    print("\n--- By Corpus ---")
    orig_corpus = df_full.groupby('corpus').size() / len(df_full) * 100
    split_corpus = df_split.groupby('corpus').size() / len(df_split) * 100
    corpus_comparison = pd.DataFrame({
        'Original %': orig_corpus,
        f'{split_name} %': split_corpus,
        'Difference': split_corpus - orig_corpus
    }).round(2)
    print(corpus_comparison)
    
    print("\n--- By Text Type ---")
    orig_text = df_full.groupby('text_type').size() / len(df_full) * 100
    split_text = df_split.groupby('text_type').size() / len(df_split) * 100
    text_comparison = pd.DataFrame({
        'Original %': orig_text,
        f'{split_name} %': split_text,
        'Difference': split_text - orig_text
    }).round(2)
    print(text_comparison)
    
    print("\n--- By Correction Status ---")
    orig_corr = df_full.groupby('corrected').size() / len(df_full) * 100
    split_corr = df_split.groupby('corrected').size() / len(df_split) * 100
    corr_comparison = pd.DataFrame({
        'Original %': orig_corr,
        f'{split_name} %': split_corr,
        'Difference': split_corr - orig_corr
    }).round(2)
    print(corr_comparison)

def create_test_set(df: pd.DataFrame, output_dir: str, test_size: float, 
                   random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """Create stratified test set using line-number-based sampling."""
    print("\n" + "=" * 80)
    print("CREATING TEST SET")
    print("=" * 80)
    print(f"Random seed: {random_seed}")
    
    # Set random seed
    random.seed(random_seed)
    
    df_test, test_indices = stratified_sample_by_line_numbers(
        df, test_size, "test", excluded_indices=set()
    )
    
    df_remaining = df.drop(test_indices).copy()
    
    print(f"\nTest set size: {len(df_test):,} sentences ({len(df_test)/len(df)*100:.2f}%)")
    print(f"Remaining: {len(df_remaining):,} sentences ({len(df_remaining)/len(df)*100:.2f}%)")
    
    print_proportion_verification(df, df_test, "Test")
    
    print("\n" + "=" * 80)
    print("SAVING TEST SET")
    print("=" * 80)
    save_split_files(df_test, "test", output_dir, test_indices)
    
    return df_test, df_remaining, test_indices

def create_train_dev_sets(df_remaining: pd.DataFrame, output_dir: str, 
                          dev_size: float, test_size: float, 
                          random_seed: int, excluded_indices: Set[int]) -> None:
    """Create train and dev sets from remaining data after test set."""
    print("\n" + "=" * 80)
    print("CREATING DEV SET FROM REMAINING DATA")
    print("=" * 80)
    print(f"Random seed: {random_seed}")
    
    # Set random seed
    random.seed(random_seed)
    
    # Calculate dev proportion relative to remaining data
    dev_proportion = dev_size / (1 - test_size)
    
    df_dev, dev_indices = stratified_sample_by_line_numbers(
        df_remaining, dev_proportion, "dev", excluded_indices=excluded_indices
    )
    
    # Train set is everything remaining after test and dev
    all_remaining_indices = set(df_remaining.index)
    train_indices_set = all_remaining_indices - set(dev_indices)
    train_indices = list(train_indices_set)
    df_train = df_remaining.loc[train_indices].copy()
    
    print(f"\nEval set size: {len(df_dev):,} sentences ({len(df_dev)/len(df_remaining)*100:.2f}% of remaining)")
    print(f"Train set size: {len(df_train):,} sentences ({len(df_train)/len(df_remaining)*100:.2f}% of remaining)")
    
    print_proportion_verification(df_remaining, df_dev, "Eval")
    print_proportion_verification(df_remaining, df_train, "Train")
    
    print("\n" + "=" * 80)
    print("SAVING DEV AND TRAIN SETS")
    print("=" * 80)
    save_split_files(df_dev, "dev", output_dir, dev_indices)
    save_split_files(df_train, "train", output_dir, train_indices)

def main(csv_path: str = Paths.EXTRACT_CSV,
         output_dir: str = Paths.SET_SPLITS,
         test_size: float = DataSplits.TEST,
         dev_size: float = DataSplits.DEV,
         random_seed: int = 42,
         create_mode: str = "interactive"):
    """
    Main function to create dataset splits using line-number-based sampling.
    
    Args:
        csv_path: Path to full corpus CSV
        output_dir: Where to save split files
        test_size: Proportion for test set
        dev_size: Proportion for dev set (of total data)
        random_seed: For reproducibility
        create_mode: 'all', 'test', 'train', 'dev', or 'interactive'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load corpus
    print(f"Loading corpus from {csv_path}...")
    df = pd.read_csv(csv_path, encoding="utf-8")
    total_sentences = len(df)
    
    print(f"\nTotal sentences: {total_sentences:,}")
    print(f"CSV line numbers: 0 to {total_sentences - 1} (header excluded)")
    print(f"\nTarget splits:")
    print(f"  Test set: {test_size*100}% = {int(total_sentences * test_size):,} sentences")
    print(f"  Eval set: {dev_size*100}% = {int(total_sentences * dev_size):,} sentences")
    print(f"  Train set: {(1-test_size-dev_size)*100:.1f}% = {int(total_sentences * (1-test_size-dev_size)):,} sentences")
    
    # Check for existing test set
    existing_test_indices = load_existing_indices(output_dir)
    test_exists = len(existing_test_indices) > 0
    
    if test_exists:
        print(f"\n⚠️  Found existing test set with {len(existing_test_indices):,} sentences")
        print(f"    Line numbers: {list(existing_test_indices)[:10]}... (showing first 10)")    
    # Interactive mode selection
    if create_mode == "interactive":
        print("\n" + "=" * 80)
        print("SELECT WHAT TO CREATE")
        print("=" * 80)
        if test_exists:
            print("1. Create train and dev sets (test already exists)")
            print("2. Recreate all (will overwrite test set)")
            print("3. Create only train set")
            print("4. Create only dev set")
        else:
            print("1. Create all sets (test, train, dev)")
            print("2. Create only test set")
            print("3. Create test and train sets")
            print("4. Create test and dev sets")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if test_exists:
            if choice == "1":
                create_mode = "train_dev"
            elif choice == "2":
                create_mode = "all"
            elif choice == "3":
                create_mode = "train"
            elif choice == "4":
                create_mode = "dev"
            else:
                print("Invalid choice. Exiting.")
                return
        else:
            if choice == "1":
                create_mode = "all"
            elif choice == "2":
                create_mode = "test"
            elif choice == "3":
                create_mode = "test_train"
            elif choice == "4":
                create_mode = "test_dev"
            else:
                print("Invalid choice. Exiting.")
                return
    
    # Execute based on mode
    if create_mode == "all":
        df_test, df_remaining, test_indices = create_test_set(df, output_dir, 
                                                              test_size, random_seed)
        create_train_dev_sets(df_remaining, output_dir, dev_size, test_size,
                              random_seed, set(test_indices))
        print("\n✅ All sets created successfully!")
        
    elif create_mode == "test":
        df_test, df_remaining, test_indices = create_test_set(df, output_dir, 
                                                              test_size, random_seed)
        
        response = input("\nDo you want to create train and dev sets now? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            create_train_dev_sets(df_remaining, output_dir, dev_size, test_size,
                                  random_seed, set(test_indices))
            print("\n✅ All sets created successfully!")
        else:
            print("\n✅ Test set created. You can create train/dev sets later.")
    
    elif create_mode == "train_dev":
        if not test_exists:
            print("Error: Test set must exist first. Run with 'test' mode.")
            return
        
        df_remaining = df.drop(list(existing_test_indices)).copy()
        create_train_dev_sets(df_remaining, output_dir, dev_size, test_size,
                              random_seed, existing_test_indices)
        print("\n✅ Train and dev sets created successfully!")
    
    elif create_mode == "train":
        if not test_exists:
            df_test, df_remaining, test_indices = create_test_set(df, output_dir, 
                                                                  test_size, random_seed)
            excluded = set(test_indices)
        else:
            df_remaining = df.drop(list(existing_test_indices)).copy()
            excluded = existing_test_indices
        
        response = input("\nDo you want to create dev set too? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            create_train_dev_sets(df_remaining, output_dir, dev_size, test_size,
                                  random_seed, excluded)
            print("\n✅ Train and dev sets created successfully!")
        else:
            # Create only train set (all remaining data)
            train_indices = df_remaining.index.tolist()
            df_train = df_remaining.copy()
            print("\n" + "=" * 80)
            print("SAVING TRAIN SET (ALL REMAINING DATA)")
            print("=" * 80)
            save_split_files(df_train, "train", output_dir, train_indices)
            print("\n✅ Train set created!")
    
    elif create_mode == "dev":
        if not test_exists:
            df_test, df_remaining, test_indices = create_test_set(df, output_dir, 
                                                                  test_size, random_seed)
            excluded = set(test_indices)
        else:
            df_remaining = df.drop(list(existing_test_indices)).copy()
            excluded = existing_test_indices
        
        response = input("\nDo you want to create train set too? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            create_train_dev_sets(df_remaining, output_dir, dev_size, test_size,
                                  random_seed, excluded)
            print("\n✅ Eval and train sets created successfully!")
        else:
            # Create only dev set
            random.seed(random_seed)
            dev_proportion = dev_size / (1 - test_size)
            df_dev, dev_indices = stratified_sample_by_line_numbers(
                df_remaining, dev_proportion, "dev", excluded_indices=excluded
            )
            print("\n" + "=" * 80)
            print("SAVING DEV SET")
            print("=" * 80)
            save_split_files(df_dev, "dev", output_dir, dev_indices)
            print("\n✅ Eval set created!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create stratified dataset splits using line-number-based sampling'
    )
    parser.add_argument('--csv', default=Paths.EXTRACT_CSV,
                       help='Input CSV file')
    parser.add_argument('--output-dir', default=Paths.SET_SPLITS,
                       help='Output directory for splits')
    parser.add_argument('--test-size', type=float, default=DataSplits.TEST,
                       help='Test set proportion (default: 0.10)')
    parser.add_argument('--dev-size', type=float, default=DataSplits.DEV,
                       help='Eval set proportion (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--mode', choices=['all', 'test', 'train', 'dev', 
                                          'train_dev', 'interactive'],
                       default='interactive',
                       help='What to create (default: interactive)')
    
    args = parser.parse_args()
    
    main(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        dev_size=args.dev_size,
        random_seed=args.seed,
        create_mode=args.mode
    )