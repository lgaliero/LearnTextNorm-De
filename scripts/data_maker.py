#!/usr/bin/env python3
"""
CLI wrapper for dataset splitting pipeline.
Handles argument parsing and orchestrates the splits modules.
"""

import argparse
import sys
import random
from pathlib import Path
from configs import Paths, DataSplits

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from splits import (
    stratify_sample,
    check_proportions,
    save_splits,
    load_indices,
    load_tsv,
    create_json,
    create_norm_files,
    regenerate_splits_from_indices
)


def create_test_set(df, output_dir, test_size, random_seed, paths_config):
    """Create stratified test set."""
    print("\n" + "=" * 80)
    print("CREATING TEST SET")
    print("=" * 80)
    
    random.seed(random_seed)
    
    df_test, test_indices = stratify_sample(
        df, test_size, "test", excluded_indices=set()
    )
    
    df_remaining = df.drop(test_indices).copy()
    
    print(f"\nTest set size: {len(df_test):,} sentences ({len(df_test)/len(df)*100:.2f}%)")
    print(f"Remaining: {len(df_remaining):,} sentences ({len(df_remaining)/len(df)*100:.2f}%)")
    
    check_proportions(df_test, df, "Test")
    
    print("\n" + "=" * 80)
    print("SAVING TEST SET")
    print("=" * 80)
    save_splits(df_test, "test", output_dir, test_indices, df, paths_config)
    
    return df_test, df_remaining, test_indices


def create_train_dev_sets(df, df_remaining, output_dir, dev_size, test_size, 
                          random_seed, excluded_indices, paths_config):
    """Create train and dev sets from remaining data."""
    print("\n" + "=" * 80)
    print("CREATING TRAIN AND DEV SETS")
    print("=" * 80)
    
    random.seed(random_seed)
    
    # Calculate train proportion relative to remaining data
    train_proportion = (1 - test_size - dev_size) / (1 - test_size)
    
    # Train set: sample from all data EXCLUDING test set indices
    df_train, train_indices = stratify_sample(
        df, train_proportion, "train", excluded_indices=excluded_indices
    )
    
    # Dev set is everything remaining after test and train
    all_indices = set(df.index)
    used_indices = excluded_indices | set(train_indices)
    dev_indices_set = all_indices - used_indices
    dev_indices = list(dev_indices_set)
    df_dev = df.loc[dev_indices].copy()

    print(f"\nTrain set size: {len(df_train):,} sentences ({len(df_train)/len(df_remaining)*100:.2f}% of remaining)")
    print(f"Dev set size: {len(df_dev):,} sentences ({len(df_dev)/len(df_remaining)*100:.2f}% of remaining)")
    
    check_proportions(df_train, df_remaining, "Train")
    check_proportions(df_dev, df_remaining, "Dev")
    
    print("\n" + "=" * 80)
    print("SAVING TRAIN AND DEV SETS")
    print("=" * 80)
    save_splits(df_train, "train", output_dir, train_indices, df, paths_config)
    save_splits(df_dev, "dev", output_dir, dev_indices, df, paths_config)


def command_create_all(args, paths_config):
    """Create all splits (test, train, dev)."""
    print(f"\nLoading corpus from: {args.tsv}")
    df = load_tsv(args.tsv)
    print(f"✓ Loaded {len(df):,} sentences")
    
    # Create test set
    df_test, df_remaining, test_indices = create_test_set(
        df, args.output_dir, args.test_size, args.seed, paths_config
    )
    
    # Create train and dev sets
    create_train_dev_sets(
        df, df_remaining, args.output_dir, args.dev_size, 
        args.test_size, args.seed, set(test_indices), paths_config
    )
    
    print("\n✅ All sets created successfully!")


def command_create_test(args, paths_config):
    """Create only test set."""
    print(f"\nLoading corpus from: {args.tsv}")
    df = load_tsv(args.tsv)
    print(f"✓ Loaded {len(df):,} sentences")
    
    # Create test set
    df_test, df_remaining, test_indices = create_test_set(
        df, args.output_dir, args.test_size, args.seed, paths_config
    )
    
    print("\n✅ Test set created!")
    
    # Ask if user wants to continue
    if not args.no_prompt:
        response = input("\nCreate train and dev sets now? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            create_train_dev_sets(
                df, df_remaining, args.output_dir, args.dev_size,
                args.test_size, args.seed, set(test_indices), paths_config
            )
            print("\n✅ All sets created successfully!")


def command_create_train_dev(args, paths_config):
    """Create train and dev sets (test must exist)."""
    print(f"\nLoading corpus from: {args.tsv}")
    df = load_tsv(args.tsv)
    print(f"✓ Loaded {len(df):,} sentences")
    
    # Load existing test indices
    existing_test_indices = load_indices(args.output_dir, "test")
    
    if not existing_test_indices:
        print("❌ Error: Test set must exist first. Run with 'test' or 'all' mode.")
        sys.exit(1)
    
    print(f"✓ Loaded {len(existing_test_indices)} test indices")
    
    df_remaining = df.drop(list(existing_test_indices)).copy()
    
    create_train_dev_sets(
        df, df_remaining, args.output_dir, args.dev_size,
        args.test_size, args.seed, existing_test_indices, paths_config
    )
    
    print("\n✅ Train and dev sets created successfully!")


def command_generate_json(args, paths_config):
    """Generate few-shot prompt JSON."""
    create_json(
        output_dir=args.output_dir,
        baseline_file=args.baseline,
        json_output=args.output,
        update_mode=args.update,
        model_name=args.model,
        paths_config=paths_config
    )


def command_create_norm(args, paths_config):
    """Create NORM files for splits."""
    # Get extract_dir from args or paths_config
    if args.extract_dir:
        extract_dir = args.extract_dir
    elif paths_config and 'EXTRACT_DIR' in paths_config:
        extract_dir = paths_config['EXTRACT_DIR']
    else:
        extract_dir = 'output/extraction'
    
    create_norm_files(
        output_dir=args.output_dir,
        tsv_path=args.tsv,
        extract_dir=extract_dir,
        splits=args.splits
    )


def command_regenerate(args, paths_config):
    """Regenerate split files from existing indices."""
    regenerate_splits_from_indices(
        output_dir=args.output_dir,
        tsv_path=args.tsv,
        paths_config=paths_config
    )


def interactive_mode(args, paths_config):
    """Interactive mode with menu."""
    # Check if test set exists
    test_exists = len(load_indices(args.output_dir, "test")) > 0
    
    print("\n" + "=" * 80)
    print("DATASET SPLITTING - INTERACTIVE MODE")
    print("=" * 80)
    print(f"\nTest set exists: {'Yes' if test_exists else 'No'}")
    print(f"Output directory: {args.output_dir}")
    print(f"TSV file: {args.tsv}")
    
    if test_exists:
        print("\nOptions:")
        print("1. Create train and dev sets (using existing test)")
        print("2. Recreate all sets (will overwrite test)")
        print("3. Generate few-shot JSON")
        print("4. Create NORM files")
        print("5. Regenerate splits from indices")
        print("6. Exit")
    else:
        print("\nOptions:")
        print("1. Create all sets (test, train, dev)")
        print("2. Create test set only")
        print("3. Generate few-shot JSON")
        print("4. Create NORM files")
        print("5. Exit")
    
    choice = input("\nEnter your choice: ").strip()
    
    if test_exists:
        if choice == "1":
            command_create_train_dev(args, paths_config)
        elif choice == "2":
            response = input("\n⚠️  This will overwrite existing test set. Continue? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                command_create_all(args, paths_config)
        elif choice == "3":
            # Ask for model
            print("\nSelect model:")
            print("1. LLaMA")
            print("2. GPT")
            print("3. Gemma")
            model_choice = input("Enter choice (1-3): ").strip()
            model_map = {'1': 'llama', '2': 'gpt', '3': 'gemma'}
            args.model = model_map.get(model_choice)
            command_generate_json(args, paths_config)
        elif choice == "4":
            command_create_norm(args, paths_config)
        elif choice == "5":
            command_regenerate(args, paths_config)
        elif choice == "6":
            print("Bye 👋")
            return
        else:
            print("Invalid choice.")
    else:
        if choice == "1":
            command_create_all(args, paths_config)
        elif choice == "2":
            command_create_test(args, paths_config)
        elif choice == "3":
            print("\nSelect model:")
            print("1. LLaMA")
            print("2. GPT")
            print("3. Gemma")
            model_choice = input("Enter choice (1-3): ").strip()
            model_map = {'1': 'llama', '2': 'gpt', '3': 'gemma'}
            args.model = model_map.get(model_choice)
            command_generate_json(args, paths_config)
        elif choice == "4":
            command_create_norm(args, paths_config)
        elif choice == "5":
            print("Bye 👋")
            return
        else:
            print("Invalid choice.")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Create stratified dataset splits and generate prompts/NORM files'
    )
    
    # Common arguments with defaults from Paths config
    parser.add_argument(
        '--tsv',
        default=Paths.EXTRACT_TSV,
        help=f'Input TSV file (default: {Paths.EXTRACT_TSV})'
    )
    parser.add_argument(
        '--output-dir',
        default=Paths.SET_SPLITS,
        help=f'Output directory for splits (default: {Paths.SET_SPLITS})'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=DataSplits.TEST,
        help=f'Test set proportion (default: {DataSplits.TEST})'
    )
    parser.add_argument(
        '--dev-size',
        type=float,
        default=DataSplits.DEV,
        help=f'Dev set proportion (default: {DataSplits.DEV})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-prompt',
        action='store_true',
        help='Disable interactive prompts'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Create all splits (test, train, dev)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Create test set only')
    
    # Train-dev command
    traindev_parser = subparsers.add_parser('train-dev', help='Create train and dev sets (test must exist)')
    
    # JSON generation command
    json_parser = subparsers.add_parser('json', help='Generate few-shot prompt JSON')
    json_parser.add_argument('--baseline', default=Paths.LLAMA_0, help=f'Path to baseline output file (default: {Paths.LLAMA_0})')
    json_parser.add_argument('--output', default=Paths.LLAMA_JSON, help=f'Output JSON path (default: {Paths.LLAMA_JSON})')
    json_parser.add_argument('--update', action='store_true', help='Update mode (only refresh baselines)')
    json_parser.add_argument('--model', choices=['llama', 'gpt', 'gemma'], default='llama', help='Model name (default: llama)')
    
    # NORM file generation command
    norm_parser = subparsers.add_parser('norm', help='Create NORM files for splits')
    norm_parser.add_argument('--extract-dir', default=Paths.EXTRACT_DIR, help=f'Directory with original corpus NORM files (default: {Paths.EXTRACT_DIR})')
    norm_parser.add_argument('--splits', nargs='*', default=['train', 'dev', 'test'], help='Splits to process')
    
    # Regenerate command
    regen_parser = subparsers.add_parser('regenerate', help='Regenerate splits from existing indices')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode with menu')
    
    args = parser.parse_args()
    
    # Build paths config
    paths_config = {
        'TEST_SRC': Paths.TEST_SRC,
        'TEST_TGT': Paths.TEST_TGT,
        'TRAIN_SRC': Paths.TRAIN_SRC,
        'TRAIN_TGT': Paths.TRAIN_TGT,
        'DEV_SRC': Paths.DEV_SRC,
        'DEV_TGT': Paths.DEV_TGT,
        'LLAMA_0': Paths.LLAMA_0,
        'GPT_0': Paths.GPT_0,
        'GEMMA_0': Paths.GEMMA_0,
        'EXTRACT_DIR': Paths.EXTRACT_DIR,
    }
    
    # Route to appropriate command
    if args.command == 'all':
        command_create_all(args, paths_config)
    elif args.command == 'test':
        command_create_test(args, paths_config)
    elif args.command == 'train-dev':
        command_create_train_dev(args, paths_config)
    elif args.command == 'json':
        command_generate_json(args, paths_config)
    elif args.command == 'norm':
        command_create_norm(args, paths_config)
    elif args.command == 'regenerate':
        command_regenerate(args, paths_config)
    elif args.command == 'interactive':
        interactive_mode(args, paths_config)
    else:
        # Default to interactive mode if no command given
        interactive_mode(args, paths_config)


if __name__ == "__main__":
    main()