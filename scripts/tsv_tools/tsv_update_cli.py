#!/usr/bin/env python3
"""
CLI wrapper for TSV batch update tool.
Handles argument parsing and orchestrates the TSV update modules.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tsv_tools import (
    batch_update_tsv,
    find_norm_files_in_directory,
    get_norm_path_for_corpus
)

# Import configs
try:
    from configs import Paths
except ImportError:
    print("Error: configs module not found. Make sure configs.py is in the Python path.")
    sys.exit(1)


def batch_update_directory(directory: str, tsv_name: str, output: str, log_edits: bool):
    """
    Update TSV from all NORM files in a directory.
    
    Args:
        directory: Directory containing TSV and NORM files
        tsv_name: Name of TSV file
        output: Output path (None to overwrite)
        log_edits: Whether to create edit log
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)
    
    # Find TSV file
    tsv_path = directory / tsv_name
    if not tsv_path.exists():
        print(f"ERROR: TSV file not found: {tsv_path}")
        sys.exit(1)
    
    # Find NORM files
    norm_files = find_norm_files_in_directory(str(directory), exclude_tsv=True)
    if not norm_files:
        print(f"ERROR: No .norm files found in {directory}")
        sys.exit(1)
    
    print(f"Found {len(norm_files)} NORM files in {directory}:")
    for f in norm_files:
        print(f"  • {Path(f).name}")
    
    # Output path
    if output is None:
        output = str(tsv_path)
    
    batch_update_tsv(
        tsv_path=str(tsv_path),
        norm_files=norm_files,
        output_path=output,
        log_edits=log_edits
    )


def update_specific_corpora(tsv_file: str, corpora: list, output: str, log_edits: bool):
    """
    Update TSV from specific corpora.
    
    Args:
        tsv_file: Path to TSV file
        corpora: List of corpus names (None for all)
        output: Output path (None to overwrite)
        log_edits: Whether to create edit log
    """
    tsv_file = Path(tsv_file)
    
    if not tsv_file.exists():
        print(f"ERROR: TSV file not found: {tsv_file}")
        sys.exit(1)
    
    # Convert corpus names to NORM file paths
    if not corpora:
        # Auto-discover all NORM files in EXTRACT_DIR
        extract_dir = Paths.EXTRACT_DIR if hasattr(Paths, 'EXTRACT_DIR') else Path('.')
        norm_files = find_norm_files_in_directory(str(extract_dir), exclude_tsv=True)
        if not norm_files:
            print(f"ERROR: No .norm files found in {extract_dir}")
            sys.exit(1)
        print(f"Auto-discovered {len(norm_files)} NORM files")
    else:
        # Get NORM paths for specified corpora
        extract_dir = Paths.EXTRACT_DIR if hasattr(Paths, 'EXTRACT_DIR') else Path('.')
        norm_files = []
        missing = []
        
        for corpus_name in corpora:
            norm_path = get_norm_path_for_corpus(corpus_name, extract_dir)
            if norm_path:
                norm_files.append(norm_path)
                print(f"✓ {corpus_name} → {Path(norm_path).name}")
            else:
                missing.append(corpus_name)
        
        if missing:
            print(f"\n❌ ERROR: NORM files not found for:")
            for corpus in missing:
                print(f"  • {corpus}")
            print(f"\nSearched in: {extract_dir}")
            sys.exit(1)
    
    batch_update_tsv(
        tsv_path=str(tsv_file),
        norm_files=norm_files,
        output_path=output,
        log_edits=log_edits
    )


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Batch update TSV with automatic index adjustment (no metadata required)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Batch update command - ALL files in directory (MAIN COMMAND)
    batch_parser = subparsers.add_parser(
        'batch-update',
        help='Update TSV from ALL .norm files in directory'
    )
    batch_parser.add_argument(
        '--directory',
        default=str(Paths.EXTRACT_DIR) if hasattr(Paths, 'EXTRACT_DIR') else '.',
        help=f'Directory with TSV and .norm files'
    )
    batch_parser.add_argument(
        '--tsv-name',
        default='all_corpora.tsv',
        help='Name of TSV file in directory (default: all_corpora.tsv)'
    )
    batch_parser.add_argument(
        '--output',
        default=None,
        help='Output TSV path (default: overwrites original in same directory)'
    )
    batch_parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable detailed edit log file'
    )
    
    # Update command - specific files only (advanced)
    update_parser = subparsers.add_parser(
        'update',
        help='Update TSV from specific corpora by name'
    )
    update_parser.add_argument(
        '--tsv-file',
        required=True,
        help='Path to TSV file'
    )
    update_parser.add_argument(
        '--corpora',
        nargs='*',
        default=None,
        help='Corpus names (e.g., LEONIDE Kolipsi_1_L2). Leave empty to process all.'
    )
    update_parser.add_argument(
        '--output',
        default=None,
        help='Output TSV path (overwrites if not provided)'
    )
    update_parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable detailed edit log file'
    )
    
    args = parser.parse_args()
    
    if args.command == 'batch-update':
        batch_update_directory(
            directory=args.directory,
            tsv_name=args.tsv_name,
            output=args.output,
            log_edits=not args.no_log
        )
    
    elif args.command == 'update':
        update_specific_corpora(
            tsv_file=args.tsv_file,
            corpora=args.corpora,
            output=args.output,
            log_edits=not args.no_log
        )
    
    else:
        print("\nUsage examples:")
        print("\n1. Update from all NORM files in directory:")
        print(f"   python {Path(__file__).name} batch-update \\")
        print("       --directory output/extraction \\")
        print("       --tsv-name all_corpora.tsv")
        print("\n2. Update from specific corpora:")
        print(f"   python {Path(__file__).name} update \\")
        print("       --tsv-file output/all_corpora.tsv \\")
        print("       --corpora LEONIDE Kolipsi_1_L2")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
