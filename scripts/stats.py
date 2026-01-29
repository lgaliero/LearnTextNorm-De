"""
Command-line interface for corpus statistics computation.
Main entry point for the stats module.
"""

import json
import argparse
from typing import Dict, Optional, Literal

from stats.raw_stats import compute_raw_stats
from stats.processed_stats import compute_processed_stats
from stats.display import display_raw_stats, display_processed_stats, display_comparison

def compute_corpus_statistics(
    mode: Literal['raw', 'processed', 'both'] = 'both',
    corpus_configs: Optional[Dict[str, Dict]] = None,
    tsv_path: Optional[str] = None,
    tokenize_func=None,
    output_json: Optional[str] = None
) -> Dict:
    """
    Compute corpus statistics in specified mode.
    
    Main wrapper function that orchestrates raw and/or processed statistics
    computation based on the specified mode.
    
    Args:
        mode: One of 'raw' (XML), 'processed' (TSV), or 'both'
        corpus_configs: Dict of corpus configurations (required for 'raw' or 'both')
        tsv_path: Path to processed TSV file (required for 'processed' or 'both')
        tokenize_func: Tokenization function to use (defaults to extraction.sentencizer_de.tokenize_for_stats)
        output_json: Optional path to save statistics as JSON
        
    Returns:
        Dict containing statistics based on mode:
        - mode='raw': {'raw': raw_stats}
        - mode='processed': {'processed': processed_stats}
        - mode='both': {'raw': raw_stats, 'processed': processed_stats}
        
    Raises:
        ValueError: If required arguments are missing for specified mode
        FileNotFoundError: If TSV file doesn't exist
    
    Example:
        >>> from configs import Paths, ExtractionParams
        >>> from extraction.sentencizer_de import tokenize_for_stats
        >>> 
        >>> stats = compute_corpus_statistics(
        ...     mode='both',
        ...     corpus_configs=ExtractionParams.CORPORA,
        ...     tsv_path=Paths.EXTRACT_TSV,
        ...     tokenize_func=tokenize_for_stats,
        ...     output_json='corpus_stats.json'
        ... )
    """
    # Default tokenization function
    if tokenize_func is None:
        try:
            from extraction.sentencizer_de import tokenize_for_stats
            tokenize_func = tokenize_for_stats
        except ImportError:
            raise ImportError(
                "Cannot import extraction.sentencizer_de. "
                "Please ensure the extraction module is available or provide tokenize_func."
            )
    
    results = {}
    
    # Compute raw statistics
    if mode in ['raw', 'both']:
        if corpus_configs is None:
            raise ValueError("corpus_configs required for 'raw' or 'both' modes")
        
        print("\n[1/2] Computing raw XML statistics..." if mode == 'both' else "\nComputing raw XML statistics...")
        raw_stats = compute_raw_stats(corpus_configs, tokenize_func)
        results['raw'] = raw_stats
        
        if mode == 'raw':
            display_raw_stats(raw_stats)
    
    # Compute processed statistics
    if mode in ['processed', 'both']:
        if tsv_path is None:
            raise ValueError("tsv_path required for 'processed' or 'both' modes")
        
        print("\n[2/2] Computing processed TSV statistics..." if mode == 'both' else "\nComputing processed TSV statistics...")
        processed_stats = compute_processed_stats(tsv_path, tokenize_func)
        results['processed'] = processed_stats
        
        if mode == 'processed':
            display_processed_stats(processed_stats)
    
    # Display comparison if both modes
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


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description='Compute corpus statistics in different modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Raw XML statistics only
  python -m stats --mode raw
  
  # Processed TSV statistics only
  python -m stats --mode processed --tsv ../master_files/all_corpora.tsv
  
  # Both with comparison
  python -m stats --mode both --tsv ../master_files/all_corpora.tsv
  
  # Save results to JSON
  python -m stats --mode both --tsv ../master_files/all_corpora.tsv --output stats.json
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
    
    # Load tokenization function
    try:
        from extraction.sentencizer_de import tokenize_for_stats
    except ImportError:
        print("ERROR: Cannot import extraction.sentencizer_de")
        print("Please ensure the extraction module is available")
        return 1
    
    # Use provided TSV or default
    tsv_path = args.tsv or default_tsv
    
    # Validate inputs
    if args.mode in ['processed', 'both'] and not tsv_path:
        parser.error(f"--tsv required for mode '{args.mode}'")
    
    if args.mode in ['raw', 'both'] and not corpus_configs:
        parser.error(f"corpus_configs not found in configs.py (required for mode '{args.mode}')")
    
    # Run statistics
    try:
        compute_corpus_statistics(
            mode=args.mode,
            corpus_configs=corpus_configs if args.mode in ['raw', 'both'] else None,
            tsv_path=tsv_path if args.mode in ['processed', 'both'] else None,
            tokenize_func=tokenize_for_stats,
            output_json=args.output
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
