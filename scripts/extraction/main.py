
# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":

    # Setup debug logging to file (not to terminal)  
    logging.basicConfig(
        filename=Paths.EXT_LOG_FILE,
        filemode='w',
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    def debug(msg):
        logging.debug(msg)
        
    parser = argparse.ArgumentParser(description='Extract German learner corpora')
    parser.add_argument('--corpora', nargs='+', default=None,
    help='Specify which corpora to process (e.g., LEONIDE Kolipsi_1_L2)')
    parser.add_argument('--output-dir', default=Paths.EXTRACT_DIR)
    parser.add_argument('--format',
                        default=ExtractionParams.FORMAT,
                        choices=['tsv', 'norm', 'both'])
    parser.add_argument('--max-files', type=int, default=None)

    # FIX: Determine which corpora to process
    args = parser.parse_args()
    if args.corpora:
        # User specified corpora via command line
        active_corpora = args.corpora
        print(f"Processing user-specified corpora: {active_corpora}")
    elif hasattr(ExtractionParams, 'ACTIVE_CORPORA') and ExtractionParams.ACTIVE_CORPORA:
        # Use ACTIVE_CORPORA from config if defined and not empty
        active_corpora = ExtractionParams.ACTIVE_CORPORA
        print(f"Processing corpora from config: {active_corpora}")
    else:
        # If ACTIVE_CORPORA is None or empty, DON'T process anything
        active_corpora = []
        print(f"⚠️  No corpora specified - extraction disabled.")
    # Filter to only include corpora that exist in CORPORA config
    configs_to_run = {
        k: v for k, v in ExtractionParams.CORPORA.items()
        if k in active_corpora
    }

    # Validate that all requested corpora exist
    missing_corpora = set(active_corpora) - set(configs_to_run.keys())
    if missing_corpora:
        print(f"⚠️  WARNING: The following corpora are not defined in CORPORA config: {missing_corpora}")
        print(f"Available corpora: {list(ExtractionParams.CORPORA.keys())}")

    if configs_to_run:
        print(f"\n{'='*80}")
        print(f"STARTING EXTRACTION")
        print(f"{'='*80}")
        print(f"Corpora to process: {list(configs_to_run.keys())}")
        print(f"Output directory: {args.output_dir}")
        print(f"Output format: {args.format}")
        if args.max_files:
            print(f"Max files per corpus: {args.max_files}")
        print(f"{'='*80}\n")
        
        df = process_corpora(
            corpus_configs=configs_to_run,
            output_dir=args.output_dir,
            output_format=args.format,
            max_files_per_corpus=args.max_files
        )

        if not df.empty:
            print(f"\n{'='*80}")
            print("EXTRACTION SUMMARY")
            print(f"{'='*80}")
            print(f"Total rows: {len(df)}")
            print("\nCorpus breakdown:")
            print(df.groupby(['corpus', 'lang_prof']).size())
  
    else:
        print("❌ No corpora selected or found. Check your configuration.")
        print(f"Available corpora in config: {list(ExtractionParams.CORPORA.keys())}")