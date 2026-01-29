#!/usr/bin/env python3
"""
CLI wrapper for LLM inference pipeline.
Handles argument parsing and orchestrates the inference modules.
"""

import argparse
import sys
import time
import psutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import (
    ModelClient,
    load_examples_json,
    load_sentences_from_file,
    append_to_tgt,
    find_examples_for_sentence,
    get_examples_interactively,
    extract_examples_from_entry
)

# Import configs
try:
    from configs import Paths, ApiConfig
except ImportError:
    print("Error: configs module not found. Make sure configs.py is in the Python path.")
    sys.exit(1)


def process_file_mode(input_file: str, mode: str, model: str, model_client: ModelClient,
                      system_baseline: str, system_2shot: str, examples_data: list, paths_config: dict):
    """Process all sentences from a file, one at a time."""
    
    # Load sentences
    try:
        sentences = load_sentences_from_file(input_file)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
    
    print(f"Processing {len(sentences)} sentences from {input_file}\n")
    
    processed = 0
    skipped = 0
    
    # Process each sentence one at a time
    for idx, sentence in enumerate(sentences, 1):
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / 1024**3
        
        print(f"[{idx}/{len(sentences)}] Starting (Memory: {mem_before:.2f}GB)")
        print(f"[{idx}/{len(sentences)}] Processing: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
        
        examples = None
        baseline_output = None
        
        # Get examples for 2-shot mode
        if mode == "2-shot-json":
            if not examples_data:
                print(f"  [{idx}] ⚠️  Warning: No examples data provided, skipping.")
                skipped += 1
                continue
                
            entry = find_examples_for_sentence(sentence, examples_data)
            
            if not entry:
                print(f"  [{idx}] ⚠️  Warning: No examples found, skipping.")
                skipped += 1
                continue
            
            examples = extract_examples_from_entry(entry, count=2)
            baseline_output = entry.get('baseline_output', 'to be added')
            print(f"  [{idx}] Found {len(examples)} examples")
        
        # Query model
        try:
            output = model_client.query_model(
                sentence,
                mode,
                model,
                system_baseline,
                system_2shot,
                examples=examples,
                baseline_output=baseline_output
            )
            
            # Write immediately to disk
            append_to_tgt(output, mode, model, paths_config)
            
            # Timing
            elapsed = time.time() - start_time
            mem_after = psutil.Process().memory_info().rss / 1024**3
            
            print(f"  [{idx}] ✓ Output: {output[:80]}{'...' if len(output) > 80 else ''}")
            print(f"  [{idx}] ✓ Written to file")
            print(f"  [{idx}] Completed in {elapsed:.1f}s (Memory: {mem_after:.2f}GB, Δ{mem_after-mem_before:.2f}GB)\n")
            
            processed += 1
            
        except Exception as e:
            print(f"  [{idx}] ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Completed: {processed} sentences processed, {skipped} skipped")
    print(f"{'='*60}\n")


def interactive_mode(mode: str, model: str, model_client: ModelClient, 
                    system_baseline: str, system_2shot: str, examples_data: list, paths_config: dict):
    """Run in interactive mode - prompt user for sentences."""
    
    print("Paste a sentence and press Enter.")
    if mode == "2-shot-json":
        print("The system will find matching examples from JSON.")
    print("Type :q to quit.\n")
    
    while True:
        examples = None
        baseline_output = None
        
        sentence = input("\nSRC > ").strip()
        
        if sentence == ":q":
            print("Bye 👋")
            break
        
        if not sentence:
            continue
        
        if mode == "2-shot-json":
            entry = find_examples_for_sentence(sentence, examples_data)
            
            if not entry:
                print(f"⚠️  Warning: No examples found for this sentence in JSON.")
                response = input("   Continue with manual input? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    continue
                
                examples = get_examples_interactively()
                if len(examples) < 2:
                    print("Error: Need exactly 2 examples. Skipping...\n")
                    continue
                
                baseline_output = input("\nTGT_BASELINE > ").strip()
            else:
                examples = extract_examples_from_entry(entry, count=2)
                baseline_output = entry.get('baseline_output', 'to be added')
                
                print(f"\n✓ Found examples:")
                for i, (src, tgt) in enumerate(examples, 1):
                    print(f"  Example {i}:")
                    print(f"    SRC: {src[:80]}{'...' if len(src) > 80 else ''}")
                    print(f"    TGT: {tgt[:80]}{'...' if len(tgt) > 80 else ''}")
                
                if baseline_output != "to be added":
                    print(f"\n✓ Baseline: {baseline_output[:80]}{'...' if len(baseline_output) > 80 else ''}")
                print()

        try:
            output = model_client.query_model(
                sentence,
                mode,
                model,
                system_baseline,
                system_2shot,
                examples=examples,
                baseline_output=baseline_output
            )
            
            print("\nTGT >", output, "\n")
            append_to_tgt(output, mode, model, paths_config)
       
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='LLM inference for text normalization'
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "2-shot", "2-shot-json"],
        default=ApiConfig.MODE if hasattr(ApiConfig, 'MODE') else "baseline",
        help="Inference mode",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=Paths.JSON if hasattr(Paths, 'JSON') else None,
        help="Path to examples JSON file (for 2-shot-json mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ApiConfig.MODEL if hasattr(ApiConfig, 'MODEL') else "llama3.2",
        help="Model to use for inference (e.g., llama3.2, gemma2, etc.)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input file with source sentences (one per line). If not provided, runs in interactive mode.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=ApiConfig.HOST if hasattr(ApiConfig, 'HOST') else "http://localhost:11434",
        help="API host URL",
    )
    args = parser.parse_args()

    mode = args.mode
    model = args.model
    
    # Initialize model client
    model_client = ModelClient(host=args.host)
    
    # Get system prompts from config
    system_baseline = ApiConfig.SYS_BASELINE if hasattr(ApiConfig, 'SYS_BASELINE') else ""
    system_2shot = ApiConfig.SYS_2SHOT if hasattr(ApiConfig, 'SYS_2SHOT') else ""
    
    # Create paths config
    paths_config = {
        'LLAMA_0': Paths.LLAMA_0 if hasattr(Paths, 'LLAMA_0') else None,
        'GPT_0': Paths.GPT_0 if hasattr(Paths, 'GPT_0') else None,
        'GEMMA_0': Paths.GEMMA_0 if hasattr(Paths, 'GEMMA_0') else None,
        'LLAMA_2': Paths.LLAMA_2 if hasattr(Paths, 'LLAMA_2') else None,
        'GPT_2': Paths.GPT_2 if hasattr(Paths, 'GPT_2') else None,
        'GEMMA_2S': Paths.GEMMA_2S if hasattr(Paths, 'GEMMA_2S') else None,
    }
    
    # Load examples JSON if using 2-shot-json mode
    examples_data = []
    if mode == "2-shot-json":
        if not args.json:
            print("Error: --json argument required for 2-shot-json mode")
            sys.exit(1)
            
        examples_data = load_examples_json(args.json)
        if not examples_data:
            print("Error: Could not load examples JSON. Exiting.")
            sys.exit(1)
        print(f"✓ Loaded {len(examples_data)} examples from {args.json}\n")
    
    # Run in file or interactive mode
    if args.input:
        process_file_mode(args.input, mode, model, model_client, 
                         system_baseline, system_2shot, examples_data, paths_config)
    else:
        interactive_mode(mode, model, model_client,
                        system_baseline, system_2shot, examples_data, paths_config)


if __name__ == "__main__":
    main()