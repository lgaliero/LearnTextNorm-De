"""
Batch processing utilities for running inference on multiple sentences.
"""

import time
import psutil
from typing import List, Dict, Optional, Tuple

from .model_api import ModelClient, Example
from .prompt_utils import find_examples_for_sentence, extract_examples_from_entry


def process_single_sentence(
    idx: int,
    sentence: str,
    mode: str,
    model: str,
    model_client: ModelClient,
    system_baseline: str,
    system_2shot: str,
    examples_data: List[Dict] = None
) -> Optional[Tuple[int, str, str]]:
    """
    Process a single sentence through the model.
    
    Args:
        idx: Sentence index (for tracking)
        sentence: Input sentence
        mode: Inference mode
        model: Model identifier
        model_client: ModelClient instance
        system_baseline: System prompt for baseline
        system_2shot: System prompt for 2-shot
        examples_data: List of example dictionaries (for 2-shot mode)
        
    Returns:
        Tuple of (index, sentence, output) if successful, None on error
    """
    start_time = time.time()
    mem_before = psutil.Process().memory_info().rss / 1024**3
    
    print(f"[{idx}] Starting (Memory: {mem_before:.2f}GB)")
    print(f"[{idx}] Processing: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
    
    examples = None
    baseline_output = None
    
    # This block only runs for 2-shot-json mode
    if mode == "2-shot-json":
        if not examples_data:
            print(f"  [{idx}] ⚠️  Warning: No examples data provided, skipping sentence.")
            return None
            
        entry = find_examples_for_sentence(sentence, examples_data)
        
        if not entry:
            print(f"  [{idx}] ⚠️  Warning: No examples found, skipping sentence.")
            return None
        
        examples = extract_examples_from_entry(entry, count=2)
        baseline_output = entry.get('baseline_output', 'to be added')
    
    # This try block runs for BOTH baseline and 2-shot-json modes
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
        
        # Timing applies to BOTH modes
        elapsed = time.time() - start_time
        mem_after = psutil.Process().memory_info().rss / 1024**3
        print(f"  [{idx}] ✓ Generated output")
        print(f"  [{idx}] Completed in {elapsed:.1f}s (Memory: {mem_after:.2f}GB, Δ{mem_after-mem_before:.2f}GB)")
        
        return (idx, sentence, output)
        
    except Exception as e:
        print(f"  [{idx}] ✗ Error: {e}")
        return None


def process_batch(
    sentences: List[str],
    mode: str,
    model: str,
    model_client: ModelClient,
    system_baseline: str,
    system_2shot: str,
    examples_data: List[Dict] = None
) -> List[Tuple[int, str, str]]:
    """
    Process a batch of sentences sequentially.
    
    Args:
        sentences: List of input sentences
        mode: Inference mode
        model: Model identifier
        model_client: ModelClient instance
        system_baseline: System prompt for baseline
        system_2shot: System prompt for 2-shot
        examples_data: List of example dictionaries (for 2-shot mode)
        
    Returns:
        List of (index, sentence, output) tuples for successful results
    """
    results = []
    
    for idx, sentence in enumerate(sentences, 1):
        result = process_single_sentence(
            idx, sentence, mode, model, model_client,
            system_baseline, system_2shot, examples_data
        )
        if result is not None:
            results.append(result)
    
    return results
