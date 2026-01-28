"""
JSON generation for few-shot prompting examples.
Creates 2S_prompts.json with semantically similar examples for each test sentence.
"""

import os
import json
import time
from typing import List, Dict, Optional
import torch
from sentence_transformers import SentenceTransformer, util
import spacy


# Global models for lazy loading
_nlp_model = None
_sentence_model = None


def get_nlp_model():
    """Lazy load spaCy model."""
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("de_core_news_sm")
    return _nlp_model


def get_tf_model():
    """Lazy load sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model


def tokenize_and_preprocess(sentence: str, nlp) -> str:
    """
    Tokenize and preprocess sentences.
    
    Args:
        sentence: Input sentence
        nlp: spaCy model
        
    Returns:
        Preprocessed sentence
    """
    doc = nlp(sentence)
    return " ".join([token.text for token in doc if not token.is_punct and not token.is_space])


def load_files(file_path: str) -> List[str]:
    """
    Load text file lines.
    
    Args:
        file_path: Path to text file
        
    Returns:
        List of lines (empty list if file doesn't exist)
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return [line.strip() for line in f if line.strip()]


def create_json(
    output_dir: str,
    baseline_file: Optional[str] = None,
    json_output: Optional[str] = None,
    update_mode: bool = False,
    model_name: Optional[str] = None,
    paths_config: Optional[Dict] = None
) -> None:
    """
    Generate 2S_prompts.json with 2-shot examples and baseline outputs.
    
    Args:
        output_dir: Directory containing test.src, test.tgt, train.src, train.tgt, dev.src, dev.tgt
        baseline_file: Path to baseline model output file (e.g., baseline_LLaMA3_2.tgt)
        json_output: Output JSON file path
        update_mode: If True, only update baseline outputs in existing JSON
        model_name: Model name for path resolution ('llama', 'gpt', 'gemma')
        paths_config: Optional dict with path mappings (TEST_SRC, TRAIN_SRC, LLAMA_0, etc.)
    """
    print("\n" + "=" * 80)
    if update_mode:
        print("UPDATING BASELINE OUTPUTS IN 2S_prompts.json")
    else:
        print("GENERATING 2S_prompts.json WITH 2-SHOT EXAMPLES")
    print("=" * 80)
    
    # Set default paths based on model_name
    if json_output is None:
        if model_name:
            json_output = os.path.join(output_dir, f"2S_prompts_{model_name}.json")
        else:
            if paths_config and 'JSON' in paths_config:
                json_output = paths_config['JSON']
            else:
                json_output = os.path.join(output_dir, "2S_prompts.json")
    
    if baseline_file is None:
        # Map model names to config paths
        if paths_config and model_name:
            baseline_map = {
                'llama': paths_config.get('LLAMA_0'),
                'gpt': paths_config.get('GPT_0'),
                'gemma': paths_config.get('GEMMA_0')
            }
            baseline_file = baseline_map.get(model_name)
        
        if baseline_file is None:
            baseline_file = os.path.join(output_dir, "0shot.hyp")
    
    # Load baseline outputs
    baseline_outputs = load_files(baseline_file)
    print(f"\nLoaded {len(baseline_outputs)} baseline outputs from {baseline_file}")
    
    if update_mode:
        # Update mode: load existing JSON and only update baselines
        if not os.path.exists(json_output):
            print(f"Error: {json_output} does not exist. Cannot update.")
            return
        
        with open(json_output, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"Loaded {len(results)} entries from existing JSON")
        
        updates = []
        for i, entry in enumerate(results):
            old_baseline = entry.get('baseline_output', 'to be added')
            new_baseline = baseline_outputs[i] if i < len(baseline_outputs) else 'to be added'
            
            if old_baseline != new_baseline:
                entry['baseline_output'] = new_baseline
                updates.append({
                    'index': i,
                    'test_source': entry['test_source'],
                    'old_baseline': old_baseline,
                    'new_baseline': new_baseline
                })
        
        # Save updated JSON
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Updated {len(updates)} baseline outputs in {json_output}")
        
        if updates:
            print("\nUpdates made:")
            for update in updates[:10]:  # Show first 10
                print(f"  Index {update['index']}: \"{update['test_source'][:50]}...\"")
                print(f"    Old: \"{update['old_baseline'][:50]}...\"")
                print(f"    New: \"{update['new_baseline'][:50]}...\"")
            if len(updates) > 10:
                print(f"  ... and {len(updates) - 10} more updates")
        
        return
    
    # Normal mode: generate full JSON
    # Load models
    nlp = get_nlp_model()
    model = get_tf_model()
    
    # Load all files using paths from config
    if paths_config:
        test_src = load_files(paths_config.get('TEST_SRC', os.path.join(output_dir, "test.src")))
        train_src = load_files(paths_config.get('TRAIN_SRC', os.path.join(output_dir, "train.src")))
        train_tgt = load_files(paths_config.get('TRAIN_TGT', os.path.join(output_dir, "train.tgt")))
        dev_src = load_files(paths_config.get('DEV_SRC', os.path.join(output_dir, "dev.src")))
        dev_tgt = load_files(paths_config.get('DEV_TGT', os.path.join(output_dir, "dev.tgt")))
    else:
        test_src = load_files(os.path.join(output_dir, "test.src"))
        train_src = load_files(os.path.join(output_dir, "train.src"))
        train_tgt = load_files(os.path.join(output_dir, "train.tgt"))
        dev_src = load_files(os.path.join(output_dir, "dev.src"))
        dev_tgt = load_files(os.path.join(output_dir, "dev.tgt"))
    
    print(f"\nLoaded files:")
    print(f"  Test: {len(test_src)} sentences")
    print(f"  Train: {len(train_src)} sentences")
    print(f"  Dev: {len(dev_src)} sentences")
    start = time.time()
    
    # Create source-target pairs with metadata
    train_pairs = [
        {"source": src, "target": tgt, "dataset": "train"} 
        for src, tgt in zip(train_src, train_tgt)
    ]
    dev_pairs = [
        {"source": src, "target": tgt, "dataset": "dev"} 
        for src, tgt in zip(dev_src, dev_tgt)
    ]
    
    # Combine all candidate pairs
    all_candidate_pairs = train_pairs + dev_pairs
    
    # Exclude test sentences from candidate pool
    excluded_sentences = set(test_src)
    candidate_pairs = [
        pair for pair in all_candidate_pairs 
        if pair["source"] not in excluded_sentences
    ]
    
    candidate_sources = [pair["source"] for pair in candidate_pairs]
    print(f"✓ Created candidate pairs in {time.time() - start:.2f}s")
    
    print(f"\nCandidate pool: {len(candidate_pairs)} sentences (excluding test set)")
    
    # OPTIMIZATION: Encode all candidates ONCE (not 2000 times!)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print("Preprocessing and encoding all candidates (this happens once)...")
    
    # Simple preprocessing (skip spaCy for speed)
    preprocessed_candidates = [s.lower().strip() for s in candidate_sources]
    print(f"✓ Preprocessed in {time.time() - start:.2f}s")
    start = time.time()
    
    # Encode all candidates in one batch
    candidate_embeddings = model.encode(
        preprocessed_candidates,
        convert_to_tensor=True,
        batch_size=256,
        show_progress_bar=True,
        device=device
    )
    print(f"✓ Encoded {len(candidate_pairs)} candidates")
    print(f"✓ Encoding took {time.time() - start:.2f}s")
    start = time.time()
    
    # Process each test sentence
    results = []
    
    print(f"\nProcessing {len(test_src)} test sentences...")
    for idx, test_sentence in enumerate(test_src):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(test_src)}...")
        
        # Preprocess and encode just this test sentence
        preprocessed_test = test_sentence.lower().strip()
        test_embedding = model.encode(preprocessed_test, convert_to_tensor=True, device=device)
        
        # Compute similarity against ALL candidates (fast - already encoded)
        similarity_scores = util.pytorch_cos_sim(test_embedding, candidate_embeddings).squeeze(0).tolist()
        
        # Get top 2 similar sentences
        scored_pairs = [
            (candidate_pairs[i], similarity_scores[i]) 
            for i in range(len(candidate_pairs))
        ]
        scored_pairs = [(pair, score) for pair, score in scored_pairs if score > 0.00]
        scored_pairs = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:2]
        
        # Build examples
        examples = []
        for pair, score in scored_pairs:
            examples.append({
                "source": pair["source"],
                "target": pair["target"],
                "score": round(score, 4),
                "dataset": pair["dataset"]
            })
        
        # Get baseline output
        baseline_output = baseline_outputs[idx] if idx < len(baseline_outputs) else "to be added"
        
        # Add to results
        result_entry = {
            "test_source": test_sentence,
            "examples": examples,
            "baseline_output": baseline_output
        }
        results.append(result_entry)
    
    # Write to JSON file
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Verify no duplicate test sources (this would be a real problem)
    test_sources = [r['test_source'] for r in results]
    if len(test_sources) != len(set(test_sources)):
        print("\n⚠️  WARNING: Duplicate test_source entries found in JSON!")
    
    # Count duplicate baselines (this is OK - model can produce same output)
    baseline_counts = {}
    for r in results:
        baseline = r['baseline_output']
        baseline_counts[baseline] = baseline_counts.get(baseline, 0) + 1
    
    duplicates = {k: v for k, v in baseline_counts.items() if v > 1 and k != "to be added"}
    if duplicates:
        print(f"\nℹ️  Note: {len(duplicates)} baseline outputs appear multiple times (this is normal)")
        print(f"   Example: \"{list(duplicates.keys())[0][:50]}...\" appears {list(duplicates.values())[0]} times")
    
    print(f"\n✓ JSON file saved to: {json_output}")
    print(f"✓ Processed {len(results)} test sentences")
    print(f"✓ Baselines included: {sum(1 for r in results if r['baseline_output'] != 'to be added')}/{len(results)}")