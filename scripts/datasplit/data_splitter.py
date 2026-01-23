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
import json
from typing import Dict, Tuple, Optional, List, Set
from sentence_transformers import SentenceTransformer, util
import spacy
import torch
import time

# Global models for example generation (lazy loaded)
_nlp_model = None
_sentence_model = None

def get_nlp_model():
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("de_core_news_sm")
    return _nlp_model

def get_tf_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def load_indices(output_dir: str) -> set:
    """Load indices from existing test set to avoid overlap."""
    test_indices_file = Paths.TEST_INDICES if hasattr(Paths, 'TEST_INDICES') else os.path.join(output_dir, "test_indices.tsv")
    if os.path.exists(test_indices_file):
        # Read TSV and extract DF_INDEX column
        df_indices = pd.read_csv(test_indices_file, sep='\t', encoding='utf-8')
        return set(df_indices['DF_INDEX'].tolist())
    return set()

def save_splits(df_split: pd.DataFrame, split_name: str, 
                output_dir: str, indices: List[int], df_full: pd.DataFrame) -> None:
    """Save source, target, and indices files for a dataset split."""
    # Use paths from configs if available
    if split_name == "test":
        src_file = Paths.TEST_SRC if hasattr(Paths, 'TEST_SRC') else os.path.join(output_dir, f"{split_name}.src")
        tgt_file = Paths.TEST_TGT if hasattr(Paths, 'TEST_TGT') else os.path.join(output_dir, f"{split_name}.tgt")
    elif split_name == "train":
        src_file = Paths.TRAIN_SRC if hasattr(Paths, 'TRAIN_SRC') else os.path.join(output_dir, f"{split_name}.src")
        tgt_file = Paths.TRAIN_TGT if hasattr(Paths, 'TRAIN_TGT') else os.path.join(output_dir, f"{split_name}.tgt")
    elif split_name == "dev":
        src_file = Paths.DEV_SRC if hasattr(Paths, 'DEV_SRC') else os.path.join(output_dir, f"{split_name}.src")
        tgt_file = Paths.DEV_TGT if hasattr(Paths, 'DEV_TGT') else os.path.join(output_dir, f"{split_name}.tgt")
    else:
        src_file = os.path.join(output_dir, f"{split_name}.src")
        tgt_file = os.path.join(output_dir, f"{split_name}.tgt")
    
    indices_file = os.path.join(output_dir, f"{split_name}_indices.tsv")  # Changed to .tsv

    # Create directories if they don't exist
    src_dir = os.path.dirname(src_file)
    tgt_dir = os.path.dirname(tgt_file)
    indices_dir = os.path.dirname(indices_file)

    if src_dir:  # Only create if there's actually a directory component
        os.makedirs(src_dir, exist_ok=True)
    if tgt_dir:
        os.makedirs(tgt_dir, exist_ok=True)
    if indices_dir:
        os.makedirs(indices_dir, exist_ok=True)
    
    with open(src_file, 'w', encoding='utf-8') as f:
        for src in df_split['src']:
            f.write(f"{src}\n")
    
    with open(tgt_file, 'w', encoding='utf-8') as f:
        for tgt in df_split['tgt']:
            f.write(f"{tgt}\n")
    
    # Create indices TSV with full metadata
    # Get the rows from full dataframe for these indices
    df_indices = df_full.loc[indices].copy()
    
    # Add DF_INDEX column as first column
    df_indices.insert(0, 'DF_INDEX', indices)
    
    # Save as TSV with header
    df_indices.to_csv(indices_file, sep='\t', index=False, encoding='utf-8')
    
    print(f"✓ Saved {split_name} source: {src_file}")
    print(f"✓ Saved {split_name} target: {tgt_file}")
    print(f"✓ Saved {split_name} indices: {indices_file}")


def stratify_sample(df: pd.DataFrame, sample_size: float, 
                                     split_name: str, 
                                     excluded_indices: Set[int] = None) -> Tuple[pd.DataFrame, List[int]]:
    """
    Perform stratified sampling by iteratively picking random line numbers from strata.
    Maintains proportions by tracking how many samples each stratum needs.
    Excludes any indices already used in previous sets.
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
    
    # Build stratum info: available line numbers (excluding already used) and target sample size
    strata_info = {}
    total_available = 0
    
    for strat_key, group in df.groupby('strat_key'):
        # Get line numbers for this stratum, EXCLUDING any already used indices
        available_line_numbers = [idx for idx in group.index if idx not in excluded_indices]
        
        if len(available_line_numbers) == 0:
            continue
        
        n_total = len(available_line_numbers)
        n_sample = max(1, int(n_total * sample_size))
        n_sample = min(n_sample, n_total)
        
        strata_info[strat_key] = {
            'available': available_line_numbers.copy(),
            'target': n_sample,
            'sampled': []
        }
        total_available += n_total
    
    print(f"Total available indices (excluding {len(excluded_indices)} already used): {total_available:,}")
    
    # Iteratively pick random lines from strata until all targets are met
    sampled_indices = []
    
    while True:
        # Find strata that still need samples
        active_strata = {k: v for k, v in strata_info.items() 
                        if len(v['sampled']) < v['target'] and len(v['available']) > 0}
        
        if not active_strata:
            break
        
        # Pick a random stratum from those that need samples
        strat_key = random.choice(list(active_strata.keys()))
        
        # Pick a random line number from this stratum's available pool
        line_num = random.choice(strata_info[strat_key]['available'])
        
        # Record the sample
        strata_info[strat_key]['sampled'].append(line_num)
        strata_info[strat_key]['available'].remove(line_num)
        sampled_indices.append(line_num)
    
    # Prepare sampling details for reporting
    sampling_details = []
    for strat_key, info in strata_info.items():
        parts = strat_key.split('_')
        if len(parts) >= 3:
            corpus = '_'.join(parts[:-2])
            text_type = parts[-2]
            corrected = parts[-1]
        else:
            corpus, text_type, corrected = strat_key, "unknown", "unknown"
        
        total_available = len(info['available']) + len(info['sampled'])
        sampling_details.append({
            'stratum': strat_key,
            'corpus': corpus,
            'text_type': text_type,
            'corrected': corrected,
            'available_lines': total_available,
            'sampled': len(info['sampled']),
            'percentage': f"{len(info['sampled'])/total_available*100:.2f}%",
            'sample_line_numbers': info['sampled'][:5]
        })
    
    # Create sampled dataframe (keep extraction order)
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
    
    return df_sampled, sampled_indices

def check_proportion(df_full: pd.DataFrame, df_split: pd.DataFrame, 
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
    
    df_test, test_indices = stratify_sample(
        df, test_size, "test", excluded_indices=set()
    )
    
    df_remaining = df.drop(test_indices).copy()
    
    print(f"\nTest set size: {len(df_test):,} sentences ({len(df_test)/len(df)*100:.2f}%)")
    print(f"Remaining: {len(df_remaining):,} sentences ({len(df_remaining)/len(df)*100:.2f}%)")
    
    check_proportion(df, df_test, "Test")
    
    print("\n" + "=" * 80)
    print("SAVING TEST SET")
    print("=" * 80)
    save_splits(df_test, "test", output_dir, test_indices, df)  # Added df parameter
    
    return df_test, df_remaining, test_indices

def create_train_dev_sets(df: pd.DataFrame, df_remaining: pd.DataFrame, output_dir: str, 
                          dev_size: float, test_size: float, 
                          random_seed: int, excluded_indices: Set[int]) -> None:
    """Create train and dev sets from remaining data after test set."""
    print("\n" + "=" * 80)
    print("CREATING TRAIN SET FROM REMAINING DATA")
    print("=" * 80)
    
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
    
    check_proportion(df_remaining, df_train, "Train")
    check_proportion(df_remaining, df_dev, "Dev")
    
    print("\n" + "=" * 80)
    print("SAVING TRAIN AND DEV SETS")
    print("=" * 80)
    save_splits(df_train, "train", output_dir, train_indices, df)  # Added df parameter
    save_splits(df_dev, "dev", output_dir, dev_indices, df)
            # Added df parameter
def tokenize_and_preprocess(sentence: str, nlp) -> str:
    """Tokenize and preprocess sentences."""
    doc = nlp(sentence)
    return " ".join([token.text for token in doc if not token.is_punct and not token.is_space])

def load_files(file_path: str) -> List[str]:
    """Load text file lines."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        return [line.strip() for line in f if line.strip()]

def create_json(output_dir: str, baseline_file: str = None, 
                          json_output: str = None, update_mode: bool = False,
                          model_name: str = None) -> None:
    """
    Generate 2S_prompts.json with 2-shot examples and baseline outputs.
    
    Args:
        output_dir: Directory containing test.src, test.tgt, train.src, train.tgt, dev.src, dev.tgt
        baseline_file: Path to baseline model output file (e.g., baseline_LLaMA3_2.tgt)
        json_output: Output JSON file path
        update_mode: If True, only update baseline outputs in existing JSON
    """
    print("\n" + "=" * 80)
    if update_mode:
        print("UPDATING BASELINE OUTPUTS IN 2S_prompts.json")
    else:
        print("GENERATING 2S_prompts.json WITH 2-SHOT EXAMPLES")
    print("=" * 80)
    
    # Set default paths
    # Set default paths based on model_name
    if json_output is None:
        if model_name:
            json_output = os.path.join(output_dir, f"2S_prompts_{model_name}.json")
        else:
            json_output = Paths.JSON if hasattr(Paths, 'JSON') else os.path.join(output_dir, "2S_prompts.json")
    
    if baseline_file is None:
        # Map model names to config paths
        baseline_map = {
            'llama': Paths.LLAMA_0,
            'gpt': Paths.GPT_0,
            'gemma': Paths.GEMMA_0
        }
        baseline_file = baseline_map.get(model_name) if model_name else None
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
    
    # Load all files
    # Load all files using paths from config
    test_src = load_files(Paths.TEST_SRC if hasattr(Paths, 'TEST_SRC') else os.path.join(output_dir, "test.src"))
    train_src = load_files(Paths.TRAIN_SRC if hasattr(Paths, 'TRAIN_SRC') else os.path.join(output_dir, "train.src"))
    train_tgt = load_files(Paths.TRAIN_TGT if hasattr(Paths, 'TRAIN_TGT') else os.path.join(output_dir, "train.tgt"))
    dev_src = load_files(Paths.DEV_SRC if hasattr(Paths, 'DEV_SRC') else os.path.join(output_dir, "dev.src"))
    dev_tgt = load_files(Paths.DEV_TGT if hasattr(Paths, 'DEV_TGT') else os.path.join(output_dir, "dev.tgt"))
    
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
        if pair["source"] not in excluded_sentences #and len(pair["source"].split()) <= 80
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


def regenerate_splits_from_indices(output_dir: str, tsv_path: str) -> None:
    """
    Regenerate .src, .tgt files from existing indices WITHOUT creating new splits.
    Preserves the existing random splits by reading from indices files.
    
    Args:
        output_dir: Directory containing the split indices files (fallback)
        tsv_path: Path to the corpus TSV (source of truth for sentences)
    """
    print("\n" + "=" * 80)
    print("REGENERATING SPLIT FILES FROM EXISTING INDICES")
    print("=" * 80)
    print("⚠️  This will OVERWRITE .src and .tgt files but keep the same splits")
    
    # Load the full corpus TSV
    df = pd.read_csv(tsv_path, encoding="utf-8", sep="\t", on_bad_lines='warn')
    print(f"✓ Loaded corpus TSV with {len(df)} sentences\n")
    
    # Process each split
    for split_name in ['test', 'train', 'dev']:
        # *** FIX: Get indices path from same directory as .src file ***
        # Use config paths if available
        if split_name == "test":
            src_file = Paths.TEST_SRC if hasattr(Paths, 'TEST_SRC') else os.path.join(output_dir, f"{split_name}.src")
        elif split_name == "train":
            src_file = Paths.TRAIN_SRC if hasattr(Paths, 'TRAIN_SRC') else os.path.join(output_dir, f"{split_name}.src")
        elif split_name == "dev":
            src_file = Paths.DEV_SRC if hasattr(Paths, 'DEV_SRC') else os.path.join(output_dir, f"{split_name}.src")
        
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
        save_splits(df_split, split_name, output_dir, indices, df)
        
        print(f"  ✓ Regenerated {split_name}.src and {split_name}.tgt\n")
    
    print("✅ All split files regenerated successfully!")
    print("   The splits remain identical (same indices used)")
    
def create_norm_files(output_dir: str, tsv_path: str) -> None:
    """
    Generate verticalized .norm files for train, dev, and test splits using TSV metadata.
    
    Args:
        output_dir: Directory containing the split indices files
        tsv_path: Path to the corpus TSV (contains all metadata)
    """
    print("\n" + "=" * 80)
    print("GENERATING .norm FILES FOR SPLITS")
    print("=" * 80)
    
    corpus_dir = Paths.EXTRACT_DIR
    
    # Load the full dataframe (this IS our metadata)
    df = pd.read_csv(tsv_path, encoding="utf-8", sep="\t", on_bad_lines='warn')
    print(f"✓ Loaded corpus TSV with {len(df)} sentences as metadata source")
    
    
    # Process each split
    for split_name in ['train', 'dev', 'test']:
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
            norm_file_path = os.path.join(corpus_dir, f"{corpus_name}.norm")
            
            if corpus_name not in norm_cache:
                if not os.path.exists(norm_file_path):
                    print(f"\n⚠️  Error: {norm_file_path} not found")
                    continue
                
                with open(norm_file_path, 'r', encoding='utf-8') as f:
                    norm_cache[corpus_name] = [line.rstrip('\n') for line in f]
            
            norm_lines = norm_cache[corpus_name]
            
            # Convert from 1-indexed file line numbers to 0-indexed array indices
            sentence_lines = norm_lines[line_start - 1:line_end]
            
            # *** FIX: Preserve all lines including blank lines ***
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

def main(tsv_path: str = Paths.EXTRACT_TSV,
         output_dir: str = Paths.SET_SPLITS,
         test_size: float = DataSplits.TEST,
         dev_size: float = DataSplits.DEV,
         random_seed: int = 42,
         create_mode: str = "interactive"):
    """
    Main function to create dataset splits using line-number-based sampling.
    
    Args:
        tsv_path: Path to full corpus TSV
        output_dir: Where to save split files
        test_size: Proportion for test set
        dev_size: Proportion for dev set (of total data)
        random_seed: For reproducibility
        create_mode: 'all', 'test', 'train', 'dev', 'json', 'update_json', or 'interactive'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle JSON generation modes
    if create_mode == "json":
       create_json(output_dir, update_mode=False)
       return
    
    if create_mode == "update_json":
        create_json(output_dir, baseline_file=baseline_file, update_mode=True)
        return

    if create_mode == "norm":
        create_norm_files(output_dir, tsv_path)
        return

    if create_mode == "regenerate":
        regenerate_splits_from_indices(output_dir, tsv_path)
        return
    
    # Load corpus
    print(f"Loading corpus from {tsv_path}...")
    df = pd.read_csv(tsv_path, encoding="utf-8", sep="\t", on_bad_lines='warn')
    total_sentences = len(df)
    
    print(f"\nTotal sentences: {total_sentences:,}")
    print(f"TSV line numbers: 0 to {total_sentences - 1} (header excluded)")
    print(f"\nTarget splits:")
    print(f"  Test set: {test_size*100}% = {int(total_sentences * test_size):,} sentences")
    print(f"  Dev set: {dev_size*100}% = {int(total_sentences * dev_size):,} sentences")
    print(f"  Train set: {(1-test_size-dev_size)*100:.1f}% = {int(total_sentences * (1-test_size-dev_size)):,} sentences")
    
    # Check for existing test set
    existing_test_indices = load_indices(output_dir)
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
            print("5. Generate 2S_prompts.json (2-shot prompting)")
            print("6. Update baseline outputs in 2S_prompts.json")
            print("7. Generate .norm files for splits")
            print("8. Regenerate .src/.tgt from existing indices (no new splits)")  # ADD THIS
        else:
            print("1. Create all sets (test, train, dev)")
            print("2. Create only test set")
            print("3. Create test and train sets")
            print("4. Create test and dev sets")
            print("5. Generate 2S_prompts.json (2-shot prompting)")
            print("6. Update baseline outputs in 2S_prompts.json")
            print("7. Generate .norm files for splits")
            print("8. Regenerate .src/.tgt from existing indices (no new splits)")  # ADD THIS
        
        choice = input("\nEnter your choice (1-8): ").strip()  # CHANGE to 1-8
        
        if choice == "5":
            print("\nSelect model:")
            print("1. LLaMA")
            print("2. GPT")
            print("3. Gemma")
            model_choice = input("Enter choice (1-3): ").strip()
            model_map = {'1': 'llama', '2': 'gpt', '3': 'gemma'}
            model_name = model_map.get(model_choice)
            if model_name:
                create_json(output_dir, update_mode=False, model_name=model_name)
            return

        if choice == "6":
            print("\nSelect model:")
            print("1. LLaMA")
            print("2. GPT")
            print("3. Gemma")
            model_choice = input("Enter choice (1-3): ").strip()
            model_map = {'1': 'llama', '2': 'gpt', '3': 'gemma'}
            model_name = model_map.get(model_choice)
            if model_name:
                create_json(output_dir, update_mode=True, model_name=model_name)
            return
        
        if choice == "7":
            create_norm_files(output_dir, tsv_path)
            return
        
        # ADD THIS:
        if choice == "8":
            regenerate_splits_from_indices(output_dir, tsv_path)
            
            # Ask if user wants to regenerate JSON too
            response = input("\nDo you want to regenerate 2S_prompts.json now? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                print("\nSelect model:")
                print("1. LLaMA")
                print("2. GPT")
                print("3. Gemma")
                model_choice = input("Enter choice (1-3): ").strip()
                model_map = {'1': 'llama', '2': 'gpt', '3': 'gemma'}
                model_name = model_map.get(model_choice)
                if model_name:
                    create_json(output_dir, update_mode=False, model_name=model_name)
            return

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
        create_train_dev_sets(df, df_remaining, output_dir, dev_size, test_size,
                              random_seed, set(test_indices))
        print("\n✅ All sets created successfully!")
        
    elif create_mode == "test":
        df_test, df_remaining, test_indices = create_test_set(df, output_dir, 
                                                              test_size, random_seed)
        
        response = input("\nDo you want to create train and dev sets now? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            create_train_dev_sets(df, df_remaining, output_dir, dev_size, test_size,
                                  random_seed, set(test_indices))
            print("\n✅ All sets created successfully!")
        else:
            print("\n✅ Test set created. You can create train/dev sets later.")
    
    elif create_mode == "train_dev":
        if not test_exists:
            print("Error: Test set must exist first. Run with 'test' mode.")
            return
        
        df_remaining = df.drop(list(existing_test_indices)).copy()
        create_train_dev_sets(df, df_remaining, output_dir, dev_size, test_size,
                              random_seed, existing_test_indices)
        print("\n✅ Train and dev sets created successfully!")
        
        # Ask if user wants to generate JSON
        response = input("\nDo you want to generate 2S_prompts.json now? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            baseline_file = input("Enter baseline file path (press Enter for default): ").strip()
            if not baseline_file:
                baseline_file = os.path.join(output_dir, "0shot.hyp")
            create_json(output_dir, baseline_file=baseline_file, update_mode=False)
    
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
            create_train_dev_sets(df, df_remaining, output_dir, dev_size, test_size,
                                  random_seed, excluded)
            print("\n✅ Train and dev sets created successfully!")
        else:
            # Create only train set (all remaining data)
            train_indices = df_remaining.index.tolist()
            df_train = df_remaining.copy()
            print("\n" + "=" * 80)
            print("SAVING TRAIN SET (ALL REMAINING DATA)")
            print("=" * 80)
            save_splits(df_train, "train", output_dir, train_indices, df)  # Added df parameter
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
            create_train_dev_sets(df, df_remaining, output_dir, dev_size, test_size,
                                  random_seed, excluded)
            print("\n✅ Dev and train sets created successfully!")
        else:
            # Create only dev set (all remaining data after test)
            dev_indices = df_remaining.index.tolist()
            df_dev = df_remaining.copy()
            print("\n" + "=" * 80)
            print("SAVING DEV SET (ALL REMAINING DATA)")
            print("=" * 80)
            save_splits(df_dev, "dev", output_dir, dev_indices, df)  # Added df parameter
            print("\n✅ Dev set created!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create stratified dataset splits using line-number-based sampling'
    )
    parser.add_argument('--tsv', default=Paths.EXTRACT_TSV,
                       help='Input TSV file')
    parser.add_argument('--output-dir', default=Paths.SET_SPLITS,
                       help='Output directory for splits')
    parser.add_argument('--test-size', type=float, default=DataSplits.TEST,
                       help='Test set proportion (default: 0.10)')
    parser.add_argument('--dev-size', type=float, default=DataSplits.DEV,
                       help='Dev set proportion (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--mode', choices=['all', 'test', 'train', 'dev', 
                                            'train_dev', 'json', 'update_json', 'norm', 
                                            'regenerate', 'interactive'],  # ADD 'regenerate'
                        default='interactive',
                        help='What to create (default: interactive)')
    args = parser.parse_args()
    
    main(
        tsv_path=args.tsv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        dev_size=args.dev_size,
        random_seed=args.seed,
        create_mode=args.mode
    )