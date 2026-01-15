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
    test_indices_file = os.path.join(output_dir, "test_indices.txt")
    if os.path.exists(test_indices_file):
        with open(test_indices_file, 'r') as f:
            return set(int(line.strip()) for line in f if line.strip())
    return set()

def save_splits(df_split: pd.DataFrame, split_name: str, 
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

def check_proportions(df_full: pd.DataFrame, df_split: pd.DataFrame, 
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

def create_test(df: pd.DataFrame, output_dir: str, test_size: float, 
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
    
    check_proportions(df, df_test, "Test")
    
    print("\n" + "=" * 80)
    print("SAVING TEST SET")
    print("=" * 80)
    save_splits(df_test, "test", output_dir, test_indices)
    
    return df_test, df_remaining, test_indices

def create_train_dev(df: pd.DataFrame, df_remaining: pd.DataFrame, output_dir: str, 
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
    
    check_proportions(df_remaining, df_train, "Train")
    check_proportions(df_remaining, df_dev, "Dev")
    
    print("\n" + "=" * 80)
    print("SAVING TRAIN AND DEV SETS")
    print("=" * 80)
    save_splits(df_train, "train", output_dir, train_indices)
    save_splits(df_dev, "dev", output_dir, dev_indices)

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

def generate_json(output_dir: str, baseline_file: str = None, 
                          json_output: str = None, update_mode: bool = False) -> None:
    """
    Generate examples.json with 2-shot examples and baseline outputs.
    
    Args:
        output_dir: Directory containing test.src, test.tgt, train.src, train.tgt, dev.src, dev.tgt
        baseline_file: Path to baseline model output file (e.g., baseline_LLaMA3_2.tgt)
        json_output: Output JSON file path
        update_mode: If True, only update baseline outputs in existing JSON
    """
    print("\n" + "=" * 80)
    if update_mode:
        print("UPDATING BASELINE OUTPUTS IN EXAMPLES.JSON")
    else:
        print("GENERATING EXAMPLES.JSON WITH 2-SHOT EXAMPLES")
    print("=" * 80)
    
    # Set default paths
    if json_output is None:
        json_output = os.path.join(output_dir, "examples.json")
    
    if baseline_file is None:
        baseline_file = os.path.join(output_dir, "baseline_LLaMA3_2.tgt")
    
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
    test_src = load_files(os.path.join(output_dir, "test.src"))
    test_tgt = load_files(os.path.join(output_dir, "test.tgt"))
    train_src = load_files(os.path.join(output_dir, "train.src"))
    train_tgt = load_files(os.path.join(output_dir, "train.tgt"))
    dev_src = load_files(os.path.join(output_dir, "dev.src"))
    dev_tgt = load_files(os.path.join(output_dir, "dev.tgt"))
    
    print(f"\nLoaded files:")
    print(f"  Test: {len(test_src)} sentences")
    print(f"  Train: {len(train_src)} sentences")
    print(f"  Dev: {len(dev_src)} sentences")
    
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
        if pair["source"] not in excluded_sentences and len(pair["source"].split()) <= 80
    ]
    
    candidate_sources = [pair["source"] for pair in candidate_pairs]
    
    print(f"\nCandidate pool: {len(candidate_pairs)} sentences (excluding test set, max 80 words)")
    
    # Process each test sentence
    results = []
    
    for idx, (test_sentence, test_target) in enumerate(zip(test_src, test_tgt)):
        if (idx + 1) % 100 == 0:
            print(f"Processing {idx + 1}/{len(test_src)}...")
        
        # Tokenize and preprocess
        preprocessed_test = tokenize_and_preprocess(test_sentence, nlp)
        preprocessed_candidates = [tokenize_and_preprocess(s, nlp) for s in candidate_sources]
        
        # Compute embeddings and similarity
        test_embedding = model.encode(preprocessed_test, convert_to_tensor=True)
        candidate_embeddings = model.encode(preprocessed_candidates, convert_to_tensor=True)
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
            "test_target": test_target,
            "examples": examples,
            "baseline_output": baseline_output
        }
        results.append(result_entry)
    
    # Write to JSON file
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ JSON file saved to: {json_output}")
    print(f"✓ Processed {len(results)} test sentences")
    print(f"✓ Baselines included: {sum(1 for r in results if r['baseline_output'] != 'to be added')}/{len(results)}")

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
        create_mode: 'all', 'test', 'train', 'dev', 'json', 'update_json', or 'interactive'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle JSON generation modes
    if create_mode == "json":
        baseline_file = input("Enter baseline file path (press Enter for default baseline_LLaMA3_2.tgt): ").strip()
        if not baseline_file:
            baseline_file = os.path.join(output_dir, "baseline_LLaMA3_2.tgt")
        generate_json(output_dir, baseline_file=baseline_file, update_mode=False)
        return
    
    if create_mode == "update_json":
        baseline_file = input("Enter baseline file path (press Enter for default baseline_LLaMA3_2.tgt): ").strip()
        if not baseline_file:
            baseline_file = os.path.join(output_dir, "baseline_LLaMA3_2.tgt")
        generate_json(output_dir, baseline_file=baseline_file, update_mode=True)
        return
    
    # Load corpus
    print(f"Loading corpus from {csv_path}...")
    df = pd.read_csv(csv_path, encoding="utf-8")
    total_sentences = len(df)
    
    print(f"\nTotal sentences: {total_sentences:,}")
    print(f"CSV line numbers: 0 to {total_sentences - 1} (header excluded)")
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
            print("5. Generate examples.json (2-shot prompting)")
            print("6. Update baseline outputs in examples.json")
        else:
            print("1. Create all sets (test, train, dev)")
            print("2. Create only test set")
            print("3. Create test and train sets")
            print("4. Create test and dev sets")
            print("5. Generate examples.json (2-shot prompting)")
            print("6. Update baseline outputs in examples.json")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "5":
            baseline_file = input("Enter baseline file path (press Enter for default): ").strip()
            if not baseline_file:
                baseline_file = os.path.join(output_dir, "baseline_LLaMA3_2.tgt")
            generate_json(output_dir, baseline_file=baseline_file, update_mode=False)
            return
        
        if choice == "6":
            baseline_file = input("Enter baseline file path (press Enter for default): ").strip()
            if not baseline_file:
                baseline_file = os.path.join(output_dir, "baseline_LLaMA3_2.tgt")
            generate_json(output_dir, baseline_file=baseline_file, update_mode=True)
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
        df_test, df_remaining, test_indices = create_test(df, output_dir, 
                                                              test_size, random_seed)
        create_train_dev(df, df_remaining, output_dir, dev_size, test_size,
                              random_seed, set(test_indices))
        print("\n✅ All sets created successfully!")
        
        # Ask if user wants to generate JSON
        response = input("\nDo you want to generate examples.json now? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            baseline_file = input("Enter baseline file path (press Enter for default): ").strip()
            if not baseline_file:
                baseline_file = os.path.join(output_dir, "baseline_LLaMA3_2.tgt")
            generate_json(output_dir, baseline_file=baseline_file, update_mode=False)
        
    elif create_mode == "test":
        df_test, df_remaining, test_indices = create_test(df, output_dir, 
                                                              test_size, random_seed)
        
        response = input("\nDo you want to create train and dev sets now? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            create_train_dev(df, df_remaining, output_dir, dev_size, test_size,
                                  random_seed, set(test_indices))
            print("\n✅ All sets created successfully!")
        else:
            print("\n✅ Test set created. You can create train/dev sets later.")
    
    elif create_mode == "train_dev":
        if not test_exists:
            print("Error: Test set must exist first. Run with 'test' mode.")
            return
        
        df_remaining = df.drop(list(existing_test_indices)).copy()
        create_train_dev(df, df_remaining, output_dir, dev_size, test_size,
                              random_seed, existing_test_indices)
        print("\n✅ Train and dev sets created successfully!")
        
        # Ask if user wants to generate JSON
        response = input("\nDo you want to generate examples.json now? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            baseline_file = input("Enter baseline file path (press Enter for default): ").strip()
            if not baseline_file:
                baseline_file = os.path.join(output_dir, "baseline_LLaMA3_2.tgt")
            generate_json(output_dir, baseline_file=baseline_file, update_mode=False)
    
    elif create_mode == "train":
        if not test_exists:
            df_test, df_remaining, test_indices = create_test(df, output_dir, 
                                                                  test_size, random_seed)
            excluded = set(test_indices)
        else:
            df_remaining = df.drop(list(existing_test_indices)).copy()
            excluded = existing_test_indices
        
        response = input("\nDo you want to create dev set too? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            create_train_dev(df, df_remaining, output_dir, dev_size, test_size,
                                  random_seed, excluded)
            print("\n✅ Train and dev sets created successfully!")
        else:
            # Create only train set (all remaining data)
            train_indices = df_remaining.index.tolist()
            df_train = df_remaining.copy()
            print("\n" + "=" * 80)
            print("SAVING TRAIN SET (ALL REMAINING DATA)")
            print("=" * 80)
            save_splits(df_train, "train", output_dir, train_indices)
            print("\n✅ Train set created!")
    
    elif create_mode == "dev":
        if not test_exists:
            df_test, df_remaining, test_indices = create_test(df, output_dir, 
                                                                  test_size, random_seed)
            excluded = set(test_indices)
        else:
            df_remaining = df.drop(list(existing_test_indices)).copy()
            excluded = existing_test_indices
        
        response = input("\nDo you want to create train set too? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            create_train_dev(df, df_remaining, output_dir, dev_size, test_size,
                                  random_seed, excluded)
            print("\n✅ Dev and train sets created successfully!")
        else:
            # Create only dev set (all remaining data after test)
            dev_indices = df_remaining.index.tolist()
            df_dev = df_remaining.copy()
            print("\n" + "=" * 80)
            print("SAVING DEV SET (ALL REMAINING DATA)")
            print("=" * 80)
            save_splits(df_dev, "dev", output_dir, dev_indices)
            print("\n✅ Dev set created!")


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
                       help='Dev set proportion (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--mode', choices=['all', 'test', 'train', 'dev', 
                                          'train_dev', 'json', 'update_json', 'interactive'],
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