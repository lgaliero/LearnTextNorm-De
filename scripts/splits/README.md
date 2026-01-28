## Module: Splits (`splits/`)

Creates stratified train/dev/test splits maintaining corpus proportions, plus utilities for generating few-shot prompts and NORM files.

### Files

- **`split_core.py`** - Core stratification logic (~180 lines)
  - `stratify_sample()` - Stratified sampling with exclusions
  - `check_proportions()` - Validate split distributions
  
- **`file_utils.py`** - File I/O for splits (~100 lines)
  - `save_splits()` - Save .src, .tgt, and _indices.tsv files
  - `load_indices()` - Load indices from existing splits
  - `load_tsv()` - Load corpus TSV file

- **`prompt_json.py`** - JSON generation for few-shot examples (~280 lines)
  - `create_json()` - Generate 2S_prompts.json with semantically similar examples
  - `get_nlp_model()` - Lazy load spaCy model
  - `get_tf_model()` - Lazy load sentence transformer model
  - `load_files()` - Load text files into lists
  - `tokenize_and_preprocess()` - Preprocess sentences with spaCy
  - Uses sentence transformers (all-MiniLM-L6-v2) for semantic similarity
  
- **`norm_utils.py`** - NORM file creation from splits (~140 lines)
  - `create_norm_files()` - Generate verticalized .norm files for each split
  - `regenerate_splits_from_indices()` - Rebuild split files from existing indices

- **`split_cli.py`** - CLI wrapper (~400 lines)
  - Orchestrates all splitting operations
  - Interactive and batch modes
  - Subcommands for each operation

### Usage

**As a CLI:**
```bash
# Interactive mode (recommended for first-time use)
python splits/split_cli.py interactive

# Create all splits at once
python splits/split_cli.py all \
    --tsv corpus.tsv \
    --output-dir output/splits \
    --test-size 0.10 \
    --dev-size 0.10 \
    --seed 42

# Create only test set
python splits/split_cli.py test --tsv corpus.tsv

# Create train and dev (test must exist)
python splits/split_cli.py train-dev --tsv corpus.tsv

# Generate few-shot prompt JSON
python splits/split_cli.py json \
    --model llama \
    --baseline output/baseline_llama.tgt \
    --output output/2S_prompts_llama.json

# Update baseline outputs in existing JSON (fast - no re-encoding)
python splits/split_cli.py json \
    --model llama \
    --baseline output/new_baseline.tgt \
    --update

# Create NORM files for alignment
python splits/split_cli.py norm \
    --extract-dir output/extraction \
    --splits train dev test

# Regenerate .src/.tgt from existing indices (preserves splits)
python splits/split_cli.py regenerate --tsv corpus.tsv
```

**As a library:**
```python
from splits import (
    stratify_sample, 
    save_splits, 
    load_tsv, 
    check_proportions,
    create_json,
    create_norm_files,
    load_files
)

# Load data
df = load_tsv("corpus.tsv")

# Create test set
df_test, test_indices = stratify_sample(
    df,
    sample_size=0.10,
    split_name="test",
    excluded_indices=set()
)

# Check proportions
check_proportions(df_test, df, "Test")

# Save split
paths_config = {
    'TEST_SRC': 'output/test.src',
    'TEST_TGT': 'output/test.tgt',
    'LLAMA_0': 'output/baseline_llama.tgt',
    'EXTRACT_DIR': 'output/extraction'
}
save_splits(df_test, "test", "output/", test_indices, df, paths_config)

# Generate few-shot prompt examples
create_json(
    output_dir="output/",
    baseline_file="output/baseline.tgt",
    model_name="llama",
    paths_config=paths_config
)

# Create NORM files for alignment
create_norm_files(
    output_dir="output/",
    tsv_path="corpus.tsv",
    extract_dir="output/extraction/"
)
```
