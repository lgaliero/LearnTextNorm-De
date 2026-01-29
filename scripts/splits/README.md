## Module: Splits (`splits/`)

Creates stratified train/dev/test splits maintaining corpus proportions, plus utilities for generating few-shot prompts and NORM files.

### Files

- **`split_core.py`** - Core stratification logic 
  - `stratify_sample()` - Stratified sampling with exclusions
  - `check_proportions()` - Validate split distributions
  
- **`file_io.py`** - File I/O for splits 
  - `save_splits()` - Save .src, .tgt, and _indices.tsv files
  - `load_indices()` - Load indices from existing splits
  - `load_tsv()` - Load corpus TSV file

- **`prompt_json.py`** - JSON generation for few-shot examples 
  - `create_json()` - Generate 2S_prompts.json with semantically similar examples
  - `get_nlp_model()` - Lazy load spaCy model
  - `get_tf_model()` - Lazy load sentence transformer model
  - `load_files()` - Load text files into lists
  - `tokenize_and_preprocess()` - Preprocess sentences with spaCy
  
- **`norm_utils.py`** - NORM file creation from splits 
  - `create_norm_files()` - Generate verticalized .norm files for each split
  - `regenerate_splits_from_indices()` - Rebuild split files from existing indices

- **`norm_validator.py`** - NORM file validation and fixing 
  - `check_norm_file()` - Check single NORM file for formatting issues
  - `fix_norm_file()` - Fix formatting issues with optional backup
  - `validate_and_fix_norm_files()` - Batch validate/fix all NORM files in directory (supports recursive search)
  - `batch_validate_from_config()` - Batch validate using config paths (TRAIN_NORM, DEV_NORM, TEST_NORM)
  - `get_norm_statistics()` - Get statistics about NORM file structure
  - Detects: multi-column lines, space-separated format, multiple tabs, whitespace-only lines

### Usage

**As a library:**
```python
from splits import (
    stratify_sample, 
    save_splits, 
    load_tsv, 
    check_proportions,
    create_json,
    create_norm_files,
    validate_and_fix_norm_files,
    batch_validate_from_config,
    check_norm_file,
    fix_norm_file,
    get_norm_statistics
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
    'TEST_SRC': '../data/test/test.src',
    'TEST_TGT': '../data/test/test.tgt',
    'TEST_NORM': '../data/test/test.norm',
    'TRAIN_NORM': '../data/train/train.norm',
    'DEV_NORM': '../data/dev/dev.norm',
    'LLAMA_0': '../hypos/llama3-2/0shot.hyp',
    'EXTRACT_DIR': '../master_files'
}
save_splits(df_test, "test", "data/", test_indices, df, paths_config)

# Generate few-shot prompt examples
create_json(
    output_dir="data/2S_prompts",
    baseline_file="../hypos/{modelname}/0shot.hyp",
    model_name="{modelname}",
    paths_config=paths_config
)

# Create NORM files for alignment
create_norm_files(
    output_dir="../data",
    tsv_path="corpus.tsv",
    extract_dir="../master_files/",
    paths_config=paths_config
)

# Batch validate all NORM files using config paths (recommended)
issues = batch_validate_from_config(
    paths_config=paths_config,
    fix=True,  # Automatically fix issues
    backup=True  # Create .bak backups
)

# Validate NORM files in directory with recursive search
issues = validate_and_fix_norm_files(
    directory="../data",
    fix=True,  # Automatically fix issues
    backup=True,  # Create .bak backups
    recursive=True  # Search subdirectories
)

# Check single file
issues = check_norm_file("data/train/train.norm", verbose=True)
if sum(len(v) for v in issues.values()) > 0:
    fix_norm_file("data/train/train.norm", backup=True)

# Get file statistics
stats = get_norm_statistics("data/train/train.norm")
print(f"Sentences: {stats['sentences']}")
print(f"Word pairs: {stats['word_pairs']}")
```
---

**Last Updated:** 29th January 2026  
**Maintainer:** Lucia Galiero