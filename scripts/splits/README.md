## Module: Splits (`splits/`)

Creates stratified train/dev/test splits maintaining corpus proportions, plus utilities for generating few-shot prompts and NORM files.

### Files

- **`pipeline.py`** - Core stratification logic 
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
  - Uses sentence transformers (all-MiniLM-L6-v2) for semantic similarity
  
- **`norm_utils.py`** - NORM file creation from splits 
  - `create_norm_files()` - Generate verticalized .norm files for each split
  - `regenerate_splits_from_indices()` - Rebuild split files from existing indices

### CLI: `data_maker.py`

Central CLI wrapper that orchestrates all splitting operations. Uses defaults from `configs.py` (Paths and DataSplits classes).

### Usage

**As a CLI:**

```bash
# Interactive mode (recommended for first-time use)
# Uses defaults from configs.py
python data_maker.py interactive

# Or simply:
python data_maker.py

# Create all splits at once (with defaults from configs.py)
python data_maker.py all

# Override defaults if needed:
python data_maker.py all \
    --tsv ../master_files/all_corpora.tsv \
    --output-dir ../data \
    --test-size 0.10 \
    --dev-size 0.10 \
    --seed 42

# Create only test set (uses defaults)
python data_maker.py test

# Create train and dev (test must exist, uses defaults)
python data_maker.py train-dev

# Generate few-shot prompt JSON (defaults: llama model, paths from configs.py)
python data_maker.py json

# Specify model and paths:
python data_maker.py json \
    --model llama \
    --baseline ../hypos/llama3-2/0shot.hyp \
    --output ../data/2S_prompts/llama.json

# Update baseline outputs in existing JSON (fast - no re-encoding)
python data_maker.py json \
    --model llama \
    --baseline ../hypos/llama3-2/0shot_new.hyp \
    --update

# Create NORM files for alignment (uses defaults from configs.py)
python data_maker.py norm

# Specify custom extract directory:
python data_maker.py norm \
    --extract-dir ../master_files \
    --splits train dev test

# Regenerate .src/.tgt from existing indices (preserves splits, uses defaults)
python data_maker.py regenerate
```

**Configuration Defaults:**

All defaults are pulled from `configs.py`:
- `--tsv`: `Paths.EXTRACT_TSV` (default: `../master_files/all_corpora.tsv`)
- `--output-dir`: `Paths.SET_SPLITS` (default: `../data`)
- `--test-size`: `DataSplits.TEST` (default: `0.10`)
- `--dev-size`: `DataSplits.DEV` (default: `0.10`)
- `--seed`: `42` (hardcoded)

For JSON generation:
- `--baseline`: `Paths.LLAMA_0` (default: `../hypos/llama3-2/0shot.hyp`)
- `--output`: `Paths.LLAMA_JSON` (default: `../data/2S_prompts/llama.json`)
- `--model`: `llama` (default, choices: llama, gpt, gemma)

For NORM generation:
- `--extract-dir`: `Paths.EXTRACT_DIR` (default: `../master_files`)

**As a library:**

```python
from splits import (
    stratify_sample, 
    save_splits, 
    load_tsv, 
    check_proportions,
    create_json,
    create_norm_files,
    load_files,
    regenerate_splits_from_indices
)

# Load data
df = load_tsv("../master_files/all_corpora.tsv")

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
    'TRAIN_SRC': '../data/train/train.src',
    'TRAIN_TGT': '../data/train/train.tgt',
    'DEV_SRC': '../data/dev/dev.src',
    'DEV_TGT': '../data/dev/dev.tgt',
    'LLAMA_0': '../hypos/llama3-2/0shot.hyp',
    'GPT_0': '../hypos/gpt-oss/0shot.hyp',
    'GEMMA_0': '../hypos/gemma/0shot.hyp',
    'EXTRACT_DIR': '../master_files'
}
save_splits(df_test, "test", "../data/", test_indices, df, paths_config)

# Generate few-shot prompt examples
create_json(
    output_dir="../data/",
    baseline_file="../hypos/llama3-2/0shot.hyp",
    json_output="../data/2S_prompts/llama.json",
    model_name="llama",
    paths_config=paths_config
)

# Create NORM files for alignment
create_norm_files(
    output_dir="../data/",
    tsv_path="../master_files/all_corpora.tsv",
    extract_dir="../master_files/",
    splits=['train', 'dev', 'test']
)

# Regenerate splits from existing indices (useful if corpus.tsv updated)
regenerate_splits_from_indices(
    output_dir="../data/",
    tsv_path="../master_files/all_corpora.tsv",
    paths_config=paths_config
)
```

### Workflow

1. **First time setup**: Run `python data_maker.py all` to create test, train, and dev splits
2. **Generate prompts**: Run `python data_maker.py json --model llama` to create few-shot prompt JSON
3. **Create NORM files**: Run `python data_maker.py norm` to generate alignment-ready files
4. **Update baseline**: If you get new model outputs, run `python data_maker.py json --update --baseline new_outputs.hyp`
5. **Preserve splits**: If corpus changes but you want to keep the same splits, run `python data_maker.py regenerate`

### Split Preservation

The `regenerate` command is useful when:
- You've updated the corpus TSV with corrections/additions
- You want to keep the exact same train/dev/test split
- The `*_indices.tsv` files are preserved (these define which sentences belong to which split)

It will:
1. Load existing split indices from `{output_dir}/{split}/{split}_indices.tsv`
2. Load the updated corpus TSV
3. Regenerate `.src` and `.tgt` files using the original split indices
4. Preserve the exact same split boundaries