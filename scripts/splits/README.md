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

- **`data_maker.py`** - CLI wrapper 
  - Orchestrates all splitting operations
  - Interactive and batch modes
  - Subcommands for each operation

### Usage

**As a CLI:**
```bash
# Interactive mode (recommended for first-time use)
python data_maker.py interactive

# Create all splits at once
python data_maker.py all \
    --tsv corpus.tsv \
    --output-dir ../data \
    --test-size 0.10 \
    --dev-size 0.10 \
    --seed 42

# Create only test set
python data_maker.py test --tsv corpus.tsv

# Create train and dev (test must exist)
python data_maker.py train-dev --tsv corpus.tsv

# Generate few-shot prompt JSON
python data_maker.py json \
    --model llama \
    --baseline ../hypos/{modelname}/0shot.hyp \
    --output ../data/2S_prompts/{modelname}.json

# Update baseline outputs in existing JSON (fast - no re-encoding)
python data_maker.py json \
    --model {modelname} \
    --baseline ../hypos/{modelname}/0shot.hyp \
    --update

# Create NORM files for alignment
python data_maker.py norm \
    --extract-dir ../master_files \
    --splits train dev test

# Regenerate .src/.tgt from existing indices (preserves splits)
python data_maker.py regenerate --tsv corpus.tsv

# Validate NORM files (check formatting before inference)
# Option 1: Batch validate using config paths (recommended)
python data_maker.py validate --batch


# Option w: Validate specific directory
python data_maker.py validate \
    --directory ../data/train

# Validate and automatically fix issues (batch mode)
python data_maker.py validate \
    --batch \
    --fix

# Validate single NORM file
python data_maker.py validate \
    --file ../data/train/train.norm

# Get statistics about NORM file
python data_maker.py validate \
    --file ../data/train/train.norm \
    --stats
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

### Key Features

**Stratified Sampling:**
- Maintains proportions across multiple dimensions (corpus, text type, correction status)
- Uses line-number-based sampling for reproducibility
- Tracks indices to prevent overlap between splits
- Supports excluding specific indices for sequential split creation

**Few-Shot Prompts:**
- Uses sentence transformers (all-MiniLM-L6-v2) for semantic similarity
- Finds 2 most similar examples from train/dev sets for each test sentence
- Pre-encodes all candidates once for efficiency (batch size 256)
- Includes baseline outputs for iterative improvement
- Supports update mode to refresh baselines without re-computing similarities
- GPU acceleration automatically enabled when available

**NORM File Generation:**
- Preserves word-level alignment format from original corpus
- Extracts exact line ranges using metadata from TSV (line_start, line_end)
- Caches corpus files for efficient processing
- Maintains blank line separators for sentence boundaries

**NORM File Validation:**
- **Pre-inference validation** - Ensures files meet normEval.py requirements
- **Three validation modes**:
  1. **Batch mode** (recommended) - Uses config paths (TRAIN_NORM, DEV_NORM, TEST_NORM)
  2. **Directory mode** - Validates all .norm files in directory (supports recursive search)
  3. **Single file mode** - Validates individual NORM file
- **Five validation checks**:
  1. Multi-column lines (>2 tab-separated columns)
  2. Space-separated format (should be tab-separated)
  3. Multiple consecutive tabs (should be single tab)
  4. Whitespace-only lines (should be empty)
  5. Single-column lines (potential missing tabs)
- **Automatic fixing** with optional .bak backups
- **Recursive directory search** - finds NORM files in subdirectories
- **Statistics mode** - analyze NORM file structure (sentences, word pairs, etc.)

**Index Regeneration:**
- Rebuilds .src/.tgt files from existing _indices.tsv files
- Preserves exact splits when corpus TSV is updated
- Useful after TSV updates or corrections
- No new randomization - uses saved indices

**Interactive Mode:**
- Menu-driven interface for ease of use
- Detects existing splits and adjusts options accordingly
- Safe prompts before overwriting data
- Model selection for JSON generation
- NORM validation options

---

## Validation Workflow

**Recommended workflow before inference:**
```bash
# 1. Create splits
python data_maker.py all

# 2. Generate NORM files
python data_maker.py norm

# 3. Batch validate NORM files using config paths (CRITICAL - do this before inference!)
python data_maker.py validate --batch --fix

# 4. Verify fixes worked
python data_maker.py validate --batch

# 5. Proceed with inference
```

**Alternative workflow for custom directory structures:**
```bash
# 1-2. Same as above

# 3. Validate with recursive search (if NORM files are in subdirectories)
python data_maker.py validate --directory ../data --recursive --fix

# 4. Verify fixes worked
python data_maker.py validate --directory ../data --recursive

# 5. Proceed with inference
```

**Common validation issues and fixes:**

| Issue | Cause | Auto-fix |
|-------|-------|----------|
| Multi-column lines | Multiple corrections on same line | Merges all-but-last as source |
| Space-separated | Wrong delimiter used | Converts spaces to tabs |
| Multiple tabs | Formatting inconsistency | Collapses to single tab |
| Whitespace-only | Invisible characters in blank lines | Converts to empty lines |
| Single-column | Missing target word | Flags for manual review |

**Why validation matters:**
- normEval.py requires strict tab-separated format
- Malformed lines cause alignment errors
- Multi-word corrections must be on single line
- Blank lines must be truly empty (no spaces/tabs)

**Validation modes comparison:**

| Mode | Use Case | Command |
|------|----------|---------|
| Batch | Standard workflow with configs | `--batch` |
| Recursive | Custom directory structures | `--directory ../data --recursive` |
| Single file | Quick check of specific file | `--file train.norm` |
| Statistics | Analyze file structure | `--file train.norm --stats` |

---