## Module 2: Splits (`splits/`)

Creates stratified train/dev/test splits maintaining corpus proportions.

### Files

- **`split_core.py`** - Core stratification logic
  - `stratify_sample()` - Stratified sampling with exclusions
  - `check_proportions()` - Validate split distributions
  
- **`file_utils.py`** - File I/O for splits
  - `save_splits()` - Save .src, .tgt, and _indices.tsv files
  - `load_indices()` - Load indices from existing splits
  - `load_tsv()` - Load corpus TSV file

### Usage

**As a library:**
```python
from splits import stratify_sample, save_splits, load_tsv, check_proportions

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
check_proportions(df_test, df, "test")

# Save split
save_splits(df_test, "test", "output/", test_indices, df)
```

**Note**: The original `data_splitter.py` also contained prompt JSON generation and NORM file creation logic. These would be extracted into:
- `splits/prompt_json.py` - JSON generation for few-shot examples
- `splits/norm_utils.py` - NORM file creation from splits

---
