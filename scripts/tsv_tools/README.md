## Module: TSV Tools (`tsv_tools/`)

Manages batch updates of TSV files from edited NORM files.

### Files

- **`norm_parser.py`** - NORM file parsing
  - `parse_norm_file_simple()` - Parse NORM format into sentences
  - `normalize_text()` - Text normalization for comparison
  
- **`diff_core.py`** - Diff and alignment detection
  - `texts_similar()` - Fuzzy text comparison
  - `detect_operations()` - Detect edits, splits, merges, deletions
  - `calculate_operation_stats()` - Summarize detected changes
  
- **`apply_ops.py`** - Operation application
  - `apply_operations_to_corpus()` - Apply detected changes to DataFrame
  - `get_norm_path_for_corpus()` - Find NORM file for a corpus
  - `find_norm_files_in_directory()` - Discover NORM files
  
- **`batch_update.py`** - High-level orchestration
  - `batch_update_tsv()` - Main batch update function
  
- **`tsv_update_cli.py`** - CLI wrapper

### Usage

**As a CLI:**
```bash
# Update from all NORM files in directory
python tsv_tools/tsv_update_cli.py batch-update \
    --directory output/extraction \
    --tsv-name all_corpora.tsv

# Update specific corpora
python tsv_tools/tsv_update_cli.py update \
    --tsv-file output/all_corpora.tsv \
    --corpora LEONIDE Kolipsi_1_L2
```

**As a library:**
```python
from tsv_tools import batch_update_tsv, find_norm_files_in_directory

# Find NORM files
norm_files = find_norm_files_in_directory("output/extraction")

# Batch update
df_updated = batch_update_tsv(
    tsv_path="output/all_corpora.tsv",
    norm_files=norm_files,
    output_path="output/updated.tsv",
    log_edits=True
)
```

---
