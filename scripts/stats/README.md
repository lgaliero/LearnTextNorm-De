## Module: Stats (`scripts/stats/`)

Computes corpus statistics from raw XML files (pre-extraction) and processed TSV files (post-extraction). Provides comparative analysis showing filtering impact and data retention.

---

### Raw Statistics (Pre-Extraction)

- **`raw_stats.py`** - XML corpus analysis before extraction
  - `count_sentences_in_xml(xml_path, corpus_type)` - Counts sentences via punctuation patterns
  - `count_tokens_in_xml(xml_path, tokenize_func)` - Tokenizes all XML text content
  - `compute_raw_stats(corpus_configs, tokenize_func)` - Main raw statistics computation
  - Walks XML directories recursively, skips hidden files and `.pretty` variants
  - Returns: files count, approximate sentences, src/tgt token counts (same for raw)

---

### Processed Statistics (Post-Extraction)

- **`processed_stats.py`** - TSV corpus analysis after extraction
  - `compute_processed_stats(tsv_path, tokenize_func)` - Main processed statistics computation
  - Reads TSV with pandas, groups by corpus
  - Tokenizes src/tgt columns separately for accurate counts
  - Computes text type distribution (narrative, argumentative, descriptive, instructive)
  - Computes correction status (corrected vs. uncorrected sentence pairs)
  - Filters corrupted rows (numeric text_type values)

---

### Display Functions

- **`display.py`** - Formatted console output
  - `display_raw_stats(stats)` - Pretty-prints raw XML statistics table
  - `display_processed_stats(stats)` - Pretty-prints processed TSV statistics with text type/correction breakdowns
  - `display_comparison(raw_stats, processed_stats)` - Side-by-side comparison with retention percentages
  - All tables use aligned columns with thousands separators
  - Shows totals, percentages, and filtering impact summary

---

### Usage Examples

**As a Module:**
```python
from stats import compute_corpus_statistics
from configs import Paths, ExtractionParams
from extraction.sentencizer_de import tokenize_for_stats

# Compute both raw and processed statistics
stats = compute_corpus_statistics(
    mode='both',
    corpus_configs=ExtractionParams.CORPORA,
    tsv_path=Paths.EXTRACT_TSV,
    tokenize_func=tokenize_for_stats,
    output_json='corpus_stats.json'
)

# Access results
raw_stats = stats['raw']
processed_stats = stats['processed']
```

**Direct Imports:**
```python
from stats import compute_raw_stats, compute_processed_stats
from stats import display_raw_stats, display_processed_stats
from extraction.sentencizer_de import tokenize_for_stats
from configs import ExtractionParams, Paths

# Raw only
raw = compute_raw_stats(ExtractionParams.CORPORA, tokenize_for_stats)
display_raw_stats(raw)

# Processed only
processed = compute_processed_stats(Paths.EXTRACT_TSV, tokenize_for_stats)
display_processed_stats(processed)
```

---

### Dependencies

- **pandas** - TSV loading and grouping
- **extraction.sentencizer_de** - Tokenization consistency with extraction pipeline
- **configs** - Paths and corpus configurations
- **Python 3.8+** - Type hints, Literal types
- **Standard library** - `os`, `re`, `json`, `xml.etree.ElementTree`, `argparse`

---

**Last Updated:** 29th January 2026  
**Maintainer:** Lucia Galiero
