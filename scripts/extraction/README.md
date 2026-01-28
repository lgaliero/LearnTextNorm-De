## Module: Extraction (`scripts/extraction/`)

Extracts and aligns sentence pairs from German learner corpora (LEONIDE, Kolipsi 1, Kolipsi 2), producing TSV metadata and verticalized NORM word-alignment files.

---

### Configuration & Constants

- **`constants.py`** - Centralized configuration constants 
  - `ABBREV_VARIANTS` - Regex patterns for German abbreviations (w.z.B, u.s.w, etc.)
  - `ABBREVIATIONS` - Simplified patterns for quick lookups
  - `ABBREV_PATTERN` - Compiled regex for abbreviation matching
  - `QUOTE_CHARS` - Set of all quote character variants („, ", «, etc.)

---

### Text Processing

- **`sentencizer_de.py`** - Custom German sentence boundary detection 
  - `sentencizer(text)` - Splits text into sentences, handles abbreviations, numbered lists, ellipses, compounds
  - `tokenize_preserve_abbrev(text)` - Tokenizes while keeping abbreviations intact
  - `tokenize_for_stats(text)` - Simple spaCy tokenization for word counting

- **`text_utils.py`** - Quote handling utilities 
  - `strip_quotes_preserve_original(text)` - Returns (original, stripped) tuple
  - `restore_quotes_to_sentence(original, stripped, sentence)` - Maps sentence positions back to restore quotes
  - `restore_quotes_to_pair(pair, src_orig, src_strip, tgt_orig, tgt_strip)` - Restores quotes to SentencePair
  - `has_sentence_ending(text)` - Checks for sentence-ending punctuation

---

### XML Processing

- **`xml_helpers.py`** - XML parsing utilities 
  - `has_leading_whitespace(text)` - Checks if XML text node starts with whitespace
  - `has_trailing_whitespace(text)` - Checks if XML text node ends with whitespace
  - `strip_namespace(tag)` - Removes XML namespace prefixes for cleaner matching
  - `inject_spaces_between_tags(xml_string)` - Injects `<SPACEWRAPPER>` markers for meaningful whitespace

---

### Data Models

- **`data_models.py`** - Core data structures 
  - `SentencePair` (dataclass) - Aligned source/target pair with metadata (src, tgt, has_correction, has_foreign, orth_mappings)
  - `TextBuilder.add_text(text, merge)` - Adds text with intelligent spacing (merge=True for mid-word)
  - `TextBuilder.add_space()` - Explicitly adds space
  - `TextBuilder.add_marker(marker)` - Adds markers like `<SENTBREAK>`
  - `TextBuilder.get_text()` - Returns accumulated text with cleanup

---

### Corpus-Specific Extraction

- **`pair_builder.py`** - Sentence pair extraction orchestrator
  - `PairBuilder.from_kolipsi(element)` - Extracts pairs from Kolipsi XML elements
  - `PairBuilder.from_leonide(paragraph, all_paragraphs)` - Extracts pairs from LEONIDE paragraphs
  - Handles sentence splitting with `<SENTBREAK>` markers
  - Merges compound words split across tags (e.g., `Obst-` + `laden`)
  - Restores quotes immediately after sentence extraction

- **`kolipsi.py`** - Kolipsi corpus extraction
  - `extract_kolipsi(element)` - Main extraction function
  - Handles `<error>`, `<over_capitalisation>` - Orthographic/capitalization errors
  - Handles `<palimpsest>` - Overwritten text with nested corrections
  - Handles `<strikeover>` - Strikethrough corrections with `<expansion>` children
  - Handles `<correction>` - Generic corrections with `<deletion>` and `<insertion>`
  - Handles `<reduction>` - Abbreviated forms (e.g., "u." → "und")
  - Handles `<ambiguous>` - Multiple readings (takes first `<alternative>`)
  - Handles `<foreign_word>` - Non-German text (marked for exclusion)
  - Returns (src_text, tgt_text, has_corrections, orth_mappings)

- **`leonide.py`** - LEONIDE corpus extraction
  - `extract_leonide(paragraph, all_paragraphs)` - Main extraction function
  - Handles `<orth_error>` - Spelling errors with `orth_error_target` attribute
  - Handles `<tran_capitalisation>` - Capitalization errors
  - Handles `<tran_word_correction>`, `<tran_word_insertion>` - Word-level edits
  - Handles `<tran_word_deletion>` - Deleted words (excluded from output)
  - Handles `<tran_reduction>` - Abbreviations/contractions
  - Handles `<tran_ambiguous>` - Uncertain readings with nested structures
  - Handles `<tran_foreign_word>` - Foreign language content (marked for exclusion)
  - Uses `tagcode` attribute to link split words across paragraph boundaries
  - Returns (src_text, tgt_text, has_corrections, orth_error_mappings)

---

### Output Generation

- **`output_writers.py`** - File output handlers
  - `NormWriter.__enter__()/__exit__()` - Context manager for file handling
  - `NormWriter.write(text)` - Writes raw text to file
  - `NormWriter.write_word_pair(src_word, tgt_word)` - Writes tab-separated word pair
  - `NormWriter.write_blank_line()` - Writes sentence separator
  - `NormWriter.end_sentence(corpus, xml_file, sent_num, start_line, end_line)` - Records line mappings
  - Produces `.norm` files: tab-separated word pairs, blank lines between sentences

---

### Pipeline Orchestration

- **`pipeline.py`** - Main extraction pipeline
  - `TextExtractor.extract(xml_content)` - Routes to appropriate corpus extractor
  - `clean_sentence_pairs(pairs)` - Filters foreign words, arrows, @-symbols, short sentences, duplicates
  - `calculate_corpus_stats(pairs, corpus_name)` - Computes sentence/token counts
  - `display_corpus_statistics(raw_stats, filtered_stats, doc_counts)` - Pretty-prints before/after statistics
  - `process_file(xml_path, corpus_type)` - Processes single XML file
  - `process_corpora(corpus_configs, output_dir, max_files_per_corpus, output_format, compute_stats)` - Main orchestration function
  - Outputs: `.norm` files (verticalized word alignment), `.tsv` file (metadata with line mappings)
  - Handles paragraph merging for incomplete sentences split across `<paragraph>` tags
  - Implements sophisticated word alignment using `orth_mappings`:
    - Multi-word source → single target (e.g., "Sprachen oberschule" → "Sprachenoberschule")
    - Spaced abbreviations (e.g., "w. z. B" → "wie z.B.")
    - Many-to-one corrections (groups consecutive words mapping to same target)
    - Punctuation splitting for proper alignment

---

### Supporting Modules

- **`logger.py`** - Centralized logging
  - `debug(msg)` - Writes debug messages to file (not console)
  - Configured to write to `Paths.EXT_LOG_FILE`

---

### Dependencies

- **spaCy** - Sentence boundary detection
- **pandas** - DataFrame manipulation and TSV export
- **Python 3.8+** - Dataclasses, type hints
- **Standard library** - `re`, `xml.etree.ElementTree`, `csv`, `logging`

---

**Last Updated:** 28th January 2026  
**Maintainer:** Lucia Galiero