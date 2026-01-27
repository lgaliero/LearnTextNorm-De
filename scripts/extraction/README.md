# German Learner Corpus Extraction Pipeline

This module extracts and aligns sentence pairs from German learner corpora (LEONIDE, Kolipsi 1, Kolipsi 2), producing both TSV metadata files and verticalized NORM word-alignment files.

---

## 📁 Module Structure

### Core Entry Point

#### `extraction.py`
**Purpose:** Command-line interface and main execution entry point.

**Functionality:**
- Parses command-line arguments (corpora selection, output format, file limits)
- Configures logging to file
- Delegates processing to `pipeline.process_corpora()`
- Displays extraction summary statistics

**Usage:**
```bash
# Extract all configured corpora
python scripts/extraction.py

# Extract specific corpora
python scripts/extraction.py --corpora LEONIDE Kolipsi_1_L2

# Limit files for testing
python scripts/extraction.py --max-files 5 --format tsv

# Choose output format
python scripts/extraction.py --format norm  # or 'tsv', 'both'
```

---

## 📦 Package Modules (`scripts/extraction/`)

### Configuration & Data Models

#### `constants.py`
**Purpose:** Centralized configuration constants.

**Contents:**
- `QUOTE_CHARS`: Set of all quote character variants (`"`, `„`, `"`, `«`, `»`, etc.)
- `ABBREVIATIONS`: List of German abbreviations for sentence splitting (`z.B`, `u.a`, `d.h`, etc.)
- `ABBREV_PATTERNS_NORM`: Regex patterns for abbreviation detection during tokenization
- `ABBREV_PATTERN`: Compiled regex for fast abbreviation matching

**Why This Matters:** German has complex abbreviation rules that can incorrectly trigger sentence boundaries. These patterns ensure abbreviations like "z. B." (zum Beispiel) aren't split into separate sentences.

---

#### `data_models.py`
**Purpose:** Core data structures used throughout the pipeline.

**Classes:**

**1. `SentencePair`** (dataclass)
```python
@dataclass
class SentencePair:
    src: str                          # Original learner text
    tgt: str                          # Corrected/normalized text
    has_correction: bool              # Whether src differs from tgt
    has_foreign: bool                 # Contains foreign language words
    orth_mappings: List[Tuple[str, str]]  # (original, corrected) word pairs
```
- Represents a single aligned sentence pair with metadata
- `orth_mappings` stores specific word-level corrections for precise NORM alignment
- Used as the fundamental unit passed between extraction stages

**2. `TextBuilder`**
```python
class TextBuilder:
    def __init__(self): self.parts = []
    def add_text(text: str, merge: bool = False)
    def add_space()
    def add_marker(marker: str)  # e.g., <SENTBREAK>
    def get_text() -> str
```
- Accumulates text fragments while preserving XML-specified spacing
- Critical for handling mid-word insertions, compound words split across tags
- The `merge` parameter handles cases like `Jugend<insertion>-</insertion>herberge`

**Why This Matters:** XML tags can appear mid-word. Naive text extraction would insert unwanted spaces. TextBuilder respects the original document's whitespace intent.

---

### Corpus-Specific Extraction (The Heavy Lifters)

#### `kolipsi.py` (~700 lines)
**Purpose:** Extracts text and corrections from Kolipsi corpus XML.

**Core Function:** `extract_kolipsi(element) -> (src_text, tgt_text, has_corrections, orth_mappings)`

**Handles XML Tags:**
- `<error>`, `<over_capitalisation>`: Orthographic/capitalization errors
  - Extracts both `<originalForm>` and `<targetForm>`
- `<palimpsest>`: Overwritten/corrected text
- `<strikeover>`: Strikethrough corrections with `<expansion>` children
- `<correction>`: Generic corrections with `<deletion>` and `<insertion>`
- `<reduction>`: Abbreviated forms (e.g., "u." → "und")
- `<ambiguous>`: Multiple possible readings (takes first `<alternative>`)
- `<foreign_word>`: Non-German text (marked for exclusion)

**Special Logic:**
- **Sentence boundary detection:** Identifies case changes after punctuation to split sentences
- **Compound word handling:** Merges text split across multiple tags (e.g., `Obst-` + `laden`)
- **Whitespace preservation:** Uses XML's leading/trailing whitespace to determine merge behavior

**Example:**
```xml
<error>
  <originalForm>schreiben</originalForm>
  <targetForm>schreibe</targetForm>
</error>
```
→ Produces: `src="schreiben"`, `tgt="schreibe"`, `orth_mappings=[("schreiben", "schreibe")]`

---

#### `leonide.py` (~700 lines)
**Purpose:** Extracts text and corrections from LEONIDE corpus XML.

**Core Function:** `extract_leonide(paragraph, all_paragraphs) -> (src_text, tgt_text, has_corrections, orth_mappings)`

**Handles XML Tags:**
- `<orth_error>`: Spelling errors with `orth_error_target` attribute
  - Handles split words across paragraphs using `tagcode` attribute
- `<tran_capitalisation>`: Capitalization errors
- `<tran_word_correction>`, `<tran_word_insertion>`: Word-level edits
- `<tran_word_deletion>`: Deleted words (excluded from output)
- `<tran_reduction>`: Abbreviations/contractions (e.g., "Ms." → "Ms")
- `<tran_ambiguous>`: Uncertain readings
- `<tran_foreign_word>`: Foreign language content

**Special Logic:**
- **Continuation detection:** `tagcode` attribute links split words across paragraph boundaries
```xml
  <!-- Paragraph 1 -->
  <orth_error tagcode="E001" orth_error_target="Sprachenoberschule">Sprachen</orth_error>
  <!-- Paragraph 2 -->
  <orth_error tagcode="E001">oberschule</orth_error>
```
  Result: Only adds target once, not twice
  
- **Nested structure handling:** Recursively processes deeply nested corrections
- **Sentence break injection:** Adds `<SENTBREAK>` markers when case changes signal new sentences

**Why Two Separate Files?**
- Different XML schemas (TEI-based vs custom)
- Different error annotation philosophies
- Each ~700 lines of complex, corpus-specific logic
- Separation improves maintainability and testing

---

### Sentence Construction & Splitting

#### `pair_builder.py`
**Purpose:** Converts raw extracted text into aligned sentence pairs.

**Class:** `PairBuilder` (static methods)

**Methods:**

**1. `from_kolipsi(element) -> List[SentencePair]`**
- Calls `extract_kolipsi()` to get full text
- Splits on `<SENTBREAK>` markers into chunks
- Strips quotes before sentence splitting (to avoid quote-induced splits)
- Passes chunks to `spacy_sent()` for sentence boundary detection
- Restores quotes after splitting
- Handles foreign word detection per sentence
- Returns list of `SentencePair` objects

**2. `from_leonide(paragraph, all_paragraphs) -> List[SentencePair]`**
- Calls `extract_leonide()` to get full text
- Ignores unreliable `<SENTBREAK>` markers from `<div>` tags
- Uses spaCy-based splitting exclusively
- Handles quote restoration
- Filters `orth_mappings` to only include corrections in current sentence

**Why This Layer?** Separates extraction (getting text from XML) from sentence splitting (linguistic processing). Makes testing easier and logic clearer.

---

#### `spacy_utils_de.py`
**Purpose:** German-specific sentence boundary detection.

**Core Function:** `spacy_sent(text: str) -> List[str]`

**Pipeline:**
1. **Abbreviation Protection:**
   - Replaces `z.B.`, `u.a.`, `w. z. B.` with placeholder tokens (`ZBTOKEN0`, `ABBREVTOKEN1`)
   - Prevents spaCy from treating abbreviation periods as sentence endings

2. **Numbered List Handling:**
   - Detects patterns like `word 1) next` and inserts `<SPLIT>` markers
   - Forces sentence boundaries at list item transitions

3. **Ellipsis Protection:**
   - Detects ellipsis inside quotes (`„text... more"`) and protects from splitting
   - Replaces `...` with `ELLIPSISMARKER` temporarily

4. **Uppercase Coordinating Conjunctions:**
   - Splits at capitalized `Und`, `Aber`, `Oder` when followed by uppercase letter
   - German capitalization indicates sentence start

5. **spaCy Sentencizer:**
   - Applies spaCy's rule-based sentence splitter to protected text
   - Handles periods, exclamation marks, question marks

6. **Fragment Merging:**
   - Rejoins sentence fragments (lowercase starts after punctuation)
   - Respects forced boundaries from numbered lists (`<CHUNKEND>` markers)

7. **Restoration:**
   - Replaces placeholder tokens with original abbreviations
   - Restores ellipsis markers

**Critical Details:**
```python
# Handles complex abbreviations like "w. z. B." (wie zum Beispiel)
text = re.sub(r'\bw\.\s*z\.\s*[bB]\.?\b', zb_replacer, text)

# Forces split at period + space + uppercase AFTER abbreviations are protected
text = re.sub(r'(?<!\d)\.(?!\.)(?!<DOT>)\s+(?:\*\s+)?([A-ZÄÖÜ])', r'.<SPLIT>\1', text)
```

**Why Not Just Use spaCy Out-of-the-Box?**
German learner text has:
- Inconsistent capitalization
- Non-standard punctuation
- Abbreviated forms not in standard dictionaries
- Compound words split across lines

Custom preprocessing ensures accurate sentence boundaries despite these challenges.

---

### Text & XML Utilities

#### `text_utils.py`
**Purpose:** Text manipulation utilities for quote handling and tokenization.

**Key Functions:**

**1. Quote Handling:**
```python
strip_quotes_preserve_original(text) -> (original, stripped)
restore_quotes_to_sentence(original_chunk, stripped_chunk, stripped_sentence) -> restored_sentence
```
- **Why?** Quotes confuse sentence splitters. "sentence end." "next sentence." might split wrong.
- **How?** Strip quotes before splitting, map sentence positions back to original, restore quotes.

**Example:**
```
Input:  'Er sagte: "Das ist gut." Sie stimmte zu.'
Stripped: 'Er sagte: Das ist gut. Sie stimmte zu.'
Split: ['Er sagte: Das ist gut.', 'Sie stimmte zu.']
Restored: ['Er sagte: "Das ist gut."', 'Sie stimmte zu.']
```

**2. Tokenization:**
```python
tokenize_preserve_abbrev(text) -> List[str]
```
- Splits text into words while keeping abbreviations intact
- Separates punctuation: `hello.` → `['hello', '.']`
- Protects: `z.B.` → `['z.B.']` (NOT `['z', '.', 'B', '.']`)
- Used for NORM word-by-word alignment

---

#### `xml_helpers.py`
**Purpose:** XML parsing utilities.

**Key Functions:**

**1. `strip_namespace(tag: str) -> str`**
```python
# "{http://www.eurac.edu/transcanno}paragraph" → "paragraph"
```
- Removes XML namespace prefixes for simpler tag matching

**2. `has_leading_whitespace(text)`, `has_trailing_whitespace(text)`**
- Checks if XML text nodes start/end with whitespace
- **Critical** for determining whether to merge adjacent text or insert space

**3. `inject_spaces_between_tags(xml_string: str) -> str`**
```python
# Before: <tag>text</tag><tag>more</tag>
# After:  <tag>text</tag><SPACEWRAPPER> </SPACEWRAPPER><tag>more</tag>
```
- Inserts explicit space markers where XML has meaningful whitespace between tags
- Preserves original document spacing intent during extraction

**Why This Matters:**
XML like `<tag>Wort</tag> <tag>noch</tag>` has space between tags.
Without injection, extraction might produce `Wortnoch` instead of `Wort noch`.

---

### Output & Orchestration

#### `output_writers.py`
**Purpose:** Handles writing sentence pairs to output formats.

**Class:** `NormWriter`

**Format Produced (.norm files):**
```
word1_src    word1_tgt
word2_src    word2_tgt

word1_src    word1_tgt
...
```
- Tab-separated, one word pair per line
- Blank line separates sentences
- Used for sequence-to-sequence model training

**Key Methods:**
```python
with NormWriter(output_path) as writer:
    start_line = writer.start_sentence()
    writer.write_word_pair("schreiben", "schreibe")
    writer.write_word_pair(".", ".")
    writer.write_blank_line()
    writer.end_sentence(corpus, filename, sent_num, start_line)
    
    # Retrieve line mappings
    mappings = writer.get_all_mappings()  # {(corpus, file, sent_num): (start_line, end_line)}
```

**Why Track Line Numbers?**
- Enables efficient lookup: "Show me sentence 42 from file X"
- Links TSV metadata (sentence text) to NORM alignment (word pairs)
- Allows validation: extract sentence from NORM file and verify against TSV

---

#### `pipeline.py`
**Purpose:** Orchestrates the entire extraction pipeline.

**Core Classes/Functions:**

**1. `TextExtractor` Class:**
```python
extractor = TextExtractor(corpus_type="LEONIDE")
sentence_pairs = extractor.extract(xml_content)
```
- Parses XML, identifies corpus type
- Routes to appropriate extractor (`leonide.py` or `kolipsi.py`)
- Calls `PairBuilder` to convert text → sentence pairs
- Handles paragraph merging (incomplete sentences split across `<paragraph>` tags)

**2. `clean_sentence_pairs(pairs) -> List[SentencePair]`:**
Filters out unwanted content:
- Foreign language sentences
- Sentences with `@` symbols (email addresses)
- Arrows `->` (formatting artifacts)
- Very short sentences (≤4 words)
- Duplicate sentences
- Empty/punctuation-only sentences

**3. `process_file(xml_path, corpus_type) -> List[SentencePair]`:**
- Reads XML file
- Injects space wrappers
- Extracts sentence pairs
- Cleans pairs
- Returns cleaned list

**4. `process_corpora(corpus_configs, output_dir, output_format) -> DataFrame`:**
The main orchestration function:
1. Iterates through configured corpora
2. Finds all XML files in corpus directory
3. Processes each file → sentence pairs
4. Writes `.norm` files (if requested):
   - Word-by-word alignment with `NormWriter`
   - Complex logic handles:
     - Multi-word corrections (`Sprachen oberschule` → `Sprachenoberschule`)
     - Abbreviation corrections (`w. z. B` → `wie z.B.`)
     - Punctuation splitting (`word.` → `word` + `.`)
     - Mapping tracking for each correction
5. Builds pandas DataFrame with metadata:
   - Corpus name, language proficiency (L1/L2)
   - XML filename, sentence number
   - Source text, target text, correction status
   - Text type (picture story vs opinion)
   - Line numbers in .norm file
6. Writes `.tsv` file (if requested)
7. Returns DataFrame

**NORM Alignment Algorithm (Simplified):**
```
For each sentence pair:
    Tokenize source and target
    For each source word:
        IF word has correction mapping:
            IF mapping is multi-word (e.g., "Sprachen oberschule"):
                Write all source words → single target
            ELSE IF mapping is single-word:
                Write source word → target word
        ELSE:
            Write source word → target word (no correction)
    
    Write blank line (sentence separator)
    Record line numbers for this sentence
```

---

### Supporting Modules

#### `logger.py`
**Purpose:** Centralized logging configuration.

**Functionality:**
- Configures Python logging to write to file (not console)
- Provides `debug(msg)` helper function
- Used throughout extraction for detailed debugging

**Why Separate?** Import `debug()` in any module without circular dependencies.

---

## 🔄 Data Flow Through Pipeline
```
XML Files (LEONIDE, Kolipsi)
    ↓
[extraction.py] CLI argument parsing
    ↓
[pipeline.process_corpora] Orchestration
    ↓
[pipeline.process_file] Per-file processing
    ↓
[xml_helpers.inject_spaces_between_tags] Add space markers
    ↓
[pipeline.TextExtractor.extract] Parse XML
    ↓
┌─────────────────────────────────────┐
│  Corpus-Specific Extraction         │
│  [leonide.py] OR [kolipsi.py]       │
│  → Recursively traverse XML         │
│  → Build src/tgt text with TextBuilder │
│  → Track orth_mappings              │
└─────────────────────────────────────┘
    ↓
[pair_builder.PairBuilder] Text → Sentences
    ↓
[text_utils.strip_quotes_preserve_original] Remove quotes
    ↓
[spacy_utils_de.spacy_sent] Sentence splitting
    ↓
[text_utils.restore_quotes_to_sentence] Restore quotes
    ↓
[data_models.SentencePair] objects
    ↓
[pipeline.clean_sentence_pairs] Filtering
    ↓
┌─────────────────────────────────────┐
│  Output Writing                     │
│  [output_writers.NormWriter]        │
│  → Word-by-word alignment           │
│  → Line number tracking             │
│  DataFrame construction             │
│  → TSV metadata export              │
└─────────────────────────────────────┘
    ↓
Output Files:
  - corpus_name.norm (word alignments)
  - all_corpora.tsv (sentence metadata)
```

---

## 🎯 Key Design Decisions

### 1. **Why Two Extraction Passes?**
- **Pass 1:** Extract full text from XML (handles tags, spacing, corrections)
- **Pass 2:** Split text into sentences (linguistic processing)
- **Reason:** XML structure ≠ linguistic structure. Separating concerns makes each step testable.

### 2. **Why Strip Then Restore Quotes?**
- Quotes like `"` confuse sentence splitters
- Better to split on clean text, then map positions back
- Preserves original formatting while enabling accurate splitting

### 3. **Why Track `orth_mappings` Separately?**
- Enables precise word-level alignment in NORM files
- Without mappings: must guess which "schreiben" was corrected to "schreibe"
- With mappings: know exactly which word pair is a correction

### 4. **Why `TextBuilder` Instead of String Concatenation?**
```python
# Bad: Naive concatenation
text = elem.text + child.text + elem.tail  # Wrong spacing!

# Good: Respect XML spacing intent
builder = TextBuilder()
builder.add_text(elem.text)
if has_leading_whitespace(child.text):
    builder.add_space()
builder.add_text(child.text, merge=is_mid_word)
```

---

## 🧪 Testing Considerations

Each module can be tested independently:

- **`kolipsi.py`, `leonide.py`:** Unit test with XML snippets
- **`spacy_utils_de.py`:** Test sentence splitting with known edge cases
- **`text_utils.py`:** Test quote restoration with synthetic examples
- **`output_writers.py`:** Verify line number tracking accuracy
- **`pipeline.py`:** Integration tests with full XML files

---

## 📊 Output File Formats

### TSV Format (`all_corpora.tsv`)
```
corpus	lang_prof	xml_file	sent_num	src	tgt	corrected	text_type	line_start	line_end
LEONIDE	L2	file1.xml	1	Ich schreiben	Ich schreibe	True	picture story	1	3
```

### NORM Format (`LEONIDE.norm`)
```
Ich	Ich
schreiben	schreibe
.	.
```
- Line 1-3: Sentence 1
- Line 4: Blank separator
- Lines link back to TSV via `line_start`/`line_end` columns

---

## 🔧 Extending the Pipeline

### Adding a New Corpus:
1. Create `new_corpus.py` in `extraction/`
2. Implement `extract_new_corpus(element) -> (src, tgt, has_corrections, orth_mappings)`
3. Add to `PairBuilder` class: `@staticmethod def from_new_corpus(...)`
4. Update `TextExtractor.extract()` to route to new extractor
5. Add corpus config to `configs.py`

### Adding a New Output Format:
1. Create `NewFormatWriter` class in `output_writers.py`
2. Implement methods: `write_sentence()`, `write_word_pair()`, etc.
3. Add format option to `process_corpora()` output_format handling
4. Update CLI arguments in `extraction.py`

---

## 📝 Common Issues & Solutions

**Issue:** Sentences split incorrectly (e.g., at abbreviations)
**Solution:** Add abbreviation patterns to `constants.ABBREVIATIONS`

**Issue:** Compound words merged incorrectly
**Solution:** Check XML whitespace handling in `xml_helpers.inject_spaces_between_tags()`

**Issue:** Quotes appearing in wrong places
**Solution:** Debug `text_utils.restore_quotes_to_sentence()` position mapping

**Issue:** NORM alignment off by one word
**Solution:** Verify `orth_mappings` are correctly filtered in `pair_builder.py`

---

## 📚 Dependencies

- **spaCy:** Sentence boundary detection
- **pandas:** DataFrame manipulation and TSV export
- **Python 3.8+:** Dataclasses, type hints
- **Standard library:** `re`, `xml.etree.ElementTree`, `csv`, `logging`

---

## 🚀 Performance Notes

- **Sentence splitting** is the bottleneck (spaCy processing)
- **XML parsing** is fast (stdlib ElementTree)
- **NORM writing** is I/O bound
- Typical throughput: ~100-200 sentences/second
- For 10,000 sentences: ~1-2 minutes total

---

**Last Updated:** 27th January 2026
**Maintainer:** Lucia Galiero
**License:** [Your License]