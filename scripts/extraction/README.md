### XML Extraction and Proecessing Pipeline

This module extracts and aligns sentence pairs from German learner corpora (LEONIDE, Kolipsi 1, Kolipsi 2), producing both TSV metadata files and verticalized NORM word-alignment files.

---

### 📁 Module Structure and Packages (`scripts/extraction/`)
### Configuration & Data Models
#### `constants.py`
**Centralized configuration constants.** 

Saves abbreviations and quote character variants so that they aren't split into separate sentences.

Adds a general pattern so that abbreviatoin are transcribed swiftly and correctly in the .norm files.

---

#### `sentencizer_de.py`
**Custom sentence boundary detection for German.**

Splits text into words and separates punctuation while keeping abbreviations (both standardized and non-) intact.

Covers the following issues from triggering erroneous sentence breaks:
- inconsistent puctuation and capitalization between src/tgt
- numbered lists
- ellipses
- compounds split across lines


---

### Text & XML Utilities
#### `text_utils.py`
**Text utilities for quote handling**

- Strip quotes before splitting, map sentence positions back to original, restore quotes. (Refinement needed)
  
- Crafted to elude inconsistent quote handling by standard sentence splitters. 

---

#### `xml_helpers.py`
**XML parsing utilities.**
- Preserves original document spacing intent during extraction.
  
- Removes XML namespace prefixes for simpler tag matching.
  
- Checks if XML text nodes start/end with whitespace.
  
- Inserts explicit space markers where XML has meaningful whitespace between tags.
  
- Aim is to prevent incorrect merging between tag content:

   e.g. `<tag>Wort</tag> <tag>noch</tag>` becomes `Wort noch` instead of `Wortnoch`


---

#### `data_models.py`

**Core data structures used throughout the pipeline.**

1. `SentencePair`
   
    - Used as the fundamental unit passed between extraction stages.
    - Represents a single aligned sentence pair with metadata.
    - Respects the original document's whitespace intent.
    - Stores specific word-level corrections for precise NORM alignment.
      e.g. `src="schreiben"`, `tgt="schreibe"`, `orth_mappings=[("schreiben", "schreibe")]`.
    
2. `TextBuilder`
   
    - Accumulates text fragments while preserving XML-specified spacing.
    - Critical for handling mid-word insertions, compound words split across tags.
    - The `merge` parameter handles cases like `Jugend<insertion>-</insertion>herberge`.

---

### Corpus-Specific Extraction
#### `kolipsi.py`
**Extracts text and corrections from Kolipsi corpus family XMLs.**
- Identifies case changes after punctuation marks to further split sentences.

- Merges text split across multiple tags (e.g., `Obst-` + `laden`)

Handles the following tags:
- `<error>`, `<over_capitalisation>`: Orthographic/capitalization errors
  - Extracts both `<originalForm>` and `<targetForm>`
- `<palimpsest>`: Overwritten/corrected text
- `<strikeover>`: Strikethrough corrections with `<expansion>` children
- `<correction>`: Generic corrections with `<deletion>` and `<insertion>`
- `<reduction>`: Abbreviated forms (e.g., "u." → "und")
- `<ambiguous>`: Multiple possible readings (takes first `<alternative>`)
- `<foreign_word>`: Non-German text (marked for exclusion)



---

#### `leonide.py`
**Extracts text and corrections from LEONIDE corpus XMLs.**
- `tagcode` attribute links split words across paragraph boundaries, preventing content duplication

```xml
  <!-- Paragraph 1 -->
  <orth_error tagcode="E001" orth_error_target="Sprachenoberschule">Sprachen</orth_error>
  <!-- Paragraph 2 -->
  <orth_error tagcode="E001">oberschule</orth_error>
```
Result:    
- Recursively processes deeply nested corrections
- Adds `<SENTBREAK>` markers when case changes in the target signal new sentences.

Handles the following tags:
- `<orth_error>`: Spelling errors with `orth_error_target` attribute
  - Handles split words across paragraphs using `tagcode` attribute
- `<tran_capitalisation>`: Capitalization errors
- `<tran_word_correction>`, `<tran_word_insertion>`: Word-level edits
- `<tran_word_deletion>`: Deleted words (excluded from output)
- `<tran_reduction>`: Abbreviations/contractions (e.g., "Ms." → "Ms")
- `<tran_ambiguous>`: Uncertain readings
- `<tran_foreign_word>`: Foreign language content (marked for exclusion)


---

### Output & Orchestration
#### `output_writers.py`
**Handles writing sentence pairs to output formats**.

`NormWriter`

**Produces .norm files:**

```txt
    word1_src    word1_tgt
    word2_src    word2_tgt
    
    word1_src    word1_tgt
```

- Tab-separated, one word pair per line
- Blank line separates sentences
- Used for sequence-to-sequence model training
- Currently displays multi-word corrections and abbreviations correction on the same line 
e.g. (`Sprachen oberschule` → `Sprachenoberschule`)
        (`w. z. B` → `wie z.B.`)


---

#### `pipeline.py`
**Orchestrates the entire extraction pipeline.**
**1. `TextExtractor` Class:**
```python
extractor = TextExtractor(corpus_type="LEONIDE")
sentence_pairs = extractor.extract(xml_content)
```
- Parses XML, routes to appropriate extractor (`leonide.py` or `kolipsi.py`)
- Calls `PairBuilder` to convert text → sentence pairs
- Handles paragraph merging (incomplete sentences split across `<paragraph>` tags)

**2. `clean_sentence_pairs(pairs) -> List[SentencePair]`:**
Filters out:
- Foreign language sentences
- Sentences with `@` symbols (email addresses)
- Arrows `->` (formatting artifacts)
- Very short sentences (≤4 words)
- Duplicate sentences
- Empty/punctuation-only sentences

**3. `process_file`:**
- Reads XML file with the utilites from `xml.helpers.py`
- Extracts sentence pairs and cleans them
- Returns cleaned list

**4. `process_corpora`:**
Main orchestration function:
- Processes each file to produce the sentence pairs
- Writes `.norm` files (if requested)
- Writes `.tsv` file with metadata (if requested)


---

### Supporting Modules

#### `logger.py`
**Centralized logging configuration.**
- Configures Python logging to write to file (not console)
- Provides `debug(msg)` helper function for import without cicrular dependencies.
- Used throughout extraction for detailed debugging

---


## 📚 Dependencies
- **spaCy:** Sentence boundary detection
- **pandas:** DataFrame manipulation and TSV export
- **Python 3.8+:** Dataclasses, type hints
- **Standard library:** `re`, `xml.etree.ElementTree`, `csv`, `logging`
---

**Last Updated:** 27th January 2026
**Maintainer:** Lucia Galiero
