# CHANGELOG
All notable changes to this project will be documented in this file.

## LEONIDE 1.0
Initial version of the corpus.

### Added
- transcanno-tei: initial transcription
- pepper-xml: initial version
- annis: initial version
- docs: initial documentation

## LEONIDE 1.1
First corpus update

### Corrections
- bugs with line breaks and paragraphs fixed in all versions
- added some missing values for author_L1 metadata
- corrected author_years_in_project metadata
- remove trans-hyphen annotations for line endings and join hyphenated words
- change some metadata field names for consistency with other corpora:
    - author_age_in_y1 -> author_age_at_production (changes per task_year)
    - organisation_language_of_instruction -> school_language
    - organisation_class_id -> school_class_id
- add some metadata fields, e.g.:
    - school_grade_level (6-8, depending on text)
- change some metadata coding:
    - author_L1 residual category now coded as "OTHER" instead of "X"

### Added
- txt: full text versions of original and corrected transcripts in txt format
- pepper-xml-lines: pepper xml version including annotations that show original line breaks of handwritten essays and corresponding hyphenation
- metadata: machine-readable metadata files for corpus, corpus design, authors (learners), texts, transcribers, annotators and tasks
