# LearnTextNorm-De

**Orthographic Normalization for German Learner Text: Corpus Processing, Evaluation Infrastructure, and LLM Assessment**

This repository contains the complete pipeline for extracting, processing, and evaluating text normalization systems on German learner corpora. It includes custom extraction tools for XML-encoded learner texts, stratified data splitting, LLM inference via API, and comprehensive evaluation scripts measuring accuracy, error reduction rate (ERR), word error rate (WER), character error rate (CER), and chrF.

**Context:** This work was conducted at EURAC Research (November 2025 - January 2026) to explore automation in learner text normalization evaluation, focusing on minimal-intervention orthographic correction while preserving grammatical errors.

---

## Repository Structure

```
LearnTextNorm-De/
├── corpora/                    # XML source files (obtain from publishers)
│   ├── LEONIDE/
│   ├── Kolipsi_1_L1/
│   ├── Kolipsi_1_L2/
│   └── Kolipsi_2/
│
├── scripts/                    # All modules and CLI tools
│   ├── extraction/            # Extraction module
│   │   ├── __init__.py
│   │   ├── config.py          # Corpus configurations
│   │   ├── parsers.py         # XML parsing for LEONIDE/Kolipsi
│   │   ├── sentencizer_de.py  # German sentence boundary detection
│   │   ├── pipeline.py        # Main extraction pipeline
│   │   └── README.md
│   │
│   ├── tsv_tools/             # TSV maintenance module
│   │   ├── __init__.py
│   │   ├── updater.py         # Sync TSV with NORM edits
│   │   ├── validator.py       # TSV format validation
│   │   └── README.md
│   │
│   ├── splits/                # Data splitting module
│   │   ├── __init__.py
│   │   ├── splitter.py        # Stratified sampling
│   │   ├── format_converters.py  # NORM/SRC/TGT/JSON generation
│   │   ├── validators.py      # NORM file validation
│   │   └── README.md
│   │
│   ├── inference/             # LLM inference module
│   │   ├── __init__.py
│   │   ├── prompts.py         # Zero-shot and few-shot prompts
│   │   ├── api_client.py      # API wrapper (LLaMA, GPT)
│   │   ├── batch_processor.py # Batch inference with rate limiting
│   │   └── README.md
│   │
│   ├── eval/                  # Evaluation module
│   │   ├── baseline.py        # LAI and MFR baselines
│   │   ├── llmAlign.py        # Align LLM output to original tokenization
│   │   ├── normEval.py        # Accuracy and ERR computation
│   │   ├── wer++.py           # WER/CER with error breakdown
│   │   └── README.md
│   │
│   ├── stats/                 # Statistics module
│   │   ├── __init__.py
│   │   ├── raw_stats.py       # Pre-extraction XML statistics
│   │   ├── processed_stats.py # Post-extraction TSV statistics
│   │   ├── display.py         # Formatted output
│   │   └── README.md
│   │
│   ├── extraction.py          # CLI: Run extraction pipeline
│   ├── tsv_update.py          # CLI: Update TSV from NORM edits
│   ├── data_maker.py          # CLI: Create splits and formats
│   ├── api_inference.py       # CLI: Run LLM inference
│   ├── stats.py               # CLI: Compute corpus statistics
│   └── README.md              # Module documentation
│
├── master_files/              # Central storage
│   ├── all_corpora.tsv        # Master TSV with metadata
│   └── *.norm                 # Individual corpus NORM files
│
├── data/                      # Generated splits
│   ├── train/
│   │   ├── train.norm         # Vertical format: source\target
│   │   ├── train.src          # Source sentences
│   │   └── train.tgt          # Target sentences
│   │   └── train_indices.tsv  # Sentence Metadata (original corpus, source\target, text types,          corrections)
│   ├── dev/
│   └── test/
│
├── hypos/                     # Model outputs
│   ├── llama3-2/
│   │   ├── 0shot.hyp          # Raw LLM output in 0 shot
│   │   └── 0shot.norm         # Aligned to original tokenization
│   │   ├── 2shot.hyp          # Raw LLM output in 2 shot
│   │   └── 2shot.norm         # Aligned to original tokenization
│   └── gpt-oss/
│
├── .gitignore
└── README.md
```

---

## Installation

**Requirements:**
- Python 3.8+
- spaCy with German model
- API access (optional, for LLM inference)


**Note:** All modules are in the `scripts/` directory. Import paths use `scripts.module_name` format.

**Dependencies:**

Core requirements:
- spacy>=3.8.11 (with de_core_news_sm model)
- pandas>=2.3.5
- numpy>=1.21.0
- lxml>=4.9.0
- editdistance>=0.6.0
- tqdm>=4.67.1


Optional (for API inference):
- requests>=2.32.5
- ollama (if using ollama API)

---

## Citation

If you use this repository or dataset, please cite:

```bibtex
@misc{galiero2026learntextnorm,
  author = {Galiero, Lucia},
  title = {Exploring Automation to Evaluate Orthographic Normalization of Learner Text: The Case of German as a L2},
  year = {2026},
  publisher = {EURAC Research},
  howpublished = {\url{https://github.com/lgaliero/LearnTextNorm-De}}
}
```

**Data Sources:**

```bibtex
@inproceedings{glaznieks2022leonide,
  title={LEONIDE–A Longitudinal Trilingual Corpus of Young Learners of Italian, German and English},
  author={Glaznieks, Aivars and Frey, Jennifer-Carmen and Stopfner, Maria and Zanasi, Lorenzo and Nicolas, Lionel},
  booktitle={Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  pages={2486--2497},
  year={2022}
}

@article{glaznieks2023kolipsi,
  title={The Kolipsi Corpus Family: Resources for Learner Corpus Research in Italian and German},
  author={Glaznieks, Aivars and Frey, Jennifer-Carmen and Abel, Andrea and Nicolas, Lionel and Vettori, Chiara},
  journal={Italian Journal of Computational Linguistics},
  volume={9},
  number={1},
  pages={67--86},
  year={2023}
}
```
---

## License

Code and data in this repository are intended for **Academic Use Only**. Redistribution is **not allowed**

**Note on Data:** The original XML corpus files are **not included** in this repository. To reproduce the full pipeline, please download the data from their CLARIN repository page and comply by the intended use license:

- Kolipsi 1 L1 & L2 (V 1.1.): https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/64
- Kolipsi 2 (V 1.1.): https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/66
- LEONIDE (V 1.1.): https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/25

Then, place XML files in `corpora/` directory according to structure and run the extraction pipeline.

---

## Acknowledgments

This work was conducted at **EURAC Research, Institute for Applied Linguistics** (November 2025 - January 2026) as part of a Language Technology Internship.

**Supervision:** Dr. Jennifer-Carmen Frey, Dr. Aivars Glaznieks (EURAC Research Institute for Applied Linguistics)

**Corpus Providers:**
- LEONIDE: Glaznieks et al. (2022)
- Kolipsi: Glaznieks et al. (2023)

**Method and Evaluation Framework:** Based on MultiLexNorm shared task (van der Goot et al., 2021) and the work described by [Krupaninen et al. (2023)](https://aclanthology.org/2023.findings-emnlp.923/)

---

## Contact

**Author:** Lucia Galiero  
**GitHub:** [@lgaliero](https://github.com/lgaliero)  
**Repository:** [LearnTextNorm-De](https://github.com/lgaliero/LearnTextNorm-De)

For questions or issues, please open a GitHub issue or contact via the repository.

---

**Last Updated:** February 11th, 2026