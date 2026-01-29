Folder containing all the modules and their respective execution entry points.
Most of them are also CLI wrappers for integration with the terminal.

Intended workflow: 
`extraction.py` > `tsv_update.py` (if manual changes are present) >  `data_maker.py`* > `api_inference.py` > eval.sh

*`data_maker.py` can be queried whenever different file types are needed
`corpus_stats.py`can be queried at anytime

### Usage Guide

#### 1. `extraction.py`
**Execution entry point for the `extraction` module.**


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


#### 2. `tsv_updater.py`
**Execution entry point for the `tsv_tools` module.**

**Usage**

```bash
# Update from all NORM files in directory
python tsv_tools/tsv_update.py batch-update \
    --directory output/extraction \
    --tsv-name all_corpora.tsv

# Update specific corpora
python tsv_tools/tsv_update.py update \
    --tsv-file output/all_corpora.tsv \
    --corpora LEONIDE Kolipsi_1_L2
```


#### 3. `data_maker.py`
**Execution entry point for the `splits` module.**

**Usage**

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

#### 4. `api_inference.py`

**Execution entry point for the `inference` module**

```bash
# Baseline mode (zero-shot)
python inference/api_query_cli.py --mode baseline --input test.src --model llama3.2

# 2-shot mode with examples from JSON
python inference/api_query_cli.py --mode 2-shot-json --json examples.json --input test.src --model gpt

# Interactive mode (no input file)
python inference/api_query_cli.py --mode baseline --model llama3.2
```

**Last Updated:** 29th January 2026  
**Maintainer:** Lucia Galiero