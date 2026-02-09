## Module: Eval (`scripts/eval/`)

Evaluation scripts for text normalization systems. Computes baselines (LAI, MFR), word-level accuracy with Error Reduction Rate (ERR), and edit distance metrics (WER, CER). Includes alignment preprocessing for LLM outputs that may reformulate input.

---

### Intended Workflow

```
baseline.py (LAI/MFR) → llmAlign.py (LLM outputs only) → normEval.py → wer++.py
```

**Standard evaluation sequence:**
1. Compute **baselines** (LAI and MFR) using `baseline.py`
2. If evaluating LLMs: **align** outputs to original tokenization using `llmAlign.py`
3. Compute **accuracy and ERR** using `normEval.py`
4. Compute **WER/CER** with error breakdown using `wer++.py`

**Quick evaluation** (assumes data already aligned):
```bash
# Word-level metrics
python normEval.py --gold test.norm --pred system.norm

# Edit distance metrics
python wer++.py system.tgt gold.tgt           # WER
python wer++.py system.tgt gold.tgt --cer     # CER
```

---

### Usage Guide

#### 1. `baseline.py` - from the [MultiLexNorm project](https://bitbucket.org/robvanderg/multilexnorm)
**Computes Leave-As-Is (LAI) and Most-Frequent-Replacement (MFR) baselines.**

LAI copies input unchanged; MFR applies memorized corrections from training data.

**Usage:**

```bash
# LAI baseline - 10-fold cross-validation on training set
python baseline.py \
    --method LAI \
    --train ../data/train/train.norm

# MFR baseline - 10-fold cross-validation on training set
python baseline.py \
    --method MFR \
    --train ../data/train/train.norm

# LAI on development set
python baseline.py \
    --method LAI \
    --train ../data/train/train.norm \
    --dev ../data/dev/dev.norm

# MFR on test set with output file
python baseline.py \
    --method MFR \
    --train ../data/train/train.norm \
    --dev ../data/test/test.norm \
    --out ../hypos/mfr_test.norm

# Custom k-fold (default: 10)
python baseline.py \
    --method MFR \
    --train ../data/train/train.norm \
    --kfold 5
```

**Input format:** Vertical NORM files, separated by tabs, where blank line = sentence boundary

**Output:**
- Prints: Baseline acc.(LAI), Accuracy, ERR
- Optional: Writes predictions to file (`--out`)

---

#### 2. `llmAlign.py`
**Aligns LLM hypothesis to original tokenization with mismatch detection.**

Required preprocessing step for LLM outputs before evaluation. Detects complete reformulations/refusals and marks them with placeholder tokens.

**Usage:**

```bash
# Basic alignment
python llmAlign.py \
    ../data/test/test.norm \
    ../hypos/llama3-2/0shot.hyp \
    ../hypos/llama3-2/0shot.norm

# With verbose output (shows first 3 sentence alignments)
python llmAlign.py \
    ../data/test/test.norm \
    ../hypos/llama3-2/2shot.hyp \
    ../hypos/llama3-2/2shot.norm \
    --verbose

# Custom similarity threshold (default: 0.2)
python llmAlign.py \
    ../data/test/test.norm \
    ../hypos/gpt-oss/0shot.hyp \
    ../hypos/gpt-oss/0shot.norm \
    --similarity-threshold 0.15

# Custom placeholder for mismatches (default: <MISMATCH>)
python llmAlign.py \
    test.norm \
    hypothesis.hyp \
    output.norm \
    --placeholder "[SKIP]"
```

**Input:**
- **Original NORM file:** Gold standard with original tokenization
- **Hypothesis file:** Plain text, one sentence per line (LLM output)

**Output:**
- **Aligned NORM file:** Hypothesis words aligned to original tokens
- **Statistics:** Mismatch count, changed tokens, alignment quality

**Mismatch detection:**
- Compares word overlap using Jaccard similarity
- If similarity < threshold (default 0.2): marks as `<MISMATCH>`
- Use cases: content moderation refusals, complete reformulation, perspective shifts

**Dependencies:** `pip install editdistance`

---

#### 3. `normEval.py` - from the [MultiLexNorm project](https://bitbucket.org/robvanderg/multilexnorm)
**Computes word-level accuracy and Error Reduction Rate (ERR).**

Standard evaluation for normalization systems. ERR normalizes accuracy relative to LAI baseline.

**Usage:**

```bash
# Standard evaluation
python normEval.py \
    --gold ../data/test/test.norm \
    --pred ../hypos/llama3-2/0shot.norm

# Ignore capitalization (case-insensitive)
python normEval.py \
    --gold ../data/test/test.norm \
    --pred ../hypos/system.norm \
    --ignCaps

# Verbose mode (prints all errors)
python normEval.py \
    --gold ../data/test/test.norm \
    --pred ../hypos/system.norm \
    --verbose
```

**Input format:** Vertical NORM files (word\tword per line, blank line = sentence boundary)

**Output:**
```
Baseline acc.(LAI): 96.04 [Proportion of tokens already correct in input (lower bound)]
Accuracy:           79.14 [Proportion of tokens matching gold standard]
ERR:                -426.89  [Error Reduction Rate = (Accuracy - LAI) / (100 - LAI) × 100]
```

**ERR Breakdown:**  
  - 0%: No improvement over LAI
  - 100%: Perfect correction
  - Negative: System introduces more errors than it fixes

**Error handling:**
- Exits if sentence count mismatch between gold and pred
- Exits if word count mismatch within any sentence pair

---

#### 4. `wer++.py` - from [here](https://github.com/nsmartinez/WERpp)
**Computes Word Error Rate (WER) and Character Error Rate (CER) with detailed error breakdown.**

Calculates edit distance at word or character level. Provides top-N most frequent errors.

**Usage:**

```bash
# Word Error Rate (WER)
python wer++.py system.tgt reference.tgt

# Character Error Rate (CER)
python wer++.py system.tgt reference.tgt --cer

# Show top 10 most frequent errors
python wer++.py system.tgt reference.tgt -n 10

# WER with colored output (for terminal viewing)
python wer++.py system.tgt reference.tgt -c -n 10

# CER with top 20 errors
python wer++.py system.tgt reference.tgt --cer -n 20

# Ignore capitalization
python wer++.py system.tgt reference.tgt -e lower

# Verbose mode (shows all edit operations)
python wer++.py system.tgt reference.tgt -v
```

**Input format:** Plain text files, one sentence per line (i.e. the files produced by `scripts/api_inference.py` as they are)

**Output:**
```
WER: 21.05 (Ins: 64 Dels: 73 Subs: 5862 Ref: 28499 )
----------------------------------
Wer due to words words
----------------------------------
[Worst-01] 0.4983% 0.4983% - b'[<MISMATCH>@.]'
[Worst-02] 0.3404% 0.8386% - b'[.@,]'
...
```

**Options:**
- `-n NUM` / `--worst-events NUM`: Show top N most frequent errors
- `--cer`: Calculate character error rate instead of word error rate
- `-c` / `--colors`: Color-code error types (red=deletion, green=insertion, blue=substitution)
- `-v`: Verbose mode (display all edit operations)
- `-e FUNC` / `--equal-func FUNC`: Comparison function (`standard`, `lower`, `dummy`)
- `-i` / `--ignore-blank`: Ignore blank reference lines

**Error notation:**
- `[word1@word2]`: Substitution (word1 → word2)
- `[word]`: Deletion
- Green text: Insertion
- For CER: operates on individual characters


---

### Dependencies

- **Python 3.8+**
- **editdistance** (for `llmAlign.py`)
- **[sacrebleu](https://github.com/mjpost/sacrebleu)** to obtain chrF scores.
- **Standard library:** `re`, `codecs`, `optparse`

---

### Notes

**ERR Interpretation:**
- Positive ERR: System corrects some errors (higher is better)
- ERR = 0%: System performs same as LAI (no corrections)
- Negative ERR: System introduces more errors than it fixes (worse than doing nothing)
- Example: ERR = -426% means system introduced 4.26× more errors than existed

**Alignment Importance:**
- LLMs often reformulate instead of minimally correcting
- Without alignment, reformulated sentences would cause word count mismatches
- Mismatch detection (similarity < 0.2) marks severe reformulations
- Placeholder tokens ensure these failures are properly penalized in metrics

**Common Pitfalls:**
1. Running `normEval.py` on unaligned LLM output → sentence length mismatch error
2. Forgetting to convert NORM to TGT format for `wer++.py`
3. Comparing different sentence orders (ensure alignment between gold/pred)

---

**Last Updated:** 10th February 2026  
**Maintainer:** Lucia Galiero
