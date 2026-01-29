#!/bin/bash
# =======================================================================================
# COMPREHENSIVE EVALUATION SCRIPT
# =======================================================================================
# Calls existing Python scripts (eval/baseline.py, normEval.py, llmAlign.py, wer++.py)
# and writes all results to eval_results.txt
# =======================================================================================

set -e  # Exit on error

# Check configs.py exists
if [ ! -f "configs.py" ]; then
    echo "ERROR: configs.py not found in current directory"
    exit 1
fi

# Extract paths from configs.py
eval $(python3 << 'EOF'
from configs import Paths
print(f"TRAIN_NORM='{Paths.TRAIN_NORM}'")
print(f"DEV_NORM='{Paths.DEV_NORM}'")
print(f"TEST_NORM='{Paths.TEST_NORM}'")
print(f"EVAL_FILE='{Paths.EVAL}'")
print(f"LLAMA_0='{Paths.LLAMA_0}'")
print(f"LLAMA_2='{Paths.LLAMA_2}'")
print(f"GPT_0='{Paths.GPT_0}'")
print(f"GPT_2='{Paths.GPT_2}'")
print(f"GEMMA_0='{Paths.GEMMA_0}'")
print(f"GEMMA_2='{Paths.GEMMA_2}'")
EOF
)

echo "========================================================================="
echo "  EVALUATION PIPELINE"
echo "========================================================================="
echo "Results file: $EVAL_FILE"
echo ""

# Initialize results file
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
cat > "$EVAL_FILE" << EOF
========================================================================
  TEXT NORMALIZATION EVALUATION RESULTS
  Generated: $TIMESTAMP
========================================================================

EOF

# Helper functions
log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

section_header() {
    cat >> "$EVAL_FILE" << EOF

========================================================================
  $1
========================================================================

EOF
}

subsection_header() {
    cat >> "$EVAL_FILE" << EOF

--- $1 ---

EOF
}

# =======================================================================================
# STEP 1: BASELINE EVALUATION (LAI and MFR)
# =======================================================================================

echo "[1/4] Running baseline evaluation (LAI, MFR)..."
section_header "STEP 1: BASELINE EVALUATION (LAI and MFR)"

# LAI - 10-fold CV on training data
if [ -f "$TRAIN_NORM" ]; then
    subsection_header "LAI Baseline (10-fold Cross-Validation on Training Data)"
    log "Computing LAI baseline (10-fold CV)..."
    python3 eval/baseline.py --method lai --train "$TRAIN_NORM" >> "$EVAL_FILE" 2>&1
else
    log "SKIP: $TRAIN_NORM not found"
fi

# MFR - 10-fold CV on training data
if [ -f "$TRAIN_NORM" ]; then
    subsection_header "MFR Baseline (10-fold Cross-Validation on Training Data)"
    log "Computing MFR baseline (10-fold CV)..."
    python3 eval/baseline.py --method mfr --train "$TRAIN_NORM" >> "$EVAL_FILE" 2>&1
else
    log "SKIP: $TRAIN_NORM not found"
fi

# LAI on Dev Set
if [ -f "$TRAIN_NORM" ] && [ -f "$DEV_NORM" ]; then
    subsection_header "LAI Baseline (Dev Set)"
    log "Computing LAI on dev set..."
    python3 eval/baseline.py --method lai --train "$TRAIN_NORM" --dev "$DEV_NORM" >> "$EVAL_FILE" 2>&1
fi

# MFR on Dev Set
if [ -f "$TRAIN_NORM" ] && [ -f "$DEV_NORM" ]; then
    subsection_header "MFR Baseline (Dev Set)"
    log "Computing MFR on dev set..."
    python3 eval/baseline.py --method mfr --train "$TRAIN_NORM" --dev "$DEV_NORM" >> "$EVAL_FILE" 2>&1
fi

# LAI on Test Set
if [ -f "$TRAIN_NORM" ] && [ -f "$TEST_NORM" ]; then
    subsection_header "LAI Baseline (Test Set)"
    log "Computing LAI on test set..."
    python3 eval/baseline.py --method lai --train "$TRAIN_NORM" --dev "$TEST_NORM" >> "$EVAL_FILE" 2>&1
fi

# MFR on Test Set
if [ -f "$TRAIN_NORM" ] && [ -f "$TEST_NORM" ]; then
    subsection_header "MFR Baseline (Test Set)"
    log "Computing MFR on test set..."
    python3 eval/baseline.py --method mfr --train "$TRAIN_NORM" --dev "$TEST_NORM" >> "$EVAL_FILE" 2>&1
fi

# =======================================================================================
# STEP 2: GOLD STANDARD METRICS
# =======================================================================================

echo "[2/4] Computing gold standard metrics (chrF, WER, CER)..."
section_header "STEP 2: GOLD STANDARD METRICS (Test Set)"

if [ -f "$TEST_NORM" ]; then
    # Create temporary files
    temp_src=$(mktemp)
    temp_tgt=$(mktemp)
    temp_src_sent=$(mktemp)
    temp_tgt_sent=$(mktemp)

    # Extract and convert to sentence format in one pass
awk -F'\t' '
    BEGIN {sent=""; has_content=0} 
    {
        # Strip whitespace
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
        
        if ($0 == "") {
            if (has_content && sent != "") {
                print sent
            }
            sent = ""
            has_content = 0
        } else if (NF >= 2) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
            if (sent != "") {
                sent = sent " " $1
            } else {
                sent = $1
            }
            has_content = 1
        }
    } 
    END {
        if (has_content && sent != "") {
            print sent
        }
    }' "$TEST_NORM" > "$temp_src_sent"

# Extract target sentences (column 2)
awk -F'\t' '
    BEGIN {sent=""; has_content=0} 
    {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
        
        if ($0 == "") {
            if (has_content && sent != "") {
                print sent
            }
            sent = ""
            has_content = 0
        } else if (NF >= 2) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            if (sent != "") {
                sent = sent " " $2
            } else {
                sent = $2
            }
            has_content = 1
        }
    } 
    END {
        if (has_content && sent != "") {
            print sent
        }
    }' "$TEST_NORM" > "$temp_tgt_sent"

    # chrF Score
    if command -v sacrebleu &> /dev/null; then
        subsection_header "chrF Score (Gold Standard)"
        log "Computing chrF score..."
        if command -v jq &> /dev/null; then
            sacrebleu "$temp_tgt_sent" -m chrf < "$temp_src_sent" | jq -r .score >> "$EVAL_FILE" 2>&1 || echo "N/A" >> "$EVAL_FILE"
        else
            sacrebleu "$temp_tgt_sent" -m chrf < "$temp_src_sent" >> "$EVAL_FILE" 2>&1 || echo "N/A" >> "$EVAL_FILE"
        fi
        echo "" >> "$EVAL_FILE"
    fi
    
    # WER with breakdown
    subsection_header "WER - Word Error Rate (Gold Standard)"
    log "Computing WER with top 10 error breakdown..."
    python3 eval/wer++.py "$temp_src_sent" "$temp_tgt_sent" -n 10 >> "$EVAL_FILE" 2>&1
    echo "" >> "$EVAL_FILE"
    
    # CER with breakdown
    subsection_header "CER - Character Error Rate (Gold Standard)"
    log "Computing CER with top 10 error breakdown..."
    python3 eval/wer++.py "$temp_src_sent" "$temp_tgt_sent" --cer -n 10 >> "$EVAL_FILE" 2>&1
    echo "" >> "$EVAL_FILE"
    
    # Cleanup
    rm -f "$temp_src" "$temp_tgt" "$temp_src_sent" "$temp_tgt_sent"
else
    log "SKIP: $TEST_NORM not found"
fi

# =======================================================================================
# STEP 3: LLM ALIGNMENT
# =======================================================================================

echo "[3/4] Aligning LLM hypotheses..."
section_header "STEP 3: LLM HYPOTHESIS ALIGNMENT"

# Define model configsurations
declare -A MODEL_configsS=(
    ["llama3-2_0shot"]="$LLAMA_0"
    ["llama3-2_2shot"]="$LLAMA_2"
    ["gpt-oss_0shot"]="$GPT_0"
    ["gpt-oss_2shot"]="$GPT_2"
    ["gemma_0shot"]="$GEMMA_0"
    ["gemma_2shot"]="$GEMMA_2"
)

# Align each model's hypotheses
for model_name in "${!MODEL_configsS[@]}"; do
    hypo_file="${MODEL_configsS[$model_name]}"
    
    if [ -f "$hypo_file" ]; then
        # Output: replace .hyp extension with .norm
        aligned_file="${hypo_file%.hyp}.norm"
        
        subsection_header "Aligning: $model_name"
        log "Aligning $hypo_file to $TEST_NORM..."
        
        # Create output directory if needed
        mkdir -p "$(dirname "$aligned_file")"
        
        # Run alignment
        python3 eval/llmAlign.py "$TEST_NORM" "$hypo_file" "$aligned_file" \
            --similarity-threshold 0.2 >> "$EVAL_FILE" 2>&1
        
        echo "" >> "$EVAL_FILE"
    else
        log "SKIP: $hypo_file not found"
    fi
done

# =======================================================================================
# STEP 4: LLM SYSTEM EVALUATION
# =======================================================================================

echo "[4/4] Evaluating LLM systems..."
section_header "STEP 4: LLM SYSTEM EVALUATION"

# Evaluate each model
for model_name in "${!MODEL_configsS[@]}"; do
    hypo_file="${MODEL_configsS[$model_name]}"
    aligned_file="${hypo_file%.hyp}.norm"
    
    if [ -f "$aligned_file" ]; then
        section_header "SYSTEM: $model_name"
        
        # Word-level evaluation (LAI, Accuracy, ERR)
        subsection_header "Word-Level Metrics (normEval.py)"
        log "Computing word-level metrics for $model_name..."
        python3 eval/normEval.py --gold "$TEST_NORM" --pred "$aligned_file" >> "$EVAL_FILE" 2>&1
        echo "" >> "$EVAL_FILE"
        
        # Create sentence-level files for WER/CER/chrF
        temp_pred=$(mktemp)
        temp_gold=$(mktemp)

        # Extract predictions
        awk -F'\t' '
            BEGIN {sent=""; has_content=0} 
            {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
                if ($0 == "") {
                    if (has_content && sent != "") print sent
                    sent = ""
                    has_content = 0
                } else if (NF >= 2) {
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
                    if (sent != "") sent = sent " " $2
                    else sent = $2
                    has_content = 1
                }
            } 
            END {if (has_content && sent != "") print sent}' "$aligned_file" > "$temp_pred"

        # Extract gold (same logic)
        awk -F'\t' '
            BEGIN {sent=""; has_content=0} 
            {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
                if ($0 == "") {
                    if (has_content && sent != "") print sent
                    sent = ""
                    has_content = 0
                } else if (NF >= 2) {
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
                    if (sent != "") sent = sent " " $2
                    else sent = $2
                    has_content = 1
                }
            } 
            END {if (has_content && sent != "") print sent}' "$TEST_NORM" > "$temp_gold"
                # WER with breakdown
        subsection_header "WER - Word Error Rate"
        log "Computing WER for $model_name..."
        python3 eval/wer++.py "$temp_pred" "$temp_gold" -n 10 >> "$EVAL_FILE" 2>&1
        echo "" >> "$EVAL_FILE"
        
        # CER with breakdown
        subsection_header "CER - Character Error Rate"
        log "Computing CER for $model_name..."
        python3 eval/wer++.py "$temp_pred" "$temp_gold" --cer -n 10 >> "$EVAL_FILE" 2>&1
        echo "" >> "$EVAL_FILE"
        
        # chrF
        if command -v sacrebleu &> /dev/null; then
            subsection_header "chrF Score"
            log "Computing chrF for $model_name..."
            if command -v jq &> /dev/null; then
                sacrebleu "$temp_gold" -m chrf < "$temp_pred" | jq -r .score >> "$EVAL_FILE" 2>&1 || echo "N/A" >> "$EVAL_FILE"
            else
                sacrebleu "$temp_gold" -m chrf < "$temp_pred" >> "$EVAL_FILE" 2>&1 || echo "N/A" >> "$EVAL_FILE"
            fi
            echo "" >> "$EVAL_FILE"
        fi
        
        # Cleanup
        rm -f "$temp_pred" "$temp_gold"
    else
        log "SKIP: $aligned_file not found (alignment may have failed)"
    fi
done

# =======================================================================================
# COMPLETION
# =======================================================================================

cat >> "$EVAL_FILE" << EOF

========================================================================
  EVALUATION COMPLETED
========================================================================
All evaluations completed at $(date '+%Y-%m-%d %H:%M:%S')
Results saved to: $EVAL_FILE

EOF

echo ""
echo "========================================================================="
echo "  EVALUATION COMPLETED"
echo "========================================================================="
echo "Results saved to: $EVAL_FILE"
echo ""
echo "To view results:"
echo "  cat $EVAL_FILE"
echo "  less $EVAL_FILE"
echo ""
echo "To extract specific metrics:"
echo "  grep 'WER:' $EVAL_FILE"
echo "  grep 'Accuracy:' $EVAL_FILE"
echo "  grep 'ERR:' $EVAL_FILE"
echo "========================================================================="