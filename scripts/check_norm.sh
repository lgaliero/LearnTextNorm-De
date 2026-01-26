#!/bin/bash

# Script to check and fix .norm files for normEval.py compatibility
# Usage: ./norm_file_checker.sh <file.norm>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <file.norm> [--fix]"
    echo "  Check (and optionally fix) .norm file formatting for normEval.py"
    echo "  --fix: automatically fix issues and create .fixed file"
    exit 1
fi

INPUT_FILE="$1"
FIX_MODE=false

if [ "$2" == "--fix" ]; then
    FIX_MODE=true
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found"
    exit 1
fi

ISSUES_FOUND=false
TEMP_FILE="${INPUT_FILE}.tmp"

echo "Checking file: $INPUT_FILE"
echo "================================"

# Check 1: Lines with more than 2 tab-separated columns
echo -n "Checking for lines with >2 columns... "
MULTI_COL=$(awk -F'\t' 'NF > 2' "$INPUT_FILE")
if [ -n "$MULTI_COL" ]; then
    echo "FOUND ISSUES"
    ISSUES_FOUND=true
    echo "Lines with more than 2 columns:"
    echo "$MULTI_COL" | head -5
    [ $(echo "$MULTI_COL" | wc -l) -gt 5 ] && echo "... (showing first 5)"
    
    if [ "$FIX_MODE" = true ]; then
        echo "  → Fixing: Merging all-but-last columns as source, last as target"
        awk -F'\t' '
        NF > 2 {
            raw = $1
            for (i=2; i<=NF-1; i++) raw = raw " " $i
            print raw "\t" $NF
            next
        }
        { print }
        ' "$INPUT_FILE" > "$TEMP_FILE"
        mv "$TEMP_FILE" "$INPUT_FILE"
    fi
else
    echo "OK"
fi

# Check 2: Lines with spaces instead of tabs (detect space-separated format)
echo -n "Checking for space-separated (non-tab) lines... "
SPACE_SEP=$(awk -F'\t' 'NF == 1 && NF != 0 && $0 !~ /^[[:space:]]*$/' "$INPUT_FILE" | head -5)
if [ -n "$SPACE_SEP" ]; then
    echo "FOUND ISSUES"
    ISSUES_FOUND=true
    echo "Lines that may be space-separated instead of tab-separated:"
    echo "$SPACE_SEP"
    
    if [ "$FIX_MODE" = true ]; then
        echo "  → Fixing: Converting whitespace to single tab between columns"
        awk '{
            if (NF == 0) {
                print ""
            } else if (NF == 1) {
                print $0
            } else {
                $1=$1
                print
            }
        }' OFS='\t' "$INPUT_FILE" > "$TEMP_FILE"
        mv "$TEMP_FILE" "$INPUT_FILE"
    fi
else
    echo "OK"
fi

# Check 3: Multiple consecutive tabs
echo -n "Checking for multiple consecutive tabs... "
MULTI_TAB=$(grep -P '\t\t+' "$INPUT_FILE" || true)
if [ -n "$MULTI_TAB" ]; then
    echo "FOUND ISSUES"
    ISSUES_FOUND=true
    echo "Lines with multiple consecutive tabs:"
    echo "$MULTI_TAB" | head -5
    
    if [ "$FIX_MODE" = true ]; then
        echo "  → Fixing: Replacing multiple tabs with single tab"
        sed -i 's/\t\t*/\t/g' "$INPUT_FILE"
    fi
else
    echo "OK"
fi

# Check 4: Lines with whitespace-only content (not truly empty)
echo -n "Checking for whitespace-only lines... "
WHITESPACE_LINES=$(grep -n '^[[:space:]]\+$' "$INPUT_FILE" || true)
if [ -n "$WHITESPACE_LINES" ]; then
    echo "FOUND ISSUES"
    ISSUES_FOUND=true
    echo "Lines with only whitespace (should be completely empty):"
    echo "$WHITESPACE_LINES" | head -5
    
    if [ "$FIX_MODE" = true ]; then
        echo "  → Fixing: Converting whitespace-only lines to empty lines"
        sed -i '/^[[:space:]]*$/s/.*//g' "$INPUT_FILE"
    fi
else
    echo "OK"
fi

# Check 5: Lines with only one column (potential missing tab)
echo -n "Checking for single-column non-empty lines... "
SINGLE_COL=$(awk -F'\t' 'NF == 1 && $0 !~ /^[[:space:]]*$/' "$INPUT_FILE" | head -5)
if [ -n "$SINGLE_COL" ]; then
    echo "WARNING"
    echo "Found lines with only one column (may need manual review):"
    echo "$SINGLE_COL"
    echo "Note: These might be intentional single-word entries or errors"
else
    echo "OK"
fi

# Final summary
echo "================================"
if [ "$ISSUES_FOUND" = true ]; then
    if [ "$FIX_MODE" = true ]; then
        echo "✓ Issues found and fixed in: $INPUT_FILE"
        echo ""
        echo "Please verify the file and re-run the checker to confirm all issues are resolved."
    else
        echo "✗ Issues found in file"
        echo ""
        echo "Run with --fix flag to automatically fix these issues:"
        echo "  $0 $INPUT_FILE --fix"
    fi
    exit 1
else
    echo "✓ No issues detected - file is properly formatted"
    exit 0
fi