#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align LLM hypothesis to original tokenization with mismatch detection.
Detects completely different sentences (LLM refusals) and marks them.
"""

import sys
import editdistance
import re


def normalize_for_matching(token):
    """Normalize token for fuzzy matching"""
    return token.lower().strip().replace('"', '').replace("'", '')


def sentence_similarity(orig_tokens, hyp_sentence):
    """
    Calculate rough similarity between original and hypothesis.
    Returns a score between 0 and 1 (1 = identical, 0 = completely different)
    """
    # Tokenize hypothesis
    hyp_tokens = re.findall(r'\w+', hyp_sentence.lower())
    orig_words = [normalize_for_matching(t) for t in orig_tokens if re.match(r'\w', t)]
    
    if not orig_words or not hyp_tokens:
        return 0.0
    
    # Count matching words
    orig_set = set(orig_words)
    hyp_set = set(hyp_tokens)
    
    intersection = len(orig_set & hyp_set)
    union = len(orig_set | hyp_set)
    
    if union == 0:
        return 0.0
    
    jaccard = intersection / union
    return jaccard


def align_tokens_with_edits(orig_tokens, hyp_sentence, verbose=False):
    """
    Align hypothesis sentence to original tokens using edit distance.
    Returns (aligned_tokens, is_good_match)
    """
    # Tokenize hypothesis
    hyp_tokens = re.findall(r'\w+|[^\w\s]', hyp_sentence)
    
    if verbose:
        print(f"\nOriginal tokens ({len(orig_tokens)}): {orig_tokens}")
        print(f"Hypothesis tokens ({len(hyp_tokens)}): {hyp_tokens}")
    
    # Build alignment
    aligned = []
    hyp_idx = 0
    matches = 0
    
    for orig_idx, orig_token in enumerate(orig_tokens):
        orig_norm = normalize_for_matching(orig_token)
        
        if hyp_idx >= len(hyp_tokens):
            aligned.append(orig_token)
            if verbose:
                print(f"  [{orig_idx}] '{orig_token}' -> LAI (no more hyp)")
            continue
        
        # Try to find best match in a window
        best_match_idx = None
        best_match_score = float('inf')
        window_size = min(5, len(hyp_tokens) - hyp_idx)
        
        for offset in range(window_size):
            hyp_cand = hyp_tokens[hyp_idx + offset]
            hyp_norm = normalize_for_matching(hyp_cand)
            
            dist = editdistance.eval(orig_norm, hyp_norm)
            
            if dist == 0:
                best_match_idx = hyp_idx + offset
                best_match_score = dist
                break
            elif dist < best_match_score and dist <= max(2, len(orig_norm) // 3):
                best_match_idx = hyp_idx + offset
                best_match_score = dist
        
        if best_match_idx is not None:
            matched_token = hyp_tokens[best_match_idx]
            aligned.append(matched_token)
            hyp_idx = best_match_idx + 1
            if best_match_score <= 1:
                matches += 1
            if verbose:
                print(f"  [{orig_idx}] '{orig_token}' -> '{matched_token}' (dist={best_match_score})")
        else:
            aligned.append(orig_token)
            if verbose:
                print(f"  [{orig_idx}] '{orig_token}' -> LAI (no match)")
    
    # Determine if this is a good match
    match_ratio = matches / len(orig_tokens) if orig_tokens else 0
    is_good_match = match_ratio >= 0.3  # At least 30% of tokens should match closely
    
    return aligned, is_good_match


def load_vertical_file(filepath):
    """Load vertical format file (word\tword per line, blank line = sentence boundary)"""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        current_sent = []
        for line in f:
            line = line.rstrip('\n\r')  # Remove trailing newlines
            
            # Skip empty lines (sentence boundary)
            if not line.strip():
                if current_sent:  # Only append if we have content
                    sentences.append(current_sent)
                    current_sent = []
                continue
            
            # Split on tab and strip whitespace
            parts = line.split('\t')
            if len(parts) >= 2:
                current_sent.append((parts[0].strip(), parts[1].strip()))
        
        # Don't forget last sentence
        if current_sent:
            sentences.append(current_sent)
    
    return sentences

def load_hypothesis_sentences(filepath):
    """Load hypothesis sentences"""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip())
    return sentences


def write_verticalized_file(filepath, sentences):
    """Write aligned sentences"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for orig, hyp in sentence:
                f.write(f"{orig}\t{hyp}\n")
            f.write("\n")


def main():
    if len(sys.argv) < 4:
        print("Usage: python align_llm_to_original.py <original.norm> <hypothesis.txt> <output.norm> [--verbose] [--similarity-threshold FLOAT]")
        print("\nOptions:")
        print("  --verbose                     Show alignment details for first sentences")
        print("  --similarity-threshold FLOAT  Minimum similarity to attempt alignment (default: 0.2)")
        print("  --placeholder TEXT            Text to use for mismatched sentences (default: <MISMATCH>)")
        print("\nRequires: pip install editdistance")
        sys.exit(1)
    
    orig_file = sys.argv[1]
    hyp_file = sys.argv[2]
    out_file = sys.argv[3]
    verbose = '--verbose' in sys.argv
    
    # Parse options
    similarity_threshold = 0.2
    placeholder = "<MISMATCH>"
    
    for i, arg in enumerate(sys.argv):
        if arg == '--similarity-threshold' and i + 1 < len(sys.argv):
            similarity_threshold = float(sys.argv[i + 1])
        if arg == '--placeholder' and i + 1 < len(sys.argv):
            placeholder = sys.argv[i + 1]
    
    print(f"Loading original file: {orig_file}")
    orig_sentences = load_vertical_file(orig_file)
    print(f"Loaded {len(orig_sentences)} sentences")
    
    print(f"Loading hypothesis file: {hyp_file}")
    hyp_sentences = load_hypothesis_sentences(hyp_file)
    print(f"Loaded {len(hyp_sentences)} sentences")
    
    if len(orig_sentences) != len(hyp_sentences):
        print(f"\nERROR: Sentence count mismatch!")
        print(f"  Original: {len(orig_sentences)}")
        print(f"  Hypothesis: {len(hyp_sentences)}")
        sys.exit(1)
    
    # Align sentences
    aligned_sentences = []
    mismatches = []
    low_quality = []
    
    for i in range(len(orig_sentences)):
        orig_tokens = [pair[0] for pair in orig_sentences[i]]
        hyp_sentence = hyp_sentences[i]
        
        # Check if sentences are similar enough
        similarity = sentence_similarity(orig_tokens, hyp_sentence)
        
        if similarity < similarity_threshold:
            # Completely different sentences - use placeholder
            aligned = [(token, placeholder) for token in orig_tokens]
            mismatches.append(i + 1)
            
            if len(mismatches) <= 5:
                print(f"\n⚠ MISMATCH detected at sentence {i+1} (similarity: {similarity:.2f})")
                print(f"  Original: {' '.join(orig_tokens[:15])}{'...' if len(orig_tokens) > 15 else ''}")
                print(f"  Hypothesis: {hyp_sentence[:80]}{'...' if len(hyp_sentence) > 80 else ''}")
        else:
            # Try to align
            hyp_tokens, is_good_match = align_tokens_with_edits(
                orig_tokens, 
                hyp_sentence, 
                verbose=(verbose and i < 3)
            )
            
            if len(hyp_tokens) != len(orig_tokens):
                print(f"ERROR: Alignment length mismatch at sentence {i+1}")
                while len(hyp_tokens) < len(orig_tokens):
                    hyp_tokens.append(orig_tokens[len(hyp_tokens)])
                hyp_tokens = hyp_tokens[:len(orig_tokens)]
            
            aligned = list(zip(orig_tokens, hyp_tokens))
            
            if not is_good_match:
                low_quality.append(i + 1)
        
        aligned_sentences.append(aligned)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(orig_sentences)} sentences...")
    
    print(f"\nWriting to: {out_file}")
    write_verticalized_file(out_file, aligned_sentences)
    
    # Stats
    total = sum(len(s) for s in aligned_sentences)
    changed = sum(1 for s in aligned_sentences for o, h in s if o != h)
    mismatch_tokens = sum(1 for s in aligned_sentences for o, h in s if h == placeholder)
    
    print(f"\n{'='*70}")
    print(f"ALIGNMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Sentences: {len(aligned_sentences)}")
    print(f"Tokens: {total}")
    print(f"Changed: {changed} ({100*changed/total:.2f}%)")
    print(f"\nMismatched sentences: {len(mismatches)}")
    if mismatches:
        print(f"  (Using placeholder '{placeholder}' for {mismatch_tokens} tokens)")
        if len(mismatches) <= 20:
            print(f"  Sentence IDs: {mismatches}")
        else:
            print(f"  First 20: {mismatches[:20]}")
            print(f"  ... and {len(mismatches)-20} more")
    
    if low_quality:
        print(f"\nLow quality alignments: {len(low_quality)}")
        if len(low_quality) <= 20:
            print(f"  Sentence IDs: {low_quality}")
    
    print(f"\n{'='*70}")
    print(f"Run: python normEval.py --gold {orig_file} --pred {out_file}")
    
    if mismatches:
        print(f"\n⚠ WARNING: {len(mismatches)} sentences are mismatched!")
        print(f"These will score as 0% accuracy. Check your hypothesis file.")
        print(f"Consider filtering these out or fixing the LLM output.")


if __name__ == "__main__":
    main()