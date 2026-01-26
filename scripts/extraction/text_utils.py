from typing import Tuple, List, Optional
from .constants import *
from .data_models import SentencePair

def strip_quotes_preserve_original(text: str) -> Tuple[str, str]:
    """
    Strip all quotes from text but keep the original.
    Returns: (original_text, stripped_text)
    """
    stripped = ''.join(char for char in text if char not in QUOTE_CHARS)
    return text, stripped

def restore_quotes_to_sentence(original_chunk: str, stripped_chunk: str, stripped_sentence: str) -> str:
    """
    Restore quotes to a sentence by finding its position in the original text.
    Includes leading quotes before sentence AND trailing quotes after sentence-ending punctuation.
    
    Args:
        original_chunk: Original chunk text WITH quotes
        stripped_chunk: Same chunk WITH quotes stripped
        stripped_sentence: A sentence extracted from stripped_chunk
    
    Returns:
        The sentence with quotes restored
    """
    if not stripped_sentence:
        return stripped_sentence
    
    # Find where this sentence appears in the stripped chunk
    sent_start_in_stripped = stripped_chunk.find(stripped_sentence)
    if sent_start_in_stripped == -1:
        # Sentence not found - return as-is
        return stripped_sentence
    
    sent_end_in_stripped = sent_start_in_stripped + len(stripped_sentence)
    
    # Now map back to the original text to extract the span with quotes
    original_pos = 0
    stripped_pos = 0
    span_start_in_original = -1
    span_end_in_original = -1
    
    for i, char in enumerate(original_chunk):
        if char in QUOTE_CHARS:
            # Quote doesn't advance stripped_pos
            original_pos += 1
        else:
            # Regular character
            if stripped_pos == sent_start_in_stripped and span_start_in_original == -1:
                span_start_in_original = i
            
            if stripped_pos == sent_end_in_stripped - 1:
                span_end_in_original = i + 1
                # IMPORTANT: Don't break here - we need to capture trailing quotes
            
            original_pos += 1
            stripped_pos += 1
    
    if span_start_in_original >= 0 and span_end_in_original > 0:
        # CRITICAL: Extend span BACKWARDS to include any leading quotes (and skip over whitespace if needed)
        leading_quote_start = span_start_in_original
        
        # Step 1: Go back to capture quotes immediately before the first character
        while leading_quote_start > 0 and original_chunk[leading_quote_start - 1] in QUOTE_CHARS:
            leading_quote_start -= 1
        
        # Step 2: If we hit whitespace, check if there are quotes before that whitespace
        # This handles cases like `: "SIE` where sentence is ` SIE` and quote is between : and space
        temp_pos = leading_quote_start - 1
        while temp_pos >= 0 and original_chunk[temp_pos].isspace():
            temp_pos -= 1
        
        # If there are quotes just before the whitespace, include them
        while temp_pos >= 0 and original_chunk[temp_pos] in QUOTE_CHARS:
            leading_quote_start = temp_pos
            temp_pos -= 1
        
        # CRITICAL: Extend span FORWARDS to include any trailing quotes
        trailing_quote_end = span_end_in_original
        while trailing_quote_end < len(original_chunk) and original_chunk[trailing_quote_end] in QUOTE_CHARS:
            trailing_quote_end += 1
        
        return original_chunk[leading_quote_start:trailing_quote_end]
    
    return stripped_sentence
    
def restore_quotes_to_pair(pair: SentencePair, 
                          src_original: str, src_stripped: str,
                          tgt_original: str, tgt_stripped: str) -> SentencePair:
    """
    Restore quotes to a sentence pair.
    
    Args:
        pair: SentencePair with stripped text
        src_original: Original source chunk WITH quotes
        src_stripped: Source chunk WITHOUT quotes
        tgt_original: Original target chunk WITH quotes  
        tgt_stripped: Target chunk WITHOUT quotes
    """
    src_restored = restore_quotes_to_sentence(src_original, src_stripped, pair.src)
    tgt_restored = restore_quotes_to_sentence(tgt_original, tgt_stripped, pair.tgt)
    
    return SentencePair(
        src=src_restored,
        tgt=tgt_restored,
        has_correction=pair.has_correction,
        has_foreign=pair.has_foreign,
        orth_mappings=pair.orth_mappings
    )   

def has_sentence_ending(text: str) -> bool:
    """Check if text ends with sentence-ending punctuation."""
    if not text:
        return False
    return bool(re.search(r'[.!?]\s*$', text.strip()))

def tokenize_preserve_abbrev(text: str) -> List[str]:
    """Tokenize text, separating punctuation except for abbreviations."""
    # First, find and protect abbreviations with placeholders
    protected = text
    abbrev_map = {}
    counter = [0]
    
    def protect_match(match):
        placeholder = f'___ABBREV{counter[0]}___'
        abbrev_map[placeholder] = match.group(0)
        counter[0] += 1
        return placeholder
    
    # Protect abbreviations by matching them in order of specificity (longest first)
    for pattern in sorted(ABBREV_PATTERNS_NORM, key=len, reverse=True):
        protected = re.sub(pattern, protect_match, protected, flags=re.IGNORECASE)
    
    # CRITICAL: Build regex pattern from actual QUOTE_CHARS set
    quote_pattern = '|'.join(re.escape(q) for q in QUOTE_CHARS)
    
    # Now tokenize: separate punctuation from words
    # Add space before punctuation (except within placeholders)
    tokenized = re.sub(r'([a-zA-ZäöüÄÖÜß0-9_])([.,!?;:)\]])', r'\1 \2', protected)
    # Add space after punctuation
    tokenized = re.sub(r'([.,!?;:])([a-zA-ZäöüÄÖÜß0-9_])', r'\1 \2', tokenized)
    # Handle opening quotes/parentheses
    tokenized = re.sub(r'([\[(])([a-zA-ZäöüÄÖÜß0-9_])', r'\1 \2', tokenized)
    
    # CRITICAL FIX: Use .format() instead of f-strings to preserve backreferences
    # Handle quotes after punctuation: `: "` -> `: "`
    pattern1 = '([.,!?;:])({})'.format(quote_pattern)
    tokenized = re.sub(pattern1, r'\1 \2', tokenized)
    
    # Handle quotes before ANY character (not just letters)
    pattern2 = '({})(?=\\S)'.format(quote_pattern)
    tokenized = re.sub(pattern2, r'\1 ', tokenized)
    
    # Handle quotes after ANY character
    pattern3 = '(?<=\\S)({})'.format(quote_pattern)
    tokenized = re.sub(pattern3, r' \1', tokenized)
    
    # Split on whitespace
    tokens = tokenized.split()
    
    # Restore abbreviations in each token
    final_tokens = []
    for token in tokens:
        # Replace all placeholders in this token
        restored = token
        for placeholder, original in abbrev_map.items():
            restored = restored.replace(placeholder, original)
        final_tokens.append(restored)
    
    return final_tokens
