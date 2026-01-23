from typing import Tuple, List, Optional
from .constants import *

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

def has_leading_whitespace(text: Optional[str]) -> bool:
    """Check if text starts with whitespace in original XML."""
    return text is not None and len(text) > 0 and text[0].isspace()

def has_trailing_whitespace(text: Optional[str]) -> bool:
    """Check if text ends with whitespace in original XML."""
    return text is not None and len(text) > 0 and text[-1].isspace()

def strip_namespace(tag: str) -> str:
    """Remove XML namespace from tag."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def has_sentence_ending(text: str) -> bool:
    """Check if text ends with sentence-ending punctuation."""
    if not text:
        return False
    return bool(re.search(r'[.!?]\s*$', text.strip()))

def inject_spaces_between_tags(xml_string: str) -> str:
    """Inject explicit SPACEWRAPPER nodes for meaningful spaces in XML."""
    injected_count = 0
    
    def replacer_text_space_tag(match):
        nonlocal injected_count
        before_gt = match.group(1)
        text = match.group(2)
        spaces = match.group(3)
        after_lt = match.group(4)
        
        if '\n' in spaces or not text.strip():
            return match.group(0)
        
        injected_count += 1
        return f'{before_gt}{text}<SPACEWRAPPER> </SPACEWRAPPER>{after_lt}'
    
    result = re.sub(r'(>)([^<\n]*?\S[^<\n]*?)([ \t]+)(<)', replacer_text_space_tag, xml_string)
    
    def replacer_tag_space_tag(match):
        nonlocal injected_count
        before = match.group(1)
        space = match.group(2)
        after = match.group(3)
        
        if '\n' in space:
            return match.group(0)
        
        injected_count += 1
        return f'{before}<SPACEWRAPPER> </SPACEWRAPPER>{after}'
    
    result = re.sub(r'(>)([ \t]+)(<)', replacer_tag_space_tag, result)
    
    return result
