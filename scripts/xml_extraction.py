import re
import os
import csv
import copy
import spacy
import argparse
import logging
import pandas as pd
from configs import Paths, ExtractionParams
from spacy.language import Language
from spacy.pipeline import Sentencizer
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# Compile pattern to de
# Global abbreviation patterns - used for both sentence splitting and NORM alignment
# Abbreviation patterns for NORM tokenization - match core abbreviation structure
ABBREV_PATTERNS_NORM = [
    r'w\.\s*z\.\s*[bB]\.?',   # w.z.B, w. z. B, w.z.B.
    r'[zZ]\.\s*[bB]\.?',       # z.B, z. B, z.B., Z.B
    r'[zZ][bB]\.?',            # zB, ZB, zB., ZB.
    r'u\.s\.w\.?',             # u.s.w, u.s.w.
    r'u\.n\.w\.?',             # u.n.w.
    r'u\.a\.?',                # u.a, u.a.
    r'd\.h\.?',                # d.h, d.h.
    r'c\.a\.?',                # c.a, c.a.
    r'o\.\s*[äÄ]\.?',          # o.ä, o. Ä, o.ä.
    r'o\.',                    # o. (standalone)
    r'[oO]\.[kK]\.?',          # o.k., O.K., o.k
    r'[U]\.?[A]\.?',            # U.A, U.A. etc.
    r'M\.S\.?',                # M.S, M.S.
    r'[A-ZÄÖÜ]\.',             # Single capital letter abbreviations: H., P., M., etc.
    r'Min\.', r'min\.', r'bzw\.', r'usw\.', r'etc\.', r'ecc\.', r'ca\.', r'evtl\.', 
    r'ggf\.', r'inkl\.', r'max\.', r'Nr\.', r'Tel\.', r'vs\.', 
    r'Mr\.', r'Mrs\.', r'Ms\.', r'Dr\.', r'Prof\.', r'Fam\.'
]


# Keep original ABBREVIATIONS for spacy_sent
ABBREVIATIONS = [
    r'bo\.\s',
    r'o\.\s?ä',
    r'o\.\sÄ',
    r'[zZ]\.\s?[bB]',
    r'\bw\.\s*z\.\s*[bB]\.?\b',
    r'\bu\.?s\.?w\.?\)?\b',
    r'u\.n\.w',
    r'u\.a',
    r'd\.h',
    r'c\.a',
    r'\b[oO]\.?[kK]\.?',
    r'P\.S',
    r'Min', r'min', r'bzw', r'usw', r'etc', r'ecc', r'ca', r'evtl', 
    r'ggf', r'inkl', r'max', r'Nr', r'Tel', r'vs', r'Mr', r'Mrs', 
    r'Ms', r'Dr', r'Prof', r'Fam'
]

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

# Compile pattern to detect any abbreviation with optional spacing
ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(ABBREVIATIONS) + r')\.?\b',
    re.IGNORECASE
)
# Create a blank German pipeline
nlp = spacy.blank("de")

# Add Sentencizer if not present
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# ============================================================================
# SHARED UTILITIES
# ============================================================================

@dataclass
class SentencePair:
    """Represents a source-target sentence pair with metadata."""
    src: str
    tgt: str
    has_correction: bool
    has_foreign: bool
    orth_mappings: List[Tuple[str, str, int]] = None
    
    def __post_init__(self):
        if self.orth_mappings is None:
            self.orth_mappings = []
    
    def to_tuple(self):
        return (self.src, self.tgt, self.has_correction, self.has_foreign)

# All possible quote characters
QUOTE_CHARS = {'"', '„', '"', '"', '«', '»', '‹', '›'}


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

class TextBuilder:
    """
    Handles text accumulation with proper spacing preservation.
    CRITICAL: Respects XML whitespace at all times.
    """
    def __init__(self):
        self.parts = []
    
    def add_text(self, text: str, merge: bool = False):
        """
        Add text with intelligent spacing.
        
        Args:
            text: Text to add
            merge: If True, merge directly without space (for mid-word situations)
        """
        if not text:
            return
        
        # Filter out "unreadable" literals
        text = re.sub(r'\bunreadable\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'unreadable', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        if not text:
            return
        
        if not self.parts:
            self.parts.append(text)
            return
        
        if merge:
            # Direct merge for mid-word cases
            self.parts.append(text)
        else:
            # Add space if last part doesn't end with one
            if self.parts[-1] and not self.parts[-1].endswith(' '):
                self.parts.append(' ')
            self.parts.append(text)
    
    def add_space(self):
        """Explicitly add a space."""
        if self.parts and not self.parts[-1].endswith(' '):
            self.parts.append(' ')
    
    def add_marker(self, marker: str):
        """Add a marker (like <SENTBREAK> or <FOREIGN>)."""
        self.parts.append(marker)
    
    def get_text(self) -> str:
        """Get accumulated text with cleanup."""
        text = ''.join(self.parts)
        # Clean up multiple spaces but preserve single spaces
        text = re.sub(r' +', ' ', text)
        # Preserve all original punctuation spacing
        return text.strip()

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

def spacy_sent(text: str) -> List[str]:
    """Split German text into sentences using spaCy."""
    debug("[DEBUG ENTER spacy_sent]")
    if not text or not text.strip():
        return []
    debug(f"[DEBUG ORIGINAL TEXT]: {text[:200]}")
    
    
    zb_map = {}
    zb_counter = [0]

    def zb_replacer(match):
        original = match.group(0)
        placeholder = f'ZBTOKEN{zb_counter[0]}'
        zb_map[placeholder] = original
        zb_counter[0] += 1
        return placeholder

    text = re.sub(r'\bw\.\s*z\.\s*[bB]\.?\b', zb_replacer, text, flags=re.IGNORECASE)
    
    text = re.sub(r'\b[zZ]\s*\.?\s*[bB]\.?\b', zb_replacer, text, flags=re.IGNORECASE)

    # Store ALL abbreviations to preserve original form
    abbrev_map = {}
    abbrev_counter = [0]

    def abbrev_replacer(match):
        original = match.group(0)
        placeholder = f'ABBREVTOKEN{abbrev_counter[0]}'
        abbrev_map[placeholder] = original
        abbrev_counter[0] += 1
        return placeholder

    # Protect multi-letter abbreviations with internal periods
    text = re.sub(r'\bu\.?s\.?w\.?\)?\b', abbrev_replacer, text, flags=re.IGNORECASE)

    text = re.sub(r'\bu\.n\.w\.', abbrev_replacer, text, flags=re.IGNORECASE)
    text = re.sub(r'\bu\.a\.', abbrev_replacer, text, flags=re.IGNORECASE)
    text = re.sub(r'\bd\.h\.', abbrev_replacer, text, flags=re.IGNORECASE)
    text = re.sub(r'\bc\.a\.', abbrev_replacer, text, flags=re.IGNORECASE)
    text = re.sub(r'\bo\.\s?ä\.', abbrev_replacer, text, flags=re.IGNORECASE)
    text = re.sub(r'\bo\.\sÄ\.', abbrev_replacer, text, flags=re.IGNORECASE)
    text = re.sub(r'\bo\.\s', abbrev_replacer, text, flags=re.IGNORECASE)
    text = re.sub(r'etc\.?\)?', abbrev_replacer, text, flags=re.IGNORECASE) 

    text = re.sub(r'\b[oO]\.?[kK]\.?', abbrev_replacer, text)  # Matches Ok, ok, O.K., o.k., etc.
    # Single-word abbreviations
    text = re.sub(r'\b(Min|min|bzw|usw|etc|ecc|ca|evtl|ggf|inkl|max|Nr|Tel|vs|Mr|Mrs|Ms|Dr|Prof|Fam|XXI|P\.S)\.', abbrev_replacer, text)

    
    # CRITICAL: Split at numbered markers IMMEDIATELY - before ANY other processing
    text = re.sub(r'(\S)\s*\d+\)\s*', r'\1<SPLIT>', text)


    # CRITICAL FIX: Protect ellipsis inside quotes from being treated as sentence boundary
    # Pattern: „ Text... WORD → should NOT split
    text = re.sub(r'(„[^"]*?)\.\.\.(\s+)([A-ZÄÖÜ])', r'\1ELLIPSISMARKER\2\3', text)

    # CRITICAL: Force split at period + space + uppercase (sentence boundaries)
    # This runs AFTER abbreviation protection, so z.B. is already safe as ZBTOKEN
    # Handle optional asterisks/bullets between period and uppercase
    text = re.sub(r'(?<!\d)\.(?!\.)(?!<DOT>)\s+(?:\*\s+)?([A-ZÄÖÜ])', r'.<SPLIT>\1', text)
    
    # Split at uppercase coordinating conjunctions  
    text = re.sub(r'(?<=[.!?\s])\s+(Und|Aber|Oder|Denn|Sondern|Doch)(?=\s+[A-ZÄÖÜ])', r' <SPLIT>\1', text)

    # Only split after sentence-ending punctuation + space + uppercase letter
    # But NOT after abbreviations or numbers
    # Check what comes after: if it's "und" or other lowercase text, it's likely a compound word, not sentence end
    text = re.sub(r'(?<!\d)\.(?!\.)(?!<DOT>)(\s+)(?=und\s)', r'\1', text)  # Remove period but keep space before "und" (compound word)
    # First, temporarily mark periods inside ZBPROTECT zones so they won't be split
    text = re.sub(r'(ZBPROTECT[^Z]*?)\.(.*?ZBPROTECTEND)', r'\1<ZBDOT>\2', text)
    # Now split at periods, but exclude <ZBDOT>
    text = re.sub(r'(?<!\d)\.(?!\.)(?!<DOT>)(?!<ZBDOT>)\s+([A-ZÄÖÜ])', r'.<SPLIT>\1', text)
    # Restore the protected periods
    text = text.replace('<ZBDOT>', '.')
    # For short words before "und", only split if word is longer than 5 characters
    text = re.sub(r'(?<!\d)(\w{6,})\.(?!\.)(?!<DOT>)\s+(und\s[A-ZÄÖÜ])', r'\1.<SPLIT>\2', text)

    chunks = text.split('<SPLIT>')
    all_sentences = []
    
    for chunk in chunks:
        if not chunk.strip():
            continue

        debug(f"[DEBUG CHUNK BEFORE PROCESSING]: '{chunk[:100]}'")
    
        # Remove numbered markers at the start of chunks (with word boundary protection)
        chunk = re.sub(r'^\d+\)\s*', '', chunk)

        clean = re.sub(r"<[^>]+>", " ", chunk)
        clean = re.sub(r"[ ]+", " ", clean)
        clean = re.sub(r"\n{2,}", "\n<PAR>\n", clean)
        clean = clean.strip()
        
        doc = nlp(clean)
        out = []
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            s = s.replace("<PAR>", "").strip()

             # Restore ellipsis
            s = s.replace('ELLIPSISMARKER', '...')

            # Restore original z.B. forms from map
            for placeholder, original in zb_map.items():
                s = s.replace(placeholder, original)

            # Restore ALL abbreviations with their original form
            for placeholder, original in abbrev_map.items():
                s = s.replace(placeholder, original)
            if s:
                out.append(s)
        
        # Mark end of this chunk with a special marker
        if out:
            out[-1] = out[-1] + " <CHUNKEND>"
        
        all_sentences.extend(out)

    # Merge fragments
    merged = []
    buffer = ""
    coordinating_start = re.compile(r'^(Und|Aber|Oder|Denn|Sondern|Doch)\s')

    for sentence in all_sentences:
        s_strip = sentence.strip()
        
        # Check if this sentence is marked as chunk boundary
        is_chunk_end = s_strip.endswith('<CHUNKEND>')
        if is_chunk_end:
            s_strip = s_strip.replace('<CHUNKEND>', '').strip()
        
        if not s_strip:
            continue
        if buffer:
            # Don't merge if next sentence starts with uppercase coordinating conjunction
            starts_with_coord = coordinating_start.match(s_strip)
            
            # Check if buffer has ending punctuation
            buffer_has_punct = re.search(r'[.!?]$', buffer)
            
            if starts_with_coord:
                # Force new sentence at coordinating conjunction
                merged.append(buffer)
                buffer = s_strip
            elif buffer_has_punct and s_strip[0].islower():
                # If previous has punctuation but next is lowercase, check word count
                # If next sentence has 3+ words, it's likely a new sentence
                next_words = s_strip.split()
                if len(next_words) >= 3:
                    merged.append(buffer)
                    buffer = s_strip
                else:
                    buffer += " " + s_strip
            elif s_strip[0].islower() or not buffer_has_punct:
                buffer += " " + s_strip
            else:
                merged.append(buffer)
                buffer = s_strip
        else:
            buffer = s_strip
        # CRITICAL: Force sentence boundary at chunk end (from numbered list splits)
        if is_chunk_end and buffer:
            merged.append(buffer)
            buffer = ""

    if buffer:
        merged.append(buffer)

    # Remove numbered markers after all sentence processing is complete
    cleaned = []
    for sent in merged:
        sent = re.sub(r'\d+\)', '', sent).strip()
        if sent:
            cleaned.append(sent)

    debug(f"[DEBUG SPACY OUTPUT] {cleaned}")
    return cleaned

# ============================================================================
# KOLIPSI EXTRACTION
# ============================================================================

def extract_kolipsi(element) -> Tuple[str, str, bool, List[Tuple[str, str]]]:
    """
    Extract src and tgt from Kolipsi element.
    Returns (src_text, tgt_text, has_corrections)
    """
    src_builder = TextBuilder()
    tgt_builder = TextBuilder()
    has_corrections = False
    orth_mappings = []

    def get_element_text(elem):
        """Get all text from element and descendants."""
        if elem is None:
            return ""
        return ''.join(elem.itertext()).strip()

    def get_original_form_text(elem):
        """Extract text from originalForm, handling nested structures."""
        if elem is None:
            return ""

        parts = []

        def recurse_original(node):
            tag = strip_namespace(node.tag).lower()

            if node.text and node.text.strip():
                parts.append(node.text.strip())

            if tag == "overwrite":
                over = None
                for child in node:
                    if strip_namespace(child.tag).lower() == "over":
                        over = child
                        break
                if over is not None and over.text:
                    parts.append(over.text.strip())
                if node.tail and node.tail.strip():
                    parts.append(node.tail.strip())
                return

            if tag == "palimpsest":
                palimpsest_text = ''.join(node.itertext()).strip()
                if palimpsest_text:
                    parts.append(palimpsest_text)
                if node.tail and node.tail.strip():
                    if has_leading_whitespace(node.tail):
                        parts.append(' ')
                    parts.append(node.tail.strip())
                return

            for child in node:
                recurse_original(child)

            if node.tail and node.tail.strip():
                parts.append(node.tail.strip())

        recurse_original(elem)
        return ''.join(parts)

    def recurse(node, src: TextBuilder, tgt: TextBuilder):
        nonlocal has_corrections
        tag = strip_namespace(node.tag).lower()

        # ERROR / OVER_CAPITALISATION / E
        if tag in ("error", "over_capitalisation", "e"):
            has_corrections = True
            original = None
            target = None

            for child in node:
                child_tag = strip_namespace(child.tag).lower()
                if child_tag == "originalform":
                    original = child
                elif child_tag == "targetform":
                    target = child

            # Get RAW text first to check for trailing whitespace
            orig_text_raw = get_original_form_text(original) if original is not None else ""
            tgt_text_raw = ''.join(target.itertext()) if target is not None else ""
            
            # Check for trailing whitespace BEFORE stripping
            orig_has_trailing = has_trailing_whitespace(orig_text_raw)
            tgt_has_trailing = has_trailing_whitespace(tgt_text_raw)
            
            # Now strip for processing
            orig_text = orig_text_raw.strip()
            tgt_text = tgt_text_raw.strip()
            
            if orig_text and tgt_text and orig_text != tgt_text:
                orth_mappings.append((orig_text, tgt_text))

            # Check for sentence break
            prev_src = src.get_text()
            # Don't split if target ends with hyphen (compound word component like "Obst-")
            if (orig_text and tgt_text
                and len(orig_text) > 0 and len(tgt_text) > 0
                and orig_text[0].islower() != tgt_text[0].islower()
                and has_sentence_ending(prev_src)
                and not tgt_text.endswith('-')):
                
                # Check if previous word is an adjective/adverb that shouldn't trigger split
                prev_words = prev_src.split()
                if prev_words:
                    last_word = prev_words[-1].rstrip('.,!?').lower()
                    # Don't split after common adjectives/adverbs before nouns
                    non_boundary_words = {'sehr', 'viel', 'viele', 'wenig', 'wenige', 'mehr', 'alle', 'einige', 'manche', 'solche', "in"}
                    if last_word not in non_boundary_words:
                        src.add_marker(" <SENTBREAK> ")
                        tgt.add_marker(" <SENTBREAK> ")
                        
            # Split multi-word forms by spaces and add word-by-word
            if orig_text:
                orig_words = orig_text.split()
                for i, word in enumerate(orig_words):
                    if i > 0:
                        src.add_space()
                    src.add_text(word)

            if tgt_text:
                tgt_words = tgt_text.split()
                for i, word in enumerate(tgt_words):
                    if i > 0:
                        tgt.add_space()
                    tgt.add_text(word)

            # Handle tail with proper spacing
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                tail_text = node.tail.strip()
                tail_text = node.tail.strip()
                if tail_text:
                    # Only merge if forms DON'T have trailing whitespace
                    should_merge = not (orig_has_trailing or tgt_has_trailing)
                    src.add_text(tail_text, merge=should_merge)
                    tgt.add_text(tail_text, merge=should_merge)
                    
                    # Add sentence break if tail ends with sentence-ending punctuation
                    if node.tail and re.search(r'[.!?]\s*$', node.tail):
                        src.add_marker(" <SENTBREAK> ")
                        tgt.add_marker(" <SENTBREAK> ")
                return

        # PALIMPSEST
        elif tag == "palimpsest":
            has_errors = any(
                strip_namespace(child.tag).lower() in ("error", "over_capitalisation", "e")
                for child in node
            )

            has_strikeover = any(
                strip_namespace(child.tag).lower() == "strikeover"
                for child in node
            )

            # Strikeover case
            if has_strikeover:
                if node.text and node.text.strip():
                    src.add_text(node.text.strip())
                    tgt.add_text(node.text.strip())

                for child in node:
                    child_tag = strip_namespace(child.tag).lower()
                    if child_tag == "strikeover":
                        expansion_parts = []
                        for grandchild in child:
                            if strip_namespace(grandchild.tag).lower() == "expansion" and grandchild.text:
                                expansion_parts.append(grandchild.text)

                        if expansion_parts:
                            merged = ''.join(expansion_parts)
                            src.add_text(merged, merge=True)
                            tgt.add_text(merged, merge=True)

                        if child.tail:
                            if has_leading_whitespace(child.tail):
                                src.add_space()
                                tgt.add_space()
                            if child.tail.strip():
                                src.add_text(child.tail.strip(), merge=True)
                                tgt.add_text(child.tail.strip(), merge=True)
                    else:
                        recurse(child, src, tgt)

                if node.tail:
                    if has_leading_whitespace(node.tail):
                        src.add_space()
                        tgt.add_space()
                    if node.tail.strip():
                        src.add_text(node.tail.strip(), merge=True)
                        tgt.add_text(node.tail.strip(), merge=True)
                return

            # No errors case - check XML spacing
            if not has_errors:
                if node.text and node.text.strip():
                    # Check if this is mid-word by looking at surrounding whitespace
                    merge_before = src.parts and src.parts[-1] and not src.parts[-1].endswith(' ')
                    src.add_text(node.text.strip(), merge=merge_before)
                    tgt.add_text(node.text.strip(), merge=merge_before)

                for child in node:
                    recurse(child, src, tgt)

                if node.tail:
                    if has_leading_whitespace(node.tail):
                        src.add_space()
                        tgt.add_space()
                    if node.tail.strip():
                        # Check if tail should merge (no leading space in XML)
                        merge_tail = not has_leading_whitespace(node.tail)
                        src.add_text(node.tail.strip(), merge=merge_tail)
                        tgt.add_text(node.tail.strip(), merge=merge_tail)
                return

            # Has errors case
            if node.text and node.text.strip():
                src.add_text(node.text.strip())
                tgt.add_text(node.text.strip())

            for child in node:
                child_tag = strip_namespace(child.tag).lower()

                if child_tag in ("error", "over_capitalisation", "e"):
                    has_corrections = True
                    original = None
                    target = None

                    for grandchild in child:
                        grandchild_tag = strip_namespace(grandchild.tag).lower()
                        if grandchild_tag == "originalform":
                            original = grandchild
                        elif grandchild_tag == "targetform":
                            target = grandchild

                    orig_text = get_original_form_text(original)
                    if orig_text:
                        src.add_text(orig_text)

                    tgt_text = get_element_text(target)
                    if tgt_text:
                        tgt.add_text(tgt_text)

                    if child.tail:
                        if has_leading_whitespace(child.tail):
                            src.add_space()
                            tgt.add_space()
                        if child.tail.strip():
                            merge_tail = not has_leading_whitespace(child.tail)
                            src.add_text(child.tail.strip(), merge=merge_tail)
                            tgt.add_text(child.tail.strip(), merge=merge_tail)
                else:
                    recurse(child, src, tgt)

            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # CORRECTION
        elif tag == "correction":
            for child in node:
                child_tag = strip_namespace(child.tag).lower()
                
                # Ignore deletion entirely
                if child_tag == "deletion":
                    continue
                
                elif child_tag == "insertion":
                    # Process insertion in document order: text and children mixed
                    # Check spacing before insertion content
                    should_merge = (
                        src.parts 
                        and src.parts[-1] 
                        and not src.parts[-1].endswith((' ', '\n'))
                    )
                    
                    # Add text before first child (if any)
                    if child.text and child.text.strip():
                        src.add_text(child.text.strip(), merge=should_merge)
                        tgt.add_text(child.text.strip(), merge=should_merge)
                    
                    # Process nested elements (like overwrite)
                    for grandchild in child:
                        recurse(grandchild, src, tgt)
            
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # REDUCTION
        elif tag == "reduction":
            unfolded = None
            for child in node:
                child_tag = strip_namespace(child.tag).lower()
                if child_tag == "unfoldedform":
                    unfolded = child
                    break
            
            if unfolded is not None and unfolded.text:
                unfolded_text = unfolded.text.strip()
                needs_space = False
                if src.parts:
                    last_part = src.parts[-1]
                    if last_part and last_part[-1].isalpha():
                        words = last_part.split()
                        if words and len(words[-1]) > 2:
                            needs_space = True
                
                if needs_space:
                    src.add_space()
                    tgt.add_space()
                
                src.add_text(unfolded_text)
                tgt.add_text(unfolded_text)
            
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # AMBIGUOUS
        elif tag == "ambiguous":
            # Handle text before alternatives
            if node.text and node.text.strip():
                should_merge = (
                    src.parts 
                    and src.parts[-1] 
                    and not src.parts[-1].endswith((' ', '\n'))
                )
                src.add_text(node.text.strip(), merge=should_merge)
                tgt.add_text(node.text.strip(), merge=should_merge)
            
            # Get first alternative
            first_alternative = None
            for child in node:
                child_tag = strip_namespace(child.tag).lower()
                if child_tag == "alternative":
                    first_alternative = child
                    break
            
            if first_alternative is not None and first_alternative.text:
                alt_text = first_alternative.text.strip()
                
                # Check spacing before <alternative> tag
                should_merge = (
                    src.parts 
                    and src.parts[-1] 
                    and not src.parts[-1].endswith((' ', '\n'))
                )
                
                src.add_text(alt_text, merge=should_merge)
                tgt.add_text(alt_text, merge=should_merge)
            
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # STRIKEOVER
        # STRIKEOVER
        elif tag == "strikeover":
            expansions = [child.text for child in node
            if strip_namespace(child.tag).lower() == "expansion" and child.text]
            
            # Use the appropriate expansion based on what's available
            # If there are 2+ expansions, use the second one (index 1)
            # If there's only 1 expansion, use it (index 0)
            # If there are no expansions, use empty string
            if len(expansions) >= 2:
                merged = expansions[1]  # Second expansion (the correction)
            elif len(expansions) == 1:
                merged = expansions[0]  # Only one expansion available
            else:
                merged = ""  # No expansions

            if merged:
                should_merge = (
                    src.parts
                    and src.parts[-1]
                    and not src.parts[-1].endswith((" ", "\n"))
                )

                if should_merge:
                    src.add_text(merged, merge=True)
                    tgt.add_text(merged, merge=True)
                else:
                    src.add_text(merged)
                    tgt.add_text(merged)

            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # OVERWRITE
        elif tag == "overwrite":
            over = None
            for child in node:
                child_tag = strip_namespace(child.tag).lower()
                if child_tag == "over":
                    over = child
                    break
        
            over_text = over.text if over is not None and over.text else ""
        
            if over_text:
                # Check spacing before <overwrite> tag
                should_merge = (
                    src.parts 
                    and src.parts[-1] 
                    and not src.parts[-1].endswith((' ', '\n'))
                )
                
                src.add_text(over_text, merge=should_merge)
                tgt.add_text(over_text, merge=should_merge)
        
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # OVER (standalone, not inside overwrite)
        elif tag == "over":
            over_text = node.text.strip() if node.text else ""
            
            if over_text:
                # Check spacing before <over> tag
                should_merge = (
                    src.parts 
                    and src.parts[-1] 
                    and not src.parts[-1].endswith((' ', '\n'))
                )
                
                src.add_text(over_text, merge=should_merge)
                tgt.add_text(over_text, merge=should_merge)
            
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # FOREIGN_WORD
        elif tag == "foreign_word":
            foreign_text = node.text.strip() if node.text and node.text.strip() else ""
            
            if foreign_text:
                marked_word = f'FOREIGNWORDSTART{foreign_text}FOREIGNWORDEND'
                src.add_text(marked_word)
                tgt.add_text(marked_word)
            
            for child in node:
                recurse(child, src, tgt)
            
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # IGNORE
        elif tag in ("symbol", "emoticon", "unreadable","comment","gap"):
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # PAR
        elif tag == "par":
            # For Kolipsi: DON'T add sentence breaks, just preserve spacing
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    src.add_text(node.tail.strip())
                    tgt.add_text(node.tail.strip())
            return

        # SPACEWRAPPER
        elif tag == "spacewrapper":
            src.add_space()
            tgt.add_space()
            
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # GREETING / CLOSING / ENTITY
        elif tag in ("greeting","closing","entity"):
            if node.text and node.text.strip():
                src.add_text(node.text.strip())
                tgt.add_text(node.text.strip())
        
            for child in node:
                recurse(child, src, tgt)
        
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    src.add_text(node.tail.strip())
                    tgt.add_text(node.tail.strip())
            else:
                # No tail means next sibling comes directly after
                src.add_space()
                tgt.add_space()
            return
        
        # HYPHEN
        elif tag == "hyphen":
            # Skip hyphen content, merge tail directly to previous text
            # Example: Jugend<hyphen>-</hyphen>herberge → Jugendherberge
            if node.tail and node.tail.strip():
                # Merge directly without space
                src.add_text(node.tail.strip(), merge=True)
                tgt.add_text(node.tail.strip(), merge=True)
            return

        # SIC
        elif tag == "sic":
            sic_text = node.text.strip() if node.text and node.text.strip() else ""
            
            if sic_text:
                # Add sic content to BOTH src and tgt (it's the actual text that appears)
                src.add_text(sic_text)
                tgt.add_text(sic_text)
            
            # Process any nested elements (though sic usually has just text)
            for child in node:
                recurse(child, src, tgt)
            
            # Handle tail
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)
            return

        # OTHER (default handler)
        else:
            if node.text:
                text_stripped = node.text.strip()
                has_trailing = has_trailing_whitespace(node.text)
                
                if text_stripped:
                    src.add_text(text_stripped)
                    tgt.add_text(text_stripped)
                
                if has_trailing:
                    src.add_space()
                    tgt.add_space()

            for child in node:
                recurse(child, src, tgt)

            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    merge_tail = not has_leading_whitespace(node.tail)
                    src.add_text(node.tail.strip(), merge=merge_tail)
                    tgt.add_text(node.tail.strip(), merge=merge_tail)

    recurse(element, src_builder, tgt_builder)
    return src_builder.get_text(), tgt_builder.get_text(), has_corrections, orth_mappings

def extract_kolipsi_sentences(element) -> List[SentencePair]:
    """Extract sentence pairs from Kolipsi element."""
    src_full, tgt_full, _, orth_mappings = extract_kolipsi(element)
    debug(f"[DEBUG EXTRACT_KOLIPSI] RAW src BEFORE strip_quotes: '{src_full[:200]}'")
    debug(f"[DEBUG EXTRACT_KOLIPSI] RAW tgt BEFORE strip_quotes: '{tgt_full[:200]}'")
    debug(f"[DEBUG EXTRACT_KOLIPSI] Quote check - src contains quotes: {'\"' in src_full or '„' in src_full or '"' in src_full}")
    
    debug(f"[DEBUG extract_kolipsi_sentence] RAW SRC: '{src_full}'")
    debug(f"[DEBUG extract_kolipsi_sentence] RAW TGT: '{tgt_full}'") 

    if not src_full and not tgt_full:
        return []
    
    # CRITICAL: Clean any residual markers from previous documents
    src_full = src_full.strip()
    tgt_full = tgt_full.strip()
    
    # Ensure sentence breaks at document boundaries
    if not src_full.startswith('<SENTBREAK>'):
        src_full = '<SENTBREAK>' + src_full
        tgt_full = '<SENTBREAK>' + tgt_full

    src_chunks = [s.strip() for s in src_full.split('<SENTBREAK>') if s.strip()]
    tgt_chunks = [s.strip() for s in tgt_full.split('<SENTBREAK>') if s.strip()]

    if len(src_chunks) != len(tgt_chunks):
        max_chunks = max(len(src_chunks), len(tgt_chunks))
        src_chunks.extend([''] * (max_chunks - len(src_chunks)))
        tgt_chunks.extend([''] * (max_chunks - len(tgt_chunks)))

    pairs = []
    for src_chunk, tgt_chunk in zip(src_chunks, tgt_chunks):
        if not src_chunk and not tgt_chunk:
            continue

        # Detect foreign words at chunk level but clean before splitting
        has_foreign_in_chunk = ('FOREIGNWORDSTART' in src_chunk or 
                               'FOREIGNWORDSTART' in tgt_chunk)
        
        src_chunk = re.sub(r'FOREIGNWORDSTART(.*?)FOREIGNWORDEND', r'\1', src_chunk)
        tgt_chunk = re.sub(r'FOREIGNWORDSTART(.*?)FOREIGNWORDEND', r'\1', tgt_chunk)
        
        src_chunk = re.sub(r'\s+', ' ', src_chunk).strip()
        tgt_chunk = re.sub(r'\s+', ' ', tgt_chunk).strip()

        # NEW: Keep original chunks AND create stripped versions
        src_chunk_original = src_chunk
        tgt_chunk_original = tgt_chunk
        _, src_chunk_no_quotes = strip_quotes_preserve_original(src_chunk)
        _, tgt_chunk_no_quotes = strip_quotes_preserve_original(tgt_chunk)

        src_sents = spacy_sent(src_chunk_no_quotes) if src_chunk_no_quotes else []
        tgt_sents = spacy_sent(tgt_chunk_no_quotes) if tgt_chunk_no_quotes else []
        if not src_sents and not tgt_sents:
            continue

        if src_sents is None:
            src_sents = []
        if tgt_sents is None:
            tgt_sents = []

        if len(src_sents) == 0 and len(tgt_sents) == 0:
            continue

        max_len = max(len(src_sents), len(tgt_sents))
        for i in range(max_len):
            src_sent = src_sents[i] if i < len(src_sents) else ""
            tgt_sent = tgt_sents[i] if i < len(tgt_sents) else ""
            
            # RESTORE QUOTES IMMEDIATELY after sentence extraction
            if src_sent:
                src_sent = restore_quotes_to_sentence(src_chunk_original, src_chunk_no_quotes, src_sent)
            if tgt_sent:
                tgt_sent = restore_quotes_to_sentence(tgt_chunk_original, tgt_chunk_no_quotes, tgt_sent)
     
            has_correction = (src_sent.strip() != tgt_sent.strip())
            
            # Check if this is continuation of split compound word
            if (pairs and 
                src_sent and len(src_sent.split()) == 1 and src_sent[0].islower() and
                tgt_sent and len(tgt_sent.split()) == 1 and tgt_sent[0].isupper()):
                # Merge with previous pair - replace its target with current target
                pairs[-1] = SentencePair(
                    src=pairs[-1].src,
                    tgt=tgt_sent,
                    has_correction=True,
                    has_foreign=pairs[-1].has_foreign or has_foreign_in_chunk
                )
                continue  # Skip adding this as separate pair
            
            if src_sent or tgt_sent:
                # Filter mappings that appear in this sentence
                sent_mappings = [
                    (orig, tgt_map) for orig, tgt_map in orth_mappings
                    if orig in src_sent
                ]
                
                pairs.append(SentencePair(
                    src=src_sent,
                    tgt=tgt_sent,
                    has_correction=has_correction,
                    has_foreign=has_foreign_in_chunk,
                    orth_mappings=sent_mappings
                ))

    return pairs

# ============================================================================
# LEONIDE EXTRACTION
# ============================================================================

def extract_leonide(paragraph, all_paragraphs=None) -> Tuple[str, str, bool, List[Tuple[str, str]]]:
    """Extract text from LEONIDE paragraph."""
    src_builder = TextBuilder()
    tgt_builder = TextBuilder()
    has_corrections = False
    orth_error_mappings = []  # NEW: Collect (original, target) pairs

    def get_nested_text(element) -> str:
        """Recursively extract text from nested elements."""
        if element is None:
            return ""
        
        # Try direct text first
        if element.text and element.text.strip():
            return element.text.strip()
        
        # Check if any children exist
        children_list = list(element)
        
        # If no children, use itertext as fallback
        if not children_list:
            return ''.join(element.itertext()).strip()
        
        # Has children - recursively search them
        for child in children_list:
            text = get_nested_text(child)  # ALWAYS recurse
            if text:
                return text
        
        # If no text found in children, use itertext as last resort
        return ''.join(element.itertext()).strip()

    def process_node(node, src: TextBuilder, tgt: TextBuilder):
        nonlocal has_corrections
        nonlocal orth_error_mappings 
        debug(f"[DEBUG PROCESS_NODE CALLED] tag={strip_namespace(node.tag)}")
        
        # Check if the node itself is an orth_error (when called recursively)
        if 'orth_error' in node.tag.lower():
            debug(f"[DEBUG PROCESS_NODE] Handling orth_error")
            has_corrections = True
            
            target_attr = node.get('orth_error_target', '')
            
            # Check if there are special children that need custom handling
            has_ambiguous_child = any('tran_ambiguous' in c.tag.lower() for c in node)
            has_deletion_child = any('tran_word_deletion' in c.tag.lower() for c in node)

            if has_ambiguous_child or has_deletion_child:
                # Build text manually, skipping deletions and handling ambiguous
                original_parts = []
                
                # Add node.text (e.g., "weg")
                if node.text and node.text.strip():
                    original_parts.append(node.text.strip())
                
                for child in node:
                    child_tag = child.tag.lower()                    
                    # Skip deletions entirely
                    if 'tran_word_deletion' in child_tag:
                        # But add the tail after deletion
                        if child.tail and child.tail.strip():
                            if original_parts and not original_parts[-1].endswith(' '):
                                original_parts.append(' ')
                            original_parts.append(child.tail.strip())
                        continue
                    
                    # Handle ambiguous - use get_nested_text to handle deep nesting
                    if 'tran_ambiguous' in child_tag:
                        if original_parts and not original_parts[-1].endswith(' '):
                            original_parts.append(' ')
                        # Use get_nested_text to extract from potentially nested structures
                        ambiguous_text = get_nested_text(child)
                        if ambiguous_text:
                            original_parts.append(ambiguous_text)
                
                original_text = ''.join(original_parts)
            else:
                # Use get_nested_text to handle nested structures
                original_text = get_nested_text(node)
            if node.text and has_leading_whitespace(node.text):
                src.add_space()
                tgt.add_space()
            
            if original_text:
                src.add_text(original_text)
            
            if target_attr:
                tgt.add_text(target_attr)
            elif original_text:
                tgt.add_text(original_text)
            
            return  # Don't process children or continue 

        # Check if the node itself is a tran_capitalisation (when called recursively)
        if 'tran_capitalisation' in node.tag.lower():
            original_text = node.text.strip() if node.text else ""
            target_attr = node.get('tran_capitalisation_target', '')
               
            # Only add space if we already have content
            if (original_text or target_attr) and src.parts:
                src.add_space()
                tgt.add_space()
            
            if original_text:
                src.add_text(original_text)
            
            if target_attr:
                tgt.add_text(target_attr)
            elif original_text:
                # Fallback: if no target, use original for both
                tgt.add_text(original_text)
            
            return  # Don't process children or continue      
        
        # Add node.text for tags that don't handle their own text specially
        node_tag = node.tag.lower()
        tags_that_handle_own_text = ['tran_word_correction', 'tran_word_insertion', 'tran_word_deletion', 
                                       'tran_foreign_word', 'tran_symbol', 'tran_emoticon', 
                                       'tran_unreadable', 'tran_reduction', 'tran_capitalisation']

        # Add node.text for leaf nodes or for tags that don't handle their own text specially
        if node.text and node.text.strip() and (len(node) == 0 or not any(tag in node_tag for tag in tags_that_handle_own_text)):
            src.add_text(node.text.strip())
            tgt.add_text(node.text.strip())
    
        for child in node:
            tag = child.tag.lower()

            # DIV
            if 'div' in tag.lower():
                debug(f"[DEBUG DIV] Processing div with {len(child)} children")
                # Check if this div ends mid-word (compound word split across divs)
                ends_mid_word = (src.parts and src.parts[-1] and 
                                not src.parts[-1].endswith((' ', '\n')) and
                                src.parts[-1][-1].isalpha())
                
                if child.text and child.text.strip():
                    # If previous div ended mid-word, merge directly
                    src.add_text(child.text.strip(), merge=ends_mid_word)
                    tgt.add_text(child.text.strip(), merge=ends_mid_word)
                
                for grandchild in child:
                    process_node(grandchild, src, tgt)
                
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue

            # FOREIGN WORD
            if 'tran_foreign_word' in tag:
                all_foreign_text = ''.join(child.itertext()).strip()
                
                if all_foreign_text:
                    marked_word = f'FOREIGNWORDSTART{all_foreign_text}FOREIGNWORDEND'
                    src.add_text(marked_word)
                    tgt.add_text(marked_word)
                
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue

            # SYMBOL / EMOTICON / UNREADABLE
            if 'tran_symbol' in tag or 'tran_emoticon' in tag or 'tran_unreadable' in tag:
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue

            # AMBIGUOUS
            if 'tran_ambiguous' in tag:
                src.add_space()
                tgt.add_space()
                
                # CRITICAL FIX: Always add child.text BEFORE processing nested elements
                if child.text and child.text.strip():
                    src.add_text(child.text.strip())
                    tgt.add_text(child.text.strip())
                
                # Always recurse into children to handle nested structures
                for grandchild in child:
                    process_node(grandchild, src, tgt)
                    
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue

            # WORD DELETION
            if 'tran_word_deletion' in tag:
                # Skip the deleted content entirely - do NOT add child.text to either src or tgt
                # Only process the tail
                if child.tail:
                    if has_leading_whitespace(child.tail) or src.parts:
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue

            # For pure tran_word_insertion wrapping only tran_capitalisation: skip text, just recurse
            if 'tran_word_correction' in tag or 'tran_word_insertion' in tag:
                # Check what children exist
                has_children = len(child) > 0
                direct_capitalisation_child = any('tran_capitalisation' in c.tag.lower() for c in child)
                orth_error_descendants = any('orth_error' in elem.tag.lower() for elem in child.iter())
                debug(f"[DEBUG INSERTION] tag={tag}, has_children={has_children}, direct_cap={direct_capitalisation_child}, orth_descendants={orth_error_descendants}")

                # CASE 1: Pure tran_word_insertion wrapping only tran_capitalisation (NOT inside orth_error)
                # When inside orth_error, we want the orth_error handler to manage everything
                parent_is_orth_error = 'orth_error' in node.tag.lower()
                
                if ('tran_word_insertion' in tag and direct_capitalisation_child 
                    and not orth_error_descendants and not parent_is_orth_error
                    and (not child.text or not child.text.strip())):
                    for grandchild in child:
                        process_node(grandchild, src, tgt)
                    
                    if child.tail:
                        if has_leading_whitespace(child.tail):
                            src.add_space()
                            tgt.add_space()
                        if child.tail.strip():
                            src.add_text(child.tail.strip())
                            tgt.add_text(child.tail.strip())
                    continue

                # CASE 2: tran_word_insertion with mixed content OR inside orth_error
                # Process in document order, DON'T add extra space (parent orth_error handles spacing)
                if 'tran_word_insertion' in tag:
                    # DON'T add space here - let parent orth_error handle it
                    
                    # Add child.text first (if any)
                    if child.text and child.text.strip():
                        src.add_text(child.text.strip())
                        tgt.add_text(child.text.strip())
                    
                    # Process each grandchild in order
                    for grandchild in child:
                        grandchild_tag = grandchild.tag.lower()
                        
                        # Skip deletions and unreadable, but keep their tails
                        if 'tran_word_deletion' in grandchild_tag or 'tran_unreadable' in grandchild_tag:
                            debug(f"[DEBUG INSERTION] Skipping deletion/unreadable")
                            if grandchild.tail and grandchild.tail.strip():
                                if has_leading_whitespace(grandchild.tail):
                                    src.add_space()
                                    tgt.add_space()
                                src.add_text(grandchild.tail.strip())
                                tgt.add_text(grandchild.tail.strip())
                        # Recurse for other elements
                        else:
                            debug(f"[DEBUG INSERTION] Recursing into {grandchild_tag}")
                            # CRITICAL: Extract orth_error mappings from nested elements BEFORE processing
                            if 'orth_error' in grandchild_tag:
                                target_attr = grandchild.get('orth_error_target', '')
                                # Build complete original text including nested ambiguous/other tags
                                orig_parts = []
                                if grandchild.text and grandchild.text.strip():
                                    orig_parts.append(grandchild.text.strip())
                                for nested in grandchild:
                                    nested_tag = nested.tag.lower()
                                    if 'tran_ambiguous' in nested_tag:
                                        if nested.text and nested.text.strip():
                                            orig_parts.append(nested.text.strip())
                                    elif 'tran_word_deletion' not in nested_tag:
                                        nested_text = get_nested_text(nested)
                                        if nested_text:
                                            orig_parts.append(nested_text)
                                
                                nested_original = ' '.join(orig_parts)
                                if nested_original and target_attr:
                                    orth_error_mappings.append((nested_original.strip(), target_attr.strip()))
                                    debug(f"[DEBUG INSERTION] *** STORED NESTED MAPPING: '{nested_original.strip()}' → '{target_attr.strip()}'")

                            process_node(grandchild, src, tgt)
                            
                            # Handle tail after the element
                            if grandchild.tail and grandchild.tail.strip():
                                if has_leading_whitespace(grandchild.tail):
                                    src.add_space()
                                    tgt.add_space()
                                src.add_text(grandchild.tail.strip())
                                tgt.add_text(grandchild.tail.strip())
                    
                    # Handle child.tail
                    if child.tail:
                        if has_leading_whitespace(child.tail):
                            src.add_space()
                            tgt.add_space()
                        if child.tail.strip():
                            src.add_text(child.tail.strip())
                            tgt.add_text(child.tail.strip())
                    continue
                
                # CASE 3: tran_word_correction (general case)
                if has_children and src.parts:
                    src.add_space()
                    tgt.add_space()
                
                if child.text and child.text.strip():
                    src.add_text(child.text.strip())
                    tgt.add_text(child.text.strip())
                
                for grandchild in child:
                    grandchild_tag = grandchild.tag.lower()
                    
                    if 'tran_word_deletion' in grandchild_tag:
                        if grandchild.tail:
                            if has_leading_whitespace(grandchild.tail): 
                                src.add_space()
                                tgt.add_space()
                            if grandchild.tail.strip(): 
                                src.add_text(grandchild.tail.strip())
                                tgt.add_text(grandchild.tail.strip())
                    else:
                        process_node(grandchild, src, tgt)
                
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue
                        
            # VARIANTS
            if 'tran_variants' in tag:
                if child.text and child.text.strip():
                    src.add_text(child.text.strip())
                    tgt.add_text(child.text.strip())
                
                for grandchild in child:
                    process_node(grandchild, src, tgt)
                
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue
                
            # REDUCTION
            if 'tran_reduction' in tag:
                target_attr = child.get('tran_reduction_target', '')
                
                # Get ALL text from reduction element (handles nested tags)
                reduced_text = get_nested_text(child)
                
                if reduced_text and target_attr:
                    # CRITICAL: Store mapping BEFORE any processing
                    orth_error_mappings.append((reduced_text.strip(), target_attr.strip()))
                    
                    # DON'T add space automatically - check if we need it
                    needs_space = src.parts and src.parts[-1] and not src.parts[-1].endswith(' ')
                    
                    if needs_space:
                        src.add_space()
                        tgt.add_space()
                    
                    # Split multi-word reductions word-by-word
                    src_words = reduced_text.split()
                    tgt_words = target_attr.split()
                    
                    # Add src words
                    for i, word in enumerate(src_words):
                        if i > 0:
                            src.add_space()
                        src.add_text(word)
                    
                    # Add tgt words  
                    for i, word in enumerate(tgt_words):
                        if i > 0:
                            tgt.add_space()
                        tgt.add_text(word)
                
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        merge_tail = not has_leading_whitespace(child.tail)
                        src.add_text(child.tail.strip(), merge=merge_tail)
                        tgt.add_text(child.tail.strip(), merge=merge_tail)
                continue
            # ORTH ERROR
            if 'orth_error' in tag:
                debug(f"[DEBUG ORTH_ERROR IN PROCESS_NODE] tagcode={child.get('tagcode', 'NONE')}, target={child.get('orth_error_target', 'NONE')}")
                has_corrections = True
                
                # Get the target attribute (this is the corrected form)
                target_attr = child.get('orth_error_target', '')
                tagcode = child.get('tagcode', '')
                # Check if orth_error contains nested foreign words
                has_nested_foreign = any('tran_foreign_word' in elem.tag.lower() for elem in child.iter())
                
                if has_nested_foreign:
                    # Mark the foreign content and skip processing
                    foreign_text = ''.join(child.itertext()).strip()
                    if foreign_text:
                        marked_word = f'FOREIGNWORDSTART{foreign_text}FOREIGNWORDEND'
                        src.add_text(marked_word)
                        tgt.add_text(marked_word)
                    
                    if child.tail:
                        if has_leading_whitespace(child.tail):
                            src.add_space()
                            tgt.add_space()
                        if child.tail.strip():
                            src.add_text(child.tail.strip())
                            tgt.add_text(child.tail.strip())
                            debug(f"[DEBUG ORTH_ERROR] Processing orth_error with target='{target_attr}', tagcode='{tagcode}'")                

                        if re.search(r'[.!?]\s*$', child.tail):
                            src.add_marker(" <SENTBREAK> ")
                            tgt.add_marker(" <SENTBREAK> ")
                    continue
                    
                # Check if this orth_error is a continuation (same tagcode appeared earlier)
                is_continuation = False
                
                if tagcode and all_paragraphs:
                    # Search ALL previous orth_errors in ALL paragraphs for matching tagcode
                    found_earlier = False
                    for prev_para in all_paragraphs:
                        for prev_elem in prev_para.iter():
                            if 'orth_error' in prev_elem.tag.lower():
                                prev_tagcode = prev_elem.get('tagcode', '')
                                if prev_tagcode == tagcode:
                                    # Check if this is the SAME element (not earlier occurrence)
                                    if prev_elem is child:
                                        # We've reached the current element, stop searching
                                        break
                                    else:
                                        # Found an earlier occurrence with same tagcode
                                        found_earlier = True
                                        debug(f"[DEBUG ORTH_ERROR] Found earlier orth_error with same tagcode='{tagcode}'")
                                        break
                        if found_earlier:
                            break
                    
                    is_continuation = found_earlier
                
                if is_continuation:
                    debug(f"[DEBUG ORTH_ERROR] *** CONTINUATION DETECTED *** (tagcode='{tagcode}') - SKIPPING target addition")

                
                # Build original text by processing ALL content in document order
                original_parts = []
                
                # Add child.text first
                if child.text and child.text.strip():
                    original_parts.append(child.text.strip())
                    debug(f"[DEBUG ORTH_ERROR] Added child.text: '{child.text.strip()}'")
              
                # Process each grandchild in document order
                for grandchild in child:
                    grandchild_tag = grandchild.tag.lower()
                    
                    # Skip deletion content entirely - don't extract anything from it
                    if 'tran_word_deletion' in grandchild_tag:
                        # Don't add any text from deletion to original_parts
                        # Just handle the tail
                        if grandchild.tail and grandchild.tail.strip():
                            if original_parts and not original_parts[-1].endswith(' '):
                                original_parts.append(' ')
                            original_parts.append(grandchild.tail.strip())
                        continue
                                    
                    # For tran_word_insertion or tran_word_correction, extract nested content
                    if 'tran_word_insertion' in grandchild_tag or 'tran_word_correction' in grandchild_tag:
                        # CRITICAL FIX: For tran_word_correction inside orth_error, we want the ORIGINAL text
                        # Look for the visible text content, not nested corrections
                        
                        # First add the direct text of tran_word_correction itself
                        if grandchild.text and grandchild.text.strip():
                            if original_parts and not original_parts[-1].endswith(' '):
                                original_parts.append(' ')
                            original_parts.append(grandchild.text.strip())
                        
                        # Then look for nested elements (capitalisation, ambiguous)
                        for nested in grandchild:
                            nested_tag = nested.tag.lower()
                            
                            # Direct capitalisation
                            if 'tran_capitalisation' in nested_tag:
                                if nested.text and nested.text.strip():
                                    if original_parts and not original_parts[-1].endswith(' '):
                                        original_parts.append(' ')
                                    original_parts.append(nested.text.strip())
                            
                            # Direct ambiguous - extract all text
                            elif 'tran_ambiguous' in nested_tag:
                                ambiguous_text = ''.join(nested.itertext()).strip()
                                if ambiguous_text:
                                    if original_parts and not original_parts[-1].endswith(' '):
                                        original_parts.append(' ')
                                    original_parts.append(ambiguous_text)
                            
                            # One level deeper (e.g., tran_word_correction > tran_word_insertion > tran_capitalisation)
                            elif 'tran_word_insertion' in nested_tag:
                                for deep_nested in nested:
                                    deep_tag = deep_nested.tag.lower()
                                    if 'tran_capitalisation' in deep_tag:
                                        if deep_nested.text and deep_nested.text.strip():
                                            if original_parts and not original_parts[-1].endswith(' '):
                                                original_parts.append(' ')
                                            original_parts.append(deep_nested.text.strip())
                                    elif 'tran_ambiguous' in deep_tag:
                                        ambiguous_text = ''.join(deep_nested.itertext()).strip()
                                        if ambiguous_text:
                                            if original_parts and not original_parts[-1].endswith(' '):
                                                original_parts.append(' ')
                                            original_parts.append(ambiguous_text)
                        
                        # Handle tail after tran_word_insertion/correction
                        if grandchild.tail and grandchild.tail.strip():
                            if original_parts and not original_parts[-1].endswith(' '):
                                original_parts.append(' ')
                            original_parts.append(grandchild.tail.strip())
                        continue


                    # Handle tran_emphasis - extract direct text AND nested content
                    if 'tran_emphasis' in grandchild_tag:
                        # CRITICAL: Add direct text from tran_emphasis first
                        if grandchild.text and grandchild.text.strip():
                            if original_parts and not original_parts[-1].endswith(' '):
                                original_parts.append(' ')
                            original_parts.append(grandchild.text.strip())
                        
                        # Then recursively extract from nested elements
                        for nested in grandchild:
                            nested_tag = nested.tag.lower()
                            
                            if 'tran_capitalisation' in nested_tag:
                                if nested.text and nested.text.strip():
                                    if original_parts and not original_parts[-1].endswith(' '):
                                        original_parts.append(' ')
                                    original_parts.append(nested.text.strip())
                            
                            elif 'tran_word_insertion' in nested_tag or 'tran_word_correction' in nested_tag:
                                for deep_nested in nested:
                                    deep_tag = deep_nested.tag.lower()
                                    if 'tran_capitalisation' in deep_tag:
                                        if deep_nested.text and deep_nested.text.strip():
                                            if original_parts and not original_parts[-1].endswith(' '):
                                                original_parts.append(' ')
                                            original_parts.append(deep_nested.text.strip())
                        
                        # Handle tail after tran_emphasis
                        if grandchild.tail and grandchild.tail.strip():
                            if original_parts and not original_parts[-1].endswith(' '):
                                original_parts.append(' ')
                            original_parts.append(grandchild.tail.strip())
                        continue

                    # For ambiguous, recursively extract all text
                    if 'tran_ambiguous' in grandchild_tag:
                        if original_parts and not original_parts[-1].endswith(' '):
                            original_parts.append(' ')
                        ambiguous_text = ''.join(grandchild.itertext()).strip()
                        if ambiguous_text:
                            original_parts.append(ambiguous_text)
                        
                        # CRITICAL FIX: Handle tail after ambiguous
                        if grandchild.tail and grandchild.tail.strip():
                            if original_parts and not original_parts[-1].endswith(' '):
                                original_parts.append(' ')
                            original_parts.append(grandchild.tail.strip())
                        continue
                    
                    # For other tags, just get their text
                    if grandchild.text and grandchild.text.strip():
                        if original_parts and not original_parts[-1].endswith(' '):
                            original_parts.append(' ')
                        original_parts.append(grandchild.text.strip())
                
                original_text = ''.join(original_parts)

                # Store mapping for ALL parts (both first occurrence and continuation)
                # This ensures NORM alignment can handle split words
                if original_text and target_attr:
                    # Store ALL mappings - even duplicates (they occur at different positions)
                    orth_error_mappings.append((original_text.strip(), target_attr.strip()))
                if not original_text and target_attr:
                    # This orth_error only wraps deletions - skip it entirely
                    if child.tail:
                        if has_leading_whitespace(child.tail):
                            src.add_space()
                            tgt.add_space()
                        if child.tail.strip():
                            src.add_text(child.tail.strip())
                            tgt.add_text(child.tail.strip())
                    continue
                
                # Add space if needed
                if child.text and has_leading_whitespace(child.text):
                    src.add_space()
                    tgt.add_space()
                
                # SENTBREAK logic (only if we have both original and target)
                if target_attr and original_text:
                    prev_src = src.get_text()
                    if prev_src:
                        prev_text_clean = prev_src.replace('<SENTBREAK>', '').replace('<DEL>', '').strip()
                        has_real_punctuation = prev_text_clean and prev_text_clean[-1] in '.!?'
                        
                        if has_real_punctuation:
                            if len(original_text) > 0 and len(target_attr) > 0:
                                prev_words = prev_text_clean.split()
                                last_word = prev_words[-1] if prev_words else ""
                                last_word_lower = last_word.rstrip('.,!?').lower()
                                
                                non_boundary_words = {'zum', 'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen', 'einem', 'einer', 'im', 'am', 'vom', 'beim'}
                                
                                is_abbreviation = last_word.rstrip('.,!?') in {'z.B', 'u.a', 'd.h', 'bzw', 'etc', 'ca', 'evtl', 'Mr', 'Dr', 'Prof', 'vs', 'Fam'}
                                
                                if last_word_lower not in non_boundary_words and not is_abbreviation:
                                    if original_text[0].islower() and target_attr[0].isupper():
                                        src.add_marker(" <SENTBREAK> ")
                                        tgt.add_marker(" <SENTBREAK> ")
                                    elif original_text[0].isupper() and target_attr[0].isupper():
                                        src.add_marker(" <SENTBREAK> ")
                                        tgt.add_marker(" <SENTBREAK> ")
                
                # Add original text to src
                if original_text:
                    src.add_text(original_text)
                    debug(f"[DEBUG ORTH_ERROR] Added to SRC: '{original_text}'")


                # Add target to tgt ONLY if not a continuation
                if target_attr and not is_continuation:
                    tgt.add_text(target_attr)
                    debug(f"[DEBUG ORTH_ERROR] Added to TGT: '{target_attr}'")
                elif original_text and not is_continuation:
                    debug(f"[DEBUG ORTH_ERROR] Added original to TGT: '{original_text}'")
                    tgt.add_text(original_text)
                elif is_continuation:
                    debug(f"[DEBUG ORTH_ERROR] *** SKIPPED adding target to TGT (continuation)")


                # Handle tail
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                        debug(f"[DEBUG ORTH_ERROR] Added tail: '{child.tail.strip()}'")
                    
                    # Add sentence break if tail ends with sentence-ending punctuation
                    if re.search(r'[.!?]\s*$', child.tail):
                        src.add_marker(" <SENTBREAK> ")
                        tgt.add_marker(" <SENTBREAK> ")    
                continue

            # CAPITALISATION
            if 'tran_capitalisation' in tag:
                original_text = child.text.strip() if child.text else ""
                target_attr = child.get('tran_capitalisation_target', '')
                
                # Add space before capitalisation if we have content
                if (original_text or target_attr) and src.parts:
                    src.add_space()
                    tgt.add_space()
                
                # CRITICAL FIX: Always add original text to SRC (e.g., "NIE")
                if original_text:
                    src.add_text(original_text)
                
                # Add target (lowercase) to TGT (e.g., "nie")
                if target_attr:
                    tgt.add_text(target_attr)
                elif original_text:
                    # Fallback: if no target, use original for both
                    tgt.add_text(original_text)
                
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        # Merge punctuation directly (no space before !" or .")
                        tail_stripped = child.tail.strip()
                        merge_tail = tail_stripped[0] in '!?".,;:' if tail_stripped else False
                        src.add_text(tail_stripped, merge=merge_tail)
                        tgt.add_text(tail_stripped, merge=merge_tail)
                continue

            # Recurse for other tags
            process_node(child, src, tgt)
            
            # Handle tail
            if child.tail:
                if has_leading_whitespace(child.tail):
                    src.add_space()
                    tgt.add_space()
                if child.tail.strip():
                    src.add_text(child.tail.strip())
                    tgt.add_text(child.tail.strip())

    for child in paragraph:
        process_node(child, src_builder, tgt_builder)
    
    debug(f"[DEBUG extract_leonide] Collected {len(orth_error_mappings)} orth_error mappings:")
    for orig, tgt in orth_error_mappings:
        debug(f"  '{orig}' → '{tgt}'")
    
    return src_builder.get_text(), tgt_builder.get_text(), has_corrections, orth_error_mappings
    
def extract_leonide_sentences(paragraph, all_paragraphs=None) -> List[SentencePair]:
    """Extract sentence pairs from LEONIDE paragraph."""
    src, tgt, _, orth_mappings = extract_leonide(paragraph, all_paragraphs)
    debug(f"[DEBUG extract_leonide_sentences] RAW SRC: '{src}'")
    debug(f"[DEBUG extract_leonide_sentences] RAW TGT: '{tgt}'")

    if not src and not tgt:
        return []

    # Detect foreign words before cleaning markers
    has_foreign = 'FOREIGNWORDSTART' in src or 'FOREIGNWORDSTART' in tgt

    # CRITICAL FIX: NEVER use explicit breaks from DIV tags - they're unreliable
    # Always rely on spacy for sentence splitting
    src_break_count = src.count('<SENTBREAK>')
    tgt_break_count = tgt.count('<SENTBREAK>')
    
    debug(f"[DEBUG LEONIDE BREAK COUNT] src_breaks={src_break_count}, tgt_breaks={tgt_break_count}")

    # Force spacy splitting for all cases
    use_explicit_breaks = False

    debug(f"[DEBUG BREAK DECISION] src_breaks={src_break_count}, tgt_breaks={tgt_break_count}, use_explicit_breaks={use_explicit_breaks}")

    if use_explicit_breaks:
        # Verify breaks are in roughly the same positions
        src_chunks = [s.strip() for s in src.split('<SENTBREAK>') if s.strip()]
        tgt_chunks = [s.strip() for s in tgt.split('<SENTBREAK>') if s.strip()]
        
        # If chunk counts don't match, fall back to spacy
        if len(src_chunks) != len(tgt_chunks):
            use_explicit_breaks = False

    if use_explicit_breaks:
        # Clean foreign word markers
        src_chunks = [re.sub(r'FOREIGNWORDSTART(.*?)FOREIGNWORDEND', r'\1', chunk) for chunk in src_chunks]
        tgt_chunks = [re.sub(r'FOREIGNWORDSTART(.*?)FOREIGNWORDEND', r'\1', chunk) for chunk in tgt_chunks]
        
        pairs = []
        for i in range(len(src_chunks)):
            src_chunk = src_chunks[i]
            tgt_chunk = tgt_chunks[i]
            
            # CRITICAL FIX: Still need to split chunks with spaCy in case they contain multiple sentences
            src_sents = spacy_sent(src_chunk) if src_chunk else []
            tgt_sents = spacy_sent(tgt_chunk) if tgt_chunk else []
            
            # Align sentences within this chunk
            max_len = max(len(src_sents), len(tgt_sents))
            for j in range(max_len):
                src_sent = src_sents[j] if j < len(src_sents) else ""
                tgt_sent = tgt_sents[j] if j < len(tgt_sents) else ""
                
                has_correction = (src_sent.strip() != tgt_sent.strip())
                
                if src_sent or tgt_sent:
                    # Filter mappings that appear in this sentence
                    # Keep ALL occurrences - NORM alignment will handle which one is actually corrected
                    sent_mappings = [
                        (orig, tgt_map) for orig, tgt_map in orth_mappings
                        if orig in src_sent
                    ]
                    
                    pairs.append(SentencePair(
                        src=src_sent,
                        tgt=tgt_sent,
                        has_correction=has_correction,
                        has_foreign=has_foreign_in_sent,
                        orth_mappings=sent_mappings
                    ))
        return pairs

    else:
        debug("[DEBUG USING SPACY FOR SENTENCE SPLIT - IGNORING SENTBREAK]")
        # Don't use SENTBREAK markers from DIVs - just treat as one continuous text
        # Remove ALL SENTBREAK markers and treat as continuous text
        src = src.replace('<SENTBREAK>', ' ')
        tgt = tgt.replace('<SENTBREAK>', ' ')
        
        # DON'T clean foreign word markers yet - keep them to detect per-sentence
        # Clean up spaces
        src = re.sub(r'\s+', ' ', src).strip()
        tgt = re.sub(r'\s+', ' ', tgt).strip()
        
        # ADD THESE TWO LINES (in case spaces were reintroduced):
        src = re.sub(r'\s+([.,!?;:])', r'\1', src)
        tgt = re.sub(r'\s+([.,!?;:])', r'\1', tgt)

        debug(f"[DEBUG SRC (cleaned)]: '{src[:200]}'")
        debug(f"[DEBUG TGT (cleaned)]: '{tgt[:200]}'")
        
        # NEW: Strip quotes BEFORE sentencizing
        # NEW: Strip quotes BEFORE sentencizing
        src_original, src_no_quotes = strip_quotes_preserve_original(src)
        tgt_original, tgt_no_quotes = strip_quotes_preserve_original(tgt)
        
        debug(f"[DEBUG SRC (no quotes)]: '{src_no_quotes[:200]}'")
        debug(f"[DEBUG TGT (no quotes)]: '{tgt_no_quotes[:200]}'")
        
        # Split into sentences using spacy (WITHOUT quotes)
        src_sents = spacy_sent(src_no_quotes) if src_no_quotes else []
        tgt_sents = spacy_sent(tgt_no_quotes) if tgt_no_quotes else []

        debug(f"[DEBUG SENTENCE COUNTS] SRC={len(src_sents)}, TGT={len(tgt_sents)}")

        pairs = []
        max_sents = max(len(src_sents), len(tgt_sents))

        for i in range(max_sents):
            src_sent = src_sents[i] if i < len(src_sents) else ""
            tgt_sent = tgt_sents[i] if i < len(tgt_sents) else ""
            
            # Detect foreign words in THIS sentence only
            has_foreign_in_sent = ('FOREIGNWORDSTART' in src_sent or 
                                'FOREIGNWORDSTART' in tgt_sent)
            
            # Clean foreign word markers from sentences
            src_sent = re.sub(r'FOREIGNWORDSTART(.*?)FOREIGNWORDEND', r'\1', src_sent)
            tgt_sent = re.sub(r'FOREIGNWORDSTART(.*?)FOREIGNWORDEND', r'\1', tgt_sent)
            
            # RESTORE QUOTES IMMEDIATELY after sentence extraction
            if src_sent:
                src_sent = restore_quotes_to_sentence(src_original, src_no_quotes, src_sent)
            if tgt_sent:
                tgt_sent = restore_quotes_to_sentence(tgt_original, tgt_no_quotes, tgt_sent)
                        
            has_correction = (src_sent.strip() != tgt_sent.strip())
            
            if src_sent or tgt_sent:
                sent_mappings = [
                    (orig, tgt_map) for orig, tgt_map in orth_mappings
                    if orig in src_sent
                ]
                
                pairs.append(SentencePair(
                    src=src_sent,
                    tgt=tgt_sent,
                    has_correction=has_correction,
                    has_foreign=has_foreign_in_sent,
                    orth_mappings=sent_mappings
                ))
                
        return pairs

# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

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

def extract_from_xml(xml_content: str, corpus_type: str) -> List[SentencePair]:
    """Main extraction function."""
    # Inject space wrappers
    xml_content = inject_spaces_between_tags(xml_content)

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"[ERROR] XML Parse Error: {e}")
        return []

    if corpus_type == "LEONIDE":
        paras = root.findall('.//{http://www.eurac.edu/transcanno}paragraph') or root.findall('.//paragraph')
        
        unique_paras = []
        seen_ids = set()
        for para in paras:
            para_id = id(para)
            if para_id not in seen_ids:
                seen_ids.add(para_id)
                unique_paras.append(para)
        
        all_pairs = []
        consumed = set()
        last_para_orth_errors = {}  # Track tagcodes across paragraphs
        
        for i, para in enumerate(unique_paras):
            if i in consumed:
                continue
            
            # Extract orth_errors from this paragraph BEFORE sentence extraction
            current_para_orth_errors = {}
            for elem in para.iter():
                tag = strip_namespace(elem.tag)
                if 'orth_error' in tag:
                    tagcode = elem.get('tagcode', '')
                    target = elem.get('orth_error_target', '')
                    if tagcode and target:
                        current_para_orth_errors[tagcode] = target
            
            para_pairs = extract_leonide_sentences(para, unique_paras)
            
            # Check for incomplete sentence at end (existing logic)
            if para_pairs and i + 1 < len(unique_paras):
                last_pair = para_pairs[-1]
                src_incomplete = last_pair.src and not last_pair.src.rstrip().endswith(('.', '!', '?'))
                tgt_incomplete = last_pair.tgt and not last_pair.tgt.rstrip().endswith(('.', '!', '?'))
                
                if src_incomplete or tgt_incomplete:
                    next_pairs = extract_leonide_sentences(unique_paras[i + 1])
                    
                    if next_pairs:
                        next_pair = next_pairs[0]
                        next_src_lower = next_pair.src and (len(next_pair.src) == 0 or next_pair.src[0].islower())
                        next_tgt_lower = next_pair.tgt and (len(next_pair.tgt) == 0 or next_pair.tgt[0].islower())
                        
                        if next_src_lower or next_tgt_lower:
                            merged_src = last_pair.src.rstrip() + ' ' + next_pair.src
                            merged_tgt = last_pair.tgt.rstrip() + ' ' + next_pair.tgt
                            merged_mappings = list(last_pair.orth_mappings) + list(next_pair.orth_mappings)
                            
                            para_pairs[-1] = SentencePair(
                                src=merged_src,
                                tgt=merged_tgt,
                                has_correction=(merged_src.strip() != merged_tgt.strip()),
                                has_foreign=last_pair.has_foreign or next_pair.has_foreign,
                                orth_mappings=merged_mappings
                            )
                            consumed.add(i + 1)
                            all_pairs.extend(para_pairs)
                            all_pairs.extend(next_pairs[1:])
                            last_para_orth_errors = current_para_orth_errors
                            continue
            
            all_pairs.extend(para_pairs)
            last_para_orth_errors = current_para_orth_errors
        
        return all_pairs

    else:  # Kolipsi
        if "Kolipsi_1" in corpus_type or "Kolipsi-1" in corpus_type:
            ns_body = '{http://www.eurac.edu/kolipsi}body'
        else:
            ns_body = '{http://www.eurac.edu/kolipsi_II}body'
    
        body = root.find(f'.//{ns_body}')
        if body is None:
            body = root.find('.//body')
    
        if body is None:
            print(f"[ERROR] No body element found")
            return []
        
        exercises = body.findall('.//exercise')
        if not exercises:
            exercises = [body]

        all_pairs = []
        for ex in exercises:
            if ex is None:
                continue
            pairs = extract_kolipsi_sentences(ex)
            all_pairs.extend(pairs)

        return all_pairs

def clean_sentence_pairs(pairs: List[SentencePair]) -> List[SentencePair]:
    """Clean and deduplicate sentence pairs."""
    debug(f"\n[DEBUG clean_sentence_pairs] INPUT: {len(pairs)} pairs")
    
    cleaned = []
    seen_pairs = set()
    empty_regex = r"^\s*[\.\?!]*\s*$"
    
    filter_counts = {
        'foreign': 0,
        'asterisk': 0,
        'at_symbol': 0,
        'arrow': 0,
        'empty': 0,
        'too_short': 0,
        'duplicate': 0
    }

    for idx, pair in enumerate(pairs):
        original_src = pair.src
        original_tgt = pair.tgt
        
        # Skip foreign words
        if pair.has_foreign:
            filter_counts['foreign'] += 1
            debug(f"  [{idx}] FILTERED (foreign): {original_src[:50]}...")
            continue
        
        src = re.sub(r"\s*\n\s*", " ", pair.src).strip()
        tgt = re.sub(r"\s*\n\s*", " ", pair.tgt).strip()

        # Remove spaces before punctuation for correct German typography
        src = re.sub(r'\s+([.,!?;:])', r'\1', src)
        tgt = re.sub(r'\s+([.,!?;:])', r'\1', tgt)

        # Skip sentences containing arrow ->
        if '->' in src or '->' in tgt:
            filter_counts['arrow'] += 1
            debug(f"  [{idx}] FILTERED (arrow): {src[:50]}...")
            continue

        # INSERT INTERJECTION CODE HERE
        # INTERJECT GOES HERE
        #insert interacion code before this if wanting to restore it

        # Remove leading asterisk bullet points
        src = re.sub(r'^\*\s*', '', src).strip()
        tgt = re.sub(r'^\*\s*', '', tgt).strip()

        # Remove quote encoding errors (&ltt, &gt)
        src = re.sub(r'&\s?[gl]t','', src).strip()
        tgt = re.sub(r'&\s?[gl]t','', tgt).strip()
        # Remove standalone asterisks
        src = re.sub(r'\s*\*\s*', ' ', src).strip()
        tgt = re.sub(r'\s*\*\s*', ' ', tgt).strip()

        # Clean up multiple spaces
        src = re.sub(r'\s+', ' ', src).strip()
        tgt = re.sub(r'\s+', ' ', tgt).strip()

        # Remove leading hyphens
        src = re.sub(r'^-\s+(?=[""„A-ZÄÖÜ])', '', src)
        tgt = re.sub(r'^-\s+(?=[""„A-ZÄÖÜ])', '', tgt)

        # Remove numbered list markers
        src = re.sub(r'\b\d+\)', '', src)
        tgt = re.sub(r'\b\d+\)', '', tgt)

        src = re.sub(r'\s+', ' ', src).strip()
        tgt = re.sub(r'\s+', ' ', tgt).strip()
        
        # Skip multiple asterisks
        if re.search(r'\*{2,}', src) or re.search(r'\*{2,}', tgt):
            filter_counts['asterisk'] += 1
            debug(f"  [{idx}] FILTERED (asterisk): {src[:50]}...")
            continue

        src_lower = src.lower()
        tgt_lower = tgt.lower()

        # Check for @ symbol
        if '@' in src or '@' in tgt:
            filter_counts['at_symbol'] += 1
            debug(f"  [{idx}] FILTERED (@): {src[:50]}...")
            continue

        # Check for empty
        if re.fullmatch(empty_regex, src) or re.fullmatch(empty_regex, tgt):
            filter_counts['empty'] += 1
            debug(f"  [{idx}] FILTERED (empty regex): SRC='{src}' TGT='{tgt}'")
            continue

        # Remove numbered markers
        src = re.sub(r"\s*\d+\)\s*", " ", src).strip()
        tgt = re.sub(r"\s*\d+\)\s*", " ", tgt).strip()

        src = re.sub(r'\s+', ' ', src).strip()
        tgt = re.sub(r'\s+', ' ', tgt).strip()
        
        if not src or not tgt:
            filter_counts['empty'] += 1
            debug(f"  [{idx}] FILTERED (empty after cleanup): SRC='{src}' TGT='{tgt}'")
            continue
        
        # Word count filter
        src_abbrev_collapsed = re.sub(r'\b([a-zA-Z]\.)+', 'ABBREV', src)
        tgt_abbrev_collapsed = re.sub(r'\b([a-zA-Z]\.)+', 'ABBREV', tgt)

        src_words = [w for w in src_abbrev_collapsed.split() if re.search(r'\w', w)]
        tgt_words = [w for w in tgt_abbrev_collapsed.split() if re.search(r'\w', w)]

        if len(src_words) <=4 or len(tgt_words) <= 4:
            filter_counts['too_short'] += 1
            debug(f"  [{idx}] FILTERED (word count): SRC={len(src_words)} words, TGT={len(tgt_words)} words")
            debug(f"        SRC: {src[:60]}...")
            debug(f"        TGT: {tgt[:60]}...")
            continue

                
        pair_key = (src.lower(), tgt.lower())
        if pair_key in seen_pairs:
            filter_counts['duplicate'] += 1
            debug(f"  [{idx}] FILTERED (duplicate): {src[:50]}...")
            continue
        
        seen_pairs.add(pair_key)

        cleaned.append(SentencePair(
            src=src,
            tgt=tgt,
            has_correction=pair.has_correction,
            has_foreign=pair.has_foreign,
            orth_mappings=pair.orth_mappings
        ))

        debug(f"  [{idx}] ✓ KEPT: {src[:60]}... (with {len(pair.orth_mappings)} mappings)")

    debug(f"\n[DEBUG clean_sentence_pairs] OUTPUT: {len(cleaned)} pairs")
    debug(f"[DEBUG FILTER STATS]: {filter_counts}")
    return cleaned
        
def process_file(xml_path: str, corpus_type: str) -> List[SentencePair]:
    """Process a single XML file."""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"{xml_path} not found")

    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        xml_content = f.read()
    
    try:
        # CRITICAL: Each file is a fresh extraction
        pairs = extract_from_xml(xml_content, corpus_type)
        
        # Clean pairs for THIS file only (quotes are stripped at this point)
        cleaned = clean_sentence_pairs(pairs)
        
        return cleaned
    except Exception as e:
        print(f"     ERROR: {e}")
        return []

def process_corpora(
    corpus_configs: Dict[str, Dict],
    output_dir: str = Paths.EXTRACT_DIR,
    max_files_per_corpus: Optional[int] = None,
    output_format: str = "both"  # "txt", "tsv", "norm", or "both"
) -> pd.DataFrame:
    """Process multiple corpora."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    norm_line_map = {}
    
    for corpus_name, cfg in corpus_configs.items():
        print(f"\n--- Processing {corpus_name} ---")

        base_dir = cfg["base_dir"]
        lang_prof = cfg.get("lang_prof", "L2")

        if not os.path.isdir(base_dir):
            print(f"  ERROR: Base directory not found: {base_dir}")
            continue

        xml_members = []
        for root_dir, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d != '.ipynb_checkpoints' and not d.startswith('.')]
            files.sort()
            for f in files:
                if f.lower().endswith(".xml") and not f.lower().endswith(".xml.pretty"):
                    xml_members.append(os.path.join(root_dir, f))

        xml_members.sort()  # Sort full paths to ensure consistent order

        print(f"  Found {len(xml_members)} XML files")

        if max_files_per_corpus:
            xml_members = xml_members[:max_files_per_corpus]

        corpus_pairs_with_files = []  # Changed from corpus_pairs
        for idx, member in enumerate(xml_members):
            xml_filename = os.path.basename(member)
            
            # Skip excluded files
            if xml_filename in ExtractionParams.EXCLUDE:
                print(f"   [{idx + 1}/{len(xml_members)}] {xml_filename} [SKIPPED - excluded]")
                continue
            
            debug(f"   [{idx + 1}/{len(xml_members)}] {member}")

            try:
                pairs = process_file(member, corpus_name)
            except Exception as e:
                print(f"     ERROR in {xml_filename}: {e}")  # Show filename here too
                import traceback
                traceback.print_exc()
                continue
            
            xml_filename = os.path.basename(member)
            corpus_pairs_with_files.append((xml_filename, pairs))

       # Write NORM output if requested (verticalized word-by-word format)
        if output_format in ["norm", "both"]:
            debug(f"\n[DEBUG NORM] Writing NORM output for {corpus_name}...")
            out_path = os.path.join(output_dir, f"{corpus_name}.norm")
            #Track metadata for each sentence
            current_line = 1 # Track line number in .norm file
            with open(out_path, "w", encoding="utf-8") as fh:
                debug(f"[DEBUG NORM] Processing {len(corpus_pairs_with_files)} files...")

                mapping_dict = {}
                target_counts = {}
                # After this line (around line 1556):
                for xml_filename, pairs in corpus_pairs_with_files:
                    for pair_idx, pair in enumerate(pairs):
                        sent_start_line = current_line
                        debug(f"[DEBUG NORM] Pair {pair_idx} has {len(pair.orth_mappings)} mappings: {pair.orth_mappings[:3] if len(pair.orth_mappings) > 3 else pair.orth_mappings}")
                        
                        # ADD THIS DEBUG LINE:
                        debug(f"[DEBUG NORM BEFORE TOKEN] SRC='{pair.src}', TGT='{pair.tgt}'")
                        
                        # Build mapping dict PER PAIR (like display_norm_preview does)
                        mapping_list = pair.orth_mappings
                        mapping_dict = {orig: tgt_map for orig, tgt_map in pair.orth_mappings}
                        used_mapping_indices = set()  # Track which mapping indices we've consumed

                        # Count how many src words map to the same target (for many-to-1 cases)
                        target_counts = {}
                        for orig, tgt_map in pair.orth_mappings:
                            target_counts[tgt_map] = target_counts.get(tgt_map, 0) + 1
                        debug(f"[DEBUG NORM] Pair {pair_idx} has {len(pair.orth_mappings)} mappings: {pair.orth_mappings[:3] if len(pair.orth_mappings) > 3 else pair.orth_mappings}")
         
                        # If we have orth_error mappings, use them for precise alignment
                        if True:
                            src_words = tokenize_preserve_abbrev(pair.src)
                            tgt_words = tokenize_preserve_abbrev(pair.tgt)

                            # ADD THIS DEBUG LINE:
                            debug(f"[DEBUG NORM AFTER TOKEN] src_words={src_words}, tgt_words={tgt_words}")
                            # Pre-compute tokenized lengths for all mapping targets
                            target_token_counts = {}
                            if pair.orth_mappings:
                                for _, tgt_val in pair.orth_mappings:
                                    if tgt_val not in target_token_counts:
                                        target_token_counts[tgt_val] = len(tokenize_preserve_abbrev(tgt_val))

                                                                # Helper function to separate punctuation from word
                            def separate_punct(word):
                                """Separate trailing punctuation, preserving abbreviations."""
                                # Check if it's an abbreviation (protected patterns)
                                if ABBREV_PATTERN.match(word):
                                    return word, ""
                                
                                # Separate trailing punctuation
                                match = re.match(r'^(.*?)([.,!?;:]+)$', word)
                                if match:
                                    return match.group(1), match.group(2)
                                return word, ""

                            def split_punct_for_output(word):
                                """Split word into base + punctuation for NORM output."""
                                if not word:
                                    return [""]
                                
                                # Preserve abbreviations
                                if ABBREV_PATTERN.match(word):
                                    return [word]
                                
                                # CRITICAL: If word is ONLY a quote character, return it as-is
                                if word in QUOTE_CHARS:
                                    return [word]
                                
                                # Split trailing punctuation AND quotes
                                match = re.match(r'^(.*?)([.,!?;:"„""]+)$', word)
                                if match:
                                    base = match.group(1)
                                    punct = match.group(2)
                                    # Further split punct if it contains multiple characters
                                    # e.g., '."' should become ['.', '"']
                                    punct_chars = list(punct)
                                    if base:
                                        return [base] + punct_chars
                                    else:
                                        return punct_chars
                                return [word]

                            # If we have orth_error mappings, use them for precise alignment
                            if pair.orth_mappings:
                                # Group mappings by target to detect splits
                                target_groups = {}
                                for orig, tgt_map in pair.orth_mappings:
                                    if tgt_map not in target_groups:
                                        target_groups[tgt_map] = []
                                    target_groups[tgt_map].append(orig)
                                
                                # Create final mappings with merged sources
                                final_mappings = []
                                for tgt_map, sources in target_groups.items():
                                    if len(sources) > 1:
                                        # Multi-word source for same target
                                        merged = ' '.join(sources)
                                        final_mappings.append((merged, tgt_map))
                                    else:
                                        final_mappings.append((sources[0], tgt_map))
                                
                                # Use final_mappings instead of pair.orth_mappings for alignment
                                src_i = 0
                                tgt_i = 0
                                iteration_count = 0
                                max_iterations = len(src_words) + len(tgt_words) + 100

                                while src_i < len(src_words) and tgt_i < len(tgt_words):
                                    iteration_count += 1
                                    if iteration_count > max_iterations:  # ADD THIS
                                        debug(f"[ERROR] Infinite loop detected at src_i={src_i}, tgt_i={tgt_i}")  # ADD THIS
                                        break  # ADD THIS
                                    src_word = src_words[src_i]
                                    tgt_word = tgt_words[tgt_i]
                                    
                                    
                                    src_word_base, src_punct = separate_punct(src_word)
                                    tgt_word_base, tgt_punct = separate_punct(tgt_word)
                                    
                                    # Use base forms for matching
                                    src_word_clean = src_word_base
                                    tgt_word_clean = tgt_word_base
                                    
                                   # FIRST: Check for multi-word mappings (e.g., "Sprachen oberschule" → "Sprachenoberschule")
                                    found_multiword = False
                                    for orig_key, tgt_val in mapping_dict.items():
                                        if ' ' in orig_key:  # Multi-word source
                                            orig_words_clean = [w.rstrip('.,!?;:') for w in orig_key.split()]
                                            
                                            if src_word_clean == orig_words_clean[0]:
                                                if src_i + len(orig_words_clean) <= len(src_words):
                                                    remaining_clean = [src_words[src_i + j].rstrip('.,!?;:') for j in range(len(orig_words_clean))]
                                                    # CRITICAL: Check if the EXACT sequence exists in mapping (not just similar words)
                                                    if remaining_clean == orig_words_clean:
                                                        # Verify this mapping is for THIS occurrence by checking target alignment
                                                        tgt_val_words = tgt_val.split()
                                                        expected_tgt_clean = tgt_val_words[0].rstrip('.,!?;:') if tgt_val_words else ""
                                                        # Only match if current tgt position matches expected target
                                                        if tgt_i < len(tgt_words) and tgt_words[tgt_i].rstrip('.,!?;:') == expected_tgt_clean:
                                                            src_group = [src_words[src_i + j] for j in range(len(orig_words_clean))]
                                                            # CRITICAL FIX: Split punctuation from target before writing
                                                            tgt_parts = split_punct_for_output(tgt_val)
                                                            fh.write(f"{' '.join(src_group)}\t{tgt_parts[0]}\n")
                                                            current_line += 1
                                                            for punct_part in tgt_parts[1:]:
                                                                fh.write(f"\t{punct_part}\n")
                                                                current_line += 1
                                                            src_i += len(orig_words_clean)
                                                            tgt_i += len(tgt_val_words)
                                                            found_multiword = True
                                                            break
                                                        
                                                    orig_normalized = re.sub(r'\s+', '', orig_key)
                                                    if src_i + len(orig_words_clean) <= len(src_words):
                                                        remaining_clean = [src_words[src_i + j].rstrip('.,!?;:') for j in range(len(orig_words_clean))]
                                                        remaining_normalized = re.sub(r'\s+', '', ' '.join(remaining_clean))
                                                        
                                                        if remaining_normalized == orig_normalized:
                                                            tgt_val_words = tgt_val.split()
                                                            expected_tgt_clean = tgt_val_words[0].rstrip('.,!?;:') if tgt_val_words else ""
                                                            if tgt_i < len(tgt_words) and tgt_words[tgt_i].rstrip('.,!?;:') == expected_tgt_clean:
                                                                src_group = [src_words[src_i + j] for j in range(len(orig_words_clean))]
                                                                # CRITICAL FIX: Split punctuation from target before writing
                                                                tgt_parts = split_punct_for_output(tgt_val)
                                                                fh.write(f"{' '.join(src_group)}\t{tgt_parts[0]}\n")
                                                                current_line += 1
                                                                for punct_part in tgt_parts[1:]:
                                                                    fh.write(f"\t{punct_part}\n")
                                                                    current_line += 1
                                                                src_i += len(orig_words_clean)
                                                                tgt_i += len(tgt_val_words)
                                                                found_multiword = True
                                                                break
                                            
                                    if found_multiword:
                                        continue

                                    # SPECIAL: Check if current word is start of a spaced abbreviation
                                    # e.g., "w." followed by "z." followed by "B" should match "w. z. B" → "wie z.B."
                                    if src_word_clean.endswith('.') and len(src_word_clean) <= 3:  # Short abbreviation fragment
                                        debug(f"[DEBUG ABBREV] Found potential abbreviation start: '{src_word}' (clean: '{src_word_clean}')")

                                        # Look ahead to collect potential multi-part abbreviation
                                        lookahead_words = [src_word]
                                        temp_i = src_i + 1
                                        
                                        # Collect up to 3 more single-letter abbreviations
                                        while temp_i < len(src_words) and len(lookahead_words) < 4:
                                            next_word = src_words[temp_i]
                                            next_clean = next_word.rstrip('.,!?;:')
                                            debug(f"[DEBUG ABBREV] Checking lookahead word: '{next_word}' (clean: '{next_clean}', len={len(next_clean)})")

                                            # Check if it's a single letter with period or just a letter
                                            if (len(next_clean) <= 2 and ('.' in next_word or next_clean.isalpha())):
                                                lookahead_words.append(next_word)
                                                debug(f"[DEBUG ABBREV] Added to lookahead: '{next_word}' (total words: {len(lookahead_words)})")
                                                temp_i += 1
                                            else:
                                                debug(f"[DEBUG ABBREV] Stopped lookahead at: '{next_word}'")
                                                break
                                        
                                        # Try to match the collected sequence against mappings
                                        if len(lookahead_words) > 1:
                                            lookahead_text = ' '.join(lookahead_words)
                                            lookahead_normalized = re.sub(r'\s+', '', lookahead_text.replace('.', '.'))
                                            
                                            debug(f"[DEBUG ABBREV] Lookahead collected {len(lookahead_words)} words: '{lookahead_text}'")
                                            debug(f"[DEBUG ABBREV] Lookahead normalized: '{lookahead_normalized}'")
                                            debug(f"[DEBUG ABBREV] Checking against {len(mapping_dict)} mappings...")
                                                                                        
                                            for orig_key, tgt_val in mapping_dict.items():
                                                orig_normalized = re.sub(r'\s+', '', orig_key)
                                                debug(f"[DEBUG ABBREV]   Comparing with mapping: '{orig_key}' (normalized: '{orig_normalized}') -> '{tgt_val}'")
                                                
                                                if lookahead_normalized == orig_normalized:
                                                    debug(f"[DEBUG ABBREV]   ✓ NORMALIZED MATCH: '{lookahead_text}' == '{orig_key}'")
                                                    # CRITICAL FIX: Split punctuation from target before writing
                                                    tgt_parts = split_punct_for_output(tgt_val)
                                                    fh.write(f"{lookahead_text}\t{tgt_parts[0]}\n")
                                                    current_line += 1
                                                    for punct_part in tgt_parts[1:]:
                                                        fh.write(f"\t{punct_part}\n")
                                                        current_line += 1
                                                    src_i += len(lookahead_words)
                                                    tgt_i += len(tgt_val.split())
                                                    found_multiword = True
                                                    break
                                                elif lookahead_text == orig_key:
                                                    # CRITICAL FIX: Split punctuation from target before writing
                                                    tgt_parts = split_punct_for_output(tgt_val)
                                                    fh.write(f"{lookahead_text}\t{tgt_parts[0]}\n")
                                                    current_line += 1
                                                    for punct_part in tgt_parts[1:]:
                                                        fh.write(f"\t{punct_part}\n")
                                                        current_line += 1
                                                    src_i += len(lookahead_words)
                                                    tgt_i += len(tgt_val.split())
                                                    found_multiword = True
                                                    break
                                                else:
                                                    debug(f"[DEBUG ABBREV]   ✗ No match (normalized: '{lookahead_normalized}' != '{orig_normalized}', exact: '{lookahead_text}' != '{orig_key}')")
                                            
                                            if found_multiword:
                                                debug(f"[DEBUG ABBREV] Successfully matched abbreviation, advancing src_i by {len(lookahead_words)}, tgt_i by {len(tgt_val.split())}")
                                                continue
                                            else:
                                                debug(f"[DEBUG ABBREV] No mapping found for lookahead sequence: '{lookahead_text}'")
                                        else:
                                            debug(f"[DEBUG ABBREV] Only collected 1 word, skipping abbreviation matching")
                                    
                                    # SECOND: Check if current src word has a single-word mapping (fast dict lookup)
                                    if src_word_clean in mapping_dict:
                                        # Find the FIRST unused occurrence of this mapping in the list
                                        matching_idx = None
                                        for idx, (orig, tgt_map) in enumerate(mapping_list):
                                            if idx not in used_mapping_indices and orig == src_word_clean:
                                                matching_idx = idx
                                                break
                                        
                                        # If all occurrences already used, treat as regular word
                                        if matching_idx is None:
                                            fh.write(f"{src_word}\t{tgt_word}\n")
                                            src_i += 1
                                            tgt_i += 1
                                            continue
                                        
                                        # Get the expected target from THIS specific mapping occurrence
                                        expected_tgt = mapping_list[matching_idx][1]
                                        expected_tgt_clean = expected_tgt.rstrip('.,!?;:')
                                        # NEW: Extract punctuation from src_word to restore it later
                                        src_punct = src_word[len(src_word_clean):] if len(src_word) > len(src_word_clean) else ""
                                        
                                        # Check if NEXT consecutive src word ALSO maps to SAME target (many-to-1)
                                        if src_i + 1 < len(src_words):
                                            next_src_clean = src_words[src_i + 1].rstrip('.,!?;:')
                                            
                                            # Check if next word has an UNUSED mapping to the same target
                                            next_has_unused_mapping = False
                                            for idx, (orig, tgt_map) in enumerate(mapping_list):
                                                if idx not in used_mapping_indices and orig == next_src_clean and tgt_map == expected_tgt:
                                                    next_has_unused_mapping = True
                                                    break
                                            
                                            next_tgt_clean = tgt_words[tgt_i + 1].rstrip('.,!?;:') if tgt_i + 1 < len(tgt_words) else None
                                            tgt_repeats = (next_tgt_clean and next_tgt_clean.lower() == expected_tgt_clean.lower())

                                            # Only group if next word has UNUSED mapping AND target doesn't repeat
                                            if (next_has_unused_mapping and 
                                                src_word_clean != expected_tgt_clean and 
                                                next_src_clean != expected_tgt_clean and
                                                tgt_word_clean == expected_tgt_clean and 
                                                not tgt_repeats):
                                                # Collect ALL consecutive src words with unused mappings to same target
                                                src_group = [src_word]
                                                temp_i = src_i + 1
                                                consumed_indices = [matching_idx]
                                                
                                                while temp_i < len(src_words):
                                                    temp_word_clean = src_words[temp_i].rstrip('.,!?;:')
                                                    
                                                    # Find unused mapping for this word
                                                    temp_mapping_idx = None
                                                    for idx, (orig, tgt_map) in enumerate(mapping_list):
                                                        if idx not in used_mapping_indices and idx not in consumed_indices and orig == temp_word_clean and tgt_map == expected_tgt:
                                                            temp_mapping_idx = idx
                                                            break
                                                    
                                                    if temp_mapping_idx is not None:
                                                        src_group.append(src_words[temp_i])
                                                        consumed_indices.append(temp_mapping_idx)
                                                        temp_i += 1
                                                    else:
                                                        break
                                                
                                                # CRITICAL FIX: Split punctuation from tgt_word before writing
                                                tgt_parts = split_punct_for_output(tgt_word)
                                                fh.write(f"{' '.join(src_group)}\t{tgt_parts[0]}\n")
                                                current_line += 1
                                                # Write remaining punctuation on separate lines
                                                for punct_part in tgt_parts[1:]:
                                                    fh.write(f"\t{punct_part}\n")
                                                    current_line += 1
                                                for idx in consumed_indices:
                                                    used_mapping_indices.add(idx)
                                                src_i += len(src_group)
                                                tgt_i += 1
                                                continue
                                        
                                        # Get pre-computed token count for target
                                        expected_tgt_token_count = target_token_counts.get(expected_tgt, 1)
                                        
                                        # Single-word correction (target is also single token)
                                        if expected_tgt_token_count == 1 and tgt_word_clean == expected_tgt_clean:
                                            tgt_with_punct = tgt_word if src_punct == "" else expected_tgt + src_punct
                                            # CRITICAL FIX: Split punctuation before writing
                                            src_parts = split_punct_for_output(src_word)
                                            tgt_parts = split_punct_for_output(tgt_with_punct)
                                            max_parts = max(len(src_parts), len(tgt_parts))
                                            for i in range(max_parts):
                                                s = src_parts[i] if i < len(src_parts) else ""
                                                t = tgt_parts[i] if i < len(tgt_parts) else ""
                                                fh.write(f"{s}\t{t}\n")
                                                current_line += 1
                                            used_mapping_indices.add(matching_idx)
                                            src_i += 1
                                            tgt_i += 1
                                            continue

                                        elif expected_tgt_token_count > 1:
                                            # Multi-token target: verify alignment before advancing
                                            expected_tgt_tokens = tokenize_preserve_abbrev(expected_tgt)
                                            expected_first_clean = expected_tgt_tokens[0].rstrip('.,!?;:')
                                            
                                            # Only apply mapping if current tgt position matches first token
                                            if tgt_word_clean == expected_first_clean:
                                                tgt_with_punct = expected_tgt + src_punct
                                                # CRITICAL FIX: Split punctuation before writing
                                                src_parts = split_punct_for_output(src_word)
                                                tgt_parts = split_punct_for_output(tgt_with_punct)
                                                max_parts = max(len(src_parts), len(tgt_parts))
                                                for i in range(max_parts):
                                                    s = src_parts[i] if i < len(src_parts) else ""
                                                    t = tgt_parts[i] if i < len(tgt_parts) else ""
                                                    fh.write(f"{s}\t{t}\n")
                                                    current_line += 1
                                                used_mapping_indices.add(matching_idx)
                                                src_i += 1
                                                tgt_i += expected_tgt_token_count
                                                continue
                                            # If alignment doesn't match, fall through to default alignment
                                    
                                    # Default: no mapping found, simple word-by-word alignment
                                    src_parts = split_punct_for_output(src_word)
                                    tgt_parts = split_punct_for_output(tgt_word)
                                    max_parts = max(len(src_parts), len(tgt_parts))
                                    for i in range(max_parts):
                                        s = src_parts[i] if i < len(src_parts) else ""
                                        t = tgt_parts[i] if i < len(tgt_parts) else ""
                                        fh.write(f"{s}\t{t}\n")
                                        current_line += 1
                                    src_i += 1
                                    tgt_i += 1

                                # Handle remaining words
                                while src_i < len(src_words):
                                    src_parts = split_punct_for_output(src_words[src_i])
                                    for part in src_parts:
                                        fh.write(f"{part}\t\n")
                                        current_line += 1
                                    src_i += 1

                                while tgt_i < len(tgt_words):
                                    tgt_parts = split_punct_for_output(tgt_words[tgt_i])
                                    for part in tgt_parts:
                                        fh.write(f"\t{part}\n")
                                        current_line += 1
                                    tgt_i += 1
                            else:
                                # No mappings: simple word-by-word alignment
                                max_len = max(len(src_words), len(tgt_words))
                                for i in range(max_len):
                                    src_w = src_words[i] if i < len(src_words) else ""
                                    tgt_w = tgt_words[i] if i < len(tgt_words) else ""
                                    src_parts = split_punct_for_output(src_w) if src_w else [""]
                                    tgt_parts = split_punct_for_output(tgt_w) if tgt_w else [""]
                                    max_parts = max(len(src_parts), len(tgt_parts))
                                    for j in range(max_parts):
                                        s = src_parts[j] if j < len(src_parts) else ""
                                        t = tgt_parts[j] if j < len(tgt_parts) else ""
                                        fh.write(f"{s}\t{t}\n")
                                        current_line += 1

                            fh.write("\n")
                            sent_end_line = current_line #End line is the blank line
                            current_line += 1

                            # Store line mapping
                            norm_line_map[(corpus_name, xml_filename, pair_idx + 1)] = (sent_start_line, sent_end_line)

            total_pairs = sum(len(pairs) for _, pairs in corpus_pairs_with_files)
            print(f"  Wrote {total_pairs} pairs to {out_path}")
            
        for xml_filename, pairs in corpus_pairs_with_files:    
            text_type = "unknown"
            # Detect text type from filename
            if corpus_name in ["Kolipsi_1_L1", "Kolipsi_1_L2", "Kolipsi_2"]:
                # Kolipsi: _1.xml = picture story, _2.xml = opinion
                if xml_filename.endswith("_1.xml"):
                    text_type = "picture story"
                elif xml_filename.endswith("_2.xml"):
                    text_type = "opinion"
            else:  # LEONIDE
                # LEONIDE: "pic" = picture story, "op" = opinion
                if "_pic_" in xml_filename:
                    text_type = "picture story"
                elif "_op_" in xml_filename:
                    text_type = "opinion"
            for sent_num, pair in enumerate(pairs, start=1):
                line_start, line_end = norm_line_map.get((corpus_name, xml_filename, sent_num), (None, None))
                
                all_data.append({
                    'corpus': corpus_name,
                    'lang_prof': lang_prof,
                    'xml_file': xml_filename,
                    'sent_num': sent_num,
                    'src': pair.src,
                    'tgt': pair.tgt,
                    'corrected': pair.has_correction,
                    'text_type': text_type,
                    'line_start': line_start,
                    'line_end': line_end
                })
            
    df = pd.DataFrame(all_data)
    
    # Write TSV output
    if output_format in ["tsv", "both"]:
        tsv_path = os.path.join(output_dir, "all_corpora.tsv")
        df.to_csv(tsv_path, index=False, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE, escapechar=None)
        print(f"\n=== Wrote {len(df)} rows to {tsv_path} ===")
    
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":

    # Setup debug logging to file (not to terminal)  
    logging.basicConfig(
        filename=Paths.EXT_LOG_FILE,
        filemode='w',
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    def debug(msg):
        logging.debug(msg)
        
    parser = argparse.ArgumentParser(description='Extract German learner corpora')
    parser.add_argument('--corpora', nargs='+', default=None,
    help='Specify which corpora to process (e.g., LEONIDE Kolipsi_1_L2)')
    parser.add_argument('--output-dir', default=Paths.EXTRACT_DIR)
    parser.add_argument('--format',
                        default=ExtractionParams.OUTPUT_FORMAT,
                        choices=['tsv', 'norm', 'both'])
    parser.add_argument('--max-files', type=int, default=None)

    # FIX: Determine which corpora to process
    args = parser.parse_args()
    if args.corpora:
        # User specified corpora via command line
        active_corpora = args.corpora
        print(f"Processing user-specified corpora: {active_corpora}")
    elif hasattr(ExtractionParams, 'ACTIVE_CORPORA') and ExtractionParams.ACTIVE_CORPORA:
        # Use ACTIVE_CORPORA from config if defined and not empty
        active_corpora = ExtractionParams.ACTIVE_CORPORA
        print(f"Processing corpora from config: {active_corpora}")
    else:
        # If ACTIVE_CORPORA is None or empty, DON'T process anything
        active_corpora = []
        print(f"⚠️  No corpora specified - extraction disabled.")
    # Filter to only include corpora that exist in CORPORA config
    configs_to_run = {
        k: v for k, v in ExtractionParams.CORPORA.items()
        if k in active_corpora
    }

    # Validate that all requested corpora exist
    missing_corpora = set(active_corpora) - set(configs_to_run.keys())
    if missing_corpora:
        print(f"⚠️  WARNING: The following corpora are not defined in CORPORA config: {missing_corpora}")
        print(f"Available corpora: {list(ExtractionParams.CORPORA.keys())}")

    if configs_to_run:
        print(f"\n{'='*80}")
        print(f"STARTING EXTRACTION")
        print(f"{'='*80}")
        print(f"Corpora to process: {list(configs_to_run.keys())}")
        print(f"Output directory: {args.output_dir}")
        print(f"Output format: {args.format}")
        if args.max_files:
            print(f"Max files per corpus: {args.max_files}")
        print(f"{'='*80}\n")
        
        df = process_corpora(
            corpus_configs=configs_to_run,
            output_dir=args.output_dir,
            output_format=args.format,
            max_files_per_corpus=args.max_files
        )

        if not df.empty:
            print(f"\n{'='*80}")
            print("EXTRACTION SUMMARY")
            print(f"{'='*80}")
            print(f"Total rows: {len(df)}")
            print("\nCorpus breakdown:")
            print(df.groupby(['corpus', 'lang_prof']).size())
  
    else:
        print("❌ No corpora selected or found. Check your configuration.")
        print(f"Available corpora in config: {list(ExtractionParams.CORPORA.keys())}")