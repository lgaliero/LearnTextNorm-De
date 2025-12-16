# .sort TO BE ADDED
import re
import os
import spacy
import pandas as pd
from spacy.lang.de import German
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

nlp = German()
nlp.add_pipe("sentencizer", config={"punct_chars": [".", "?", "!"]})

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
    
    def to_tuple(self):
        return (self.src, self.tgt, self.has_correction, self.has_foreign)


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
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.:;!?,])', r'\1', text)
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
    if not text or not text.strip():
        return []

    # Pre-split on double/triple periods
    text = re.sub(r'\.{2,}', '.<SPLIT>', text)
    # Add space after sentence-ending punctuation if missing
    text = re.sub(r'([.!?]+)([A-ZÄÖÜ])', r'\1 \2', text)

    chunks = text.split('<SPLIT>')
    all_sentences = []
    
    for chunk in chunks:
        if not chunk.strip():
            continue
        
        clean = re.sub(r"<[^>]+>", " ", chunk)
        clean = re.sub(r"[ ]+", " ", clean)
        clean = re.sub(r"\n{2,}", "\n<PAR>\n", clean)
        clean = clean.strip()
        
        doc = nlp(clean)
        out = []
        for sent in doc.sents:
            s = sent.text.strip()
            if not s or re.fullmatch(r"[\.\?!]+", s) or re.fullmatch(r"\d+[\.\)]\s*$", s):
                continue
            s = s.replace("<PAR>", "").strip()
            if s:
                out.append(s)
        
        all_sentences.extend(out)

    # Merge fragments
    merged = []
    buffer = ""
    for sentence in all_sentences:
        s_strip = sentence.strip()
        if not s_strip:
            continue
        if buffer:
            if s_strip[0].islower() or not re.search(r'[.!?]$', buffer):
                buffer += " " + s_strip
            else:
                merged.append(buffer)
                buffer = s_strip
        else:
            buffer = s_strip
    
    if buffer:
        merged.append(buffer)

    return merged

# ============================================================================
# KOLIPSI EXTRACTION
# ============================================================================

def extract_kolipsi(element) -> Tuple[str, str, bool]:
    """
    Extract src and tgt from Kolipsi element.
    Returns (src_text, tgt_text, has_corrections)
    """
    src_builder = TextBuilder()
    tgt_builder = TextBuilder()
    has_corrections = False

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

            orig_text = get_original_form_text(original) if original is not None else ""
            tgt_text = get_element_text(target) if target is not None else ""

            # Check for sentence break
            prev_src = src.get_text()
            if (orig_text and tgt_text
                and len(orig_text) > 0 and len(tgt_text) > 0
                and orig_text[0].islower() != tgt_text[0].islower()
                and has_sentence_ending(prev_src)):
                src.add_marker(" <SENTBREAK> ")
                tgt.add_marker(" <SENTBREAK> ")

            if orig_text:
                src.add_text(orig_text)
            if tgt_text:
                tgt.add_text(tgt_text)

            # Handle tail with proper spacing
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                tail_text = node.tail.strip()
                if tail_text:
                    src.add_text(tail_text, merge=True)
                    tgt.add_text(tail_text, merge=True)
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
            deletion_text = ""
            insertion_text = ""
            
            for child in node:
                child_tag = strip_namespace(child.tag).lower()
                
                if child_tag == "deletion":
                    if child.text and child.text.strip():
                        deletion_text = child.text.strip()
                
                elif child_tag == "insertion":
                    if child.text and child.text.strip():
                        insertion_text = child.text.strip()
                    for grandchild in child:
                        recurse(grandchild, src, tgt)
            
            if deletion_text and insertion_text:
                src.add_text(insertion_text, merge=True)
                tgt.add_text(insertion_text, merge=True)
            elif insertion_text and not deletion_text:
                needs_space_before = False
                if src.parts:
                    last_part = src.parts[-1]
                    if last_part and last_part[-1].islower():
                        words = last_part.split()
                        if words and len(words[-1]) > 2:
                            needs_space_before = True
                
                if needs_space_before:
                    src.add_space()
                    tgt.add_space()
                
                src.add_text(insertion_text, merge=True)
                tgt.add_text(insertion_text, merge=True)
            
            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    src.add_text(node.tail.strip(), merge=True)
                    tgt.add_text(node.tail.strip(), merge=True)
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
            first_alternative = None
            for child in node:
                child_tag = strip_namespace(child.tag).lower()
                if child_tag == "alternative":
                    first_alternative = child
                    break
            
            if first_alternative is not None and first_alternative.text:
                alt_text = first_alternative.text.strip()
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
                
                src.add_text(alt_text)
                tgt.add_text(alt_text)
            
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
        elif tag == "strikeover":
            expansions = [child.text for child in node
                          if strip_namespace(child.tag).lower() == "expansion" and child.text]
        
            merged = "".join(expansions)
        
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
                src.add_text(over_text, merge=True)
                tgt.add_text(over_text, merge=True)

            if node.tail:
                if has_leading_whitespace(node.tail):
                    src.add_space()
                    tgt.add_space()
                if node.tail.strip():
                    src.add_text(node.tail.strip(), merge=True)
                    tgt.add_text(node.tail.strip(), merge=True)
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
        elif tag in ("symbol", "emoticon", "unreadable"):
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
            src.add_marker(" <SENTBREAK> ")
            tgt.add_marker(" <SENTBREAK> ")
            
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
        elif tag in ("greeting", "closing", "entity"):
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
    return src_builder.get_text(), tgt_builder.get_text(), has_corrections


def extract_kolipsi_sentences(element) -> List[SentencePair]:
    """Extract sentence pairs from Kolipsi element."""
    src_full, tgt_full, _ = extract_kolipsi(element)

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

        src_sents = spacy_sent(src_chunk) if src_chunk else []
        tgt_sents = spacy_sent(tgt_chunk) if tgt_chunk else []

        if not src_sents and not tgt_sents:
            continue

        max_len = max(len(src_sents), len(tgt_sents))
        for i in range(max_len):
            src_sent = src_sents[i] if i < len(src_sents) else ""
            tgt_sent = tgt_sents[i] if i < len(tgt_sents) else ""
            
            has_correction = (src_sent.strip() != tgt_sent.strip())
            
            if src_sent or tgt_sent:
                pairs.append(SentencePair(
                    src=src_sent,
                    tgt=tgt_sent,
                    has_correction=has_correction,
                    has_foreign=has_foreign_in_chunk
                ))

    return pairs


# ============================================================================
# LEONIDE EXTRACTION
# ============================================================================

def extract_leonide(paragraph) -> Tuple[str, str, bool]:
    """Extract text from LEONIDE paragraph."""
    src_builder = TextBuilder()
    tgt_builder = TextBuilder()
    has_corrections = False

    def process_node(node, src: TextBuilder, tgt: TextBuilder):
        nonlocal has_corrections
        
        if node.text and node.text.strip():
            src.add_text(node.text.strip())
            tgt.add_text(node.text.strip())

        for child in node:
            tag = child.tag.lower()

            # FOREIGN WORD
            if 'tran_foreign_word' in tag:
                foreign_text = child.text.strip() if child.text and child.text.strip() else ""
                
                if foreign_text:
                    marked_word = f'FOREIGNWORDSTART{foreign_text}FOREIGNWORDEND'
                    src.add_text(marked_word)
                    tgt.add_text(marked_word)
                
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

            # SYMBOL / EMOTICON
            if 'tran_symbol' in tag or 'tran_emoticon' in tag:
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue

            # DIV
            if tag == 'div':
                src.add_space()
                tgt.add_space()

                process_node(child, src, tgt)

                if child.tail and child.tail.strip():
                    src.add_space()
                    tgt.add_space()
                    src.add_text(child.tail.strip())
                    tgt.add_text(child.tail.strip())
                else:
                    src.add_space()
                    tgt.add_space()
                continue

            # WORD CORRECTION / AMBIGUOUS
            if 'tran_word_correction' in tag or 'tran_ambiguous' in tag:
                src.add_space()
                tgt.add_space()
                
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

            # WORD DELETION
            if 'tran_word_deletion' in tag:
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue

            # ORTH ERROR
            if 'orth_error' in tag:
                has_corrections = True
                src.add_space()
                tgt.add_space()
                
                target_attr = child.get('orth_error_target')
                
                # Check duplicates
                should_add_target = True
                if target_attr and tgt.parts:
                    recent_text = ' '.join(tgt.parts[-3:]) if len(tgt.parts) >= 3 else ' '.join(tgt.parts)
                    if target_attr in recent_text:
                        should_add_target = False
                
                if target_attr and should_add_target:
                    tgt.add_text(target_attr)
                
                # Get all text from inside orth_error
                original_text = ''.join(child.itertext()).strip()
                if original_text:
                    src.add_text(original_text)
                
                # CRITICAL: Check for leading space in tail
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
                continue
                
            # CAPITALISATION
            if 'tran_capitalisation' in tag:
                original_attr = child.text
                target_attr = child.get('tran_capitalisation_target')
                if original_attr:
                    src.add_text(original_attr)
                if target_attr:
                    tgt.add_text(target_attr)
                    
                if child.tail:
                    if has_leading_whitespace(child.tail):
                        src.add_space()
                        tgt.add_space()
                    if child.tail.strip():
                        src.add_text(child.tail.strip())
                        tgt.add_text(child.tail.strip())
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

    process_node(paragraph, src_builder, tgt_builder)
    return src_builder.get_text(), tgt_builder.get_text(), has_corrections


def extract_leonide_sentences(paragraph) -> List[SentencePair]:
    """Extract sentence pairs from LEONIDE paragraph."""
    src, tgt, _ = extract_leonide(paragraph)

    if not src and not tgt:
        return []

    # Detect foreign words before cleaning markers
    has_foreign = 'FOREIGNWORDSTART' in src or 'FOREIGNWORDSTART' in tgt
    
    # Clean markers
    src = re.sub(r'FOREIGNWORDSTART(.*?)FOREIGNWORDEND', r'\1', src)
    tgt = re.sub(r'FOREIGNWORDSTART(.*?)FOREIGNWORDEND', r'\1', tgt)
    
    src_sents = spacy_sent(src)
    tgt_sents = spacy_sent(tgt)

    max_len = max(len(src_sents), len(tgt_sents))
    pairs = []
    
    for i in range(max_len):
        src_sent = src_sents[i] if i < len(src_sents) else ""
        tgt_sent = tgt_sents[i] if i < len(tgt_sents) else ""
        
        has_correction = (src_sent.strip() != tgt_sent.strip())
        
        if src_sent or tgt_sent:
            pairs.append(SentencePair(
                src=src_sent,
                tgt=tgt_sent,
                has_correction=has_correction,
                has_foreign=has_foreign
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
        all_pairs = []
        for para in paras:
            pairs = extract_leonide_sentences(para)
            all_pairs.extend(pairs)
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
    cleaned = []
    seen_pairs = set()
    empty_regex = r"^\s*[\.\?!]*\s*$"

    for pair in pairs:
        # Skip foreign words
        if pair.has_foreign:
            continue
        
        src = re.sub(r"\s*\n\s*", " ", pair.src).strip()
        tgt = re.sub(r"\s*\n\s*", " ", pair.tgt).strip()

        # Skip asterisks (censored content)
        if '*' in src or '*' in tgt:
            continue

        if re.fullmatch(empty_regex, src) or re.fullmatch(empty_regex, tgt):
            continue

        src = re.sub(r"^\d+[\.\)]\s+", "", src).strip()
        tgt = re.sub(r"^\d+[\.\)]\s+", "", tgt).strip()

        if not src or not tgt:
            continue
        
        # THIS IS THE FILTER FOR SINGLE-WORD SENTENCES
        src_words = re.findall(r'\b\w+\b', src)
        tgt_words = re.findall(r'\b\w+\b', tgt)
        
        if len(src_words) <= 2 or len(tgt_words) <= 2:
            continue
        # END OF FILTER
        
        pair_key = (src.lower(), tgt.lower())
        if pair_key in seen_pairs:
            continue
        
        seen_pairs.add(pair_key)
        cleaned.append(SentencePair(src, tgt, pair.has_correction, pair.has_foreign))

    return cleaned


def process_file(xml_path: str, corpus_type: str) -> List[SentencePair]:
    """Process a single XML file."""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"{xml_path} not found")

    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        xml_content = f.read()

    # CRITICAL: Each file is a fresh extraction
    pairs = extract_from_xml(xml_content, corpus_type)
    
    # Clean pairs for THIS file only
    cleaned = clean_sentence_pairs(pairs)
    
    return cleaned


def process_corpora(
    corpus_configs: Dict[str, Dict],
    output_dir: str = "output",
    max_files_per_corpus: Optional[int] = None,
    output_format: str = "norm"  # "txt", "csv", "norm", or "both"
) -> pd.DataFrame:
    """Process multiple corpora."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    file_id = 1
    
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

        print(f"  Found {len(xml_members)} XML files")

        if max_files_per_corpus:
            xml_members = xml_members[:max_files_per_corpus]

        corpus_pairs = []
        for idx, member in enumerate(xml_members):
            print(f"   [{idx + 1}/{len(xml_members)}] {member}")

            try:
                pairs = process_file(member, corpus_name)
            except Exception as e:
                print(f"     ERROR: {e}")
                continue
            
            xml_filename = os.path.basename(member)
            for sent_num, pair in enumerate(pairs, start=1):
                all_data.append({
                    'corpus': corpus_name,
                    'lang_prof': lang_prof,
                    'xml_file': xml_filename,
                    'file_id': file_id,
                    'sent_num': sent_num,
                    'src': pair.src,
                    'tgt': pair.tgt,
                    'corrected': pair.has_correction
                })
            
            corpus_pairs.extend(pairs)
            file_id += 1

        # Write NORM output if requested (verticalized word-by-word format)
        if output_format in ["norm", "both"]:
            out_path = os.path.join(output_dir, f"{corpus_name}_full.norm")
            with open(out_path, "w", encoding="utf-8") as fh:
                for pair in corpus_pairs:
                    # Split sentences into words
                    src_words = pair.src.split()
                    tgt_words = pair.tgt.split()
                    
                    # Write word pairs (align by position)
                    max_len = max(len(src_words), len(tgt_words))
                    for i in range(max_len):
                        src_word = src_words[i] if i < len(src_words) else ""
                        tgt_word = tgt_words[i] if i < len(tgt_words) else ""
                        fh.write(f"{src_word}\t{tgt_word}\n")
                    
                    # Blank line between sentences
                    fh.write("\n")
            print(f"  Wrote {len(corpus_pairs)} pairs to {out_path}")

    df = pd.DataFrame(all_data)
    
    # Write CSV output
    if output_format in ["csv", "both"]:
        csv_path = os.path.join(output_dir, "all_corpora.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\n=== Wrote {len(df)} rows to {csv_path} ===")
    
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import argparse
    import config
    
    parser = argparse.ArgumentParser(description='Extract German learner corpora')
    parser.add_argument('--corpora', nargs='+', 
                       help='Corpora to process (space-separated)',
                       default=None)
    parser.add_argument('--output-dir', default=config.OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--format', default=config.OUTPUT_FORMAT,
                       choices=['txt', 'csv', 'norm', 'both'],
                       help='Output format')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Max files per corpus (for testing)')
    
    args = parser.parse_args()
    
    # Use command-line args if provided, otherwise use config
    active_corpora = args.corpora if args.corpora else config.ACTIVE_CORPORA
    
    configs_to_run = {
        k: v for k, v in config.CORPUS_CONFIGS.items() 
        if k in active_corpora
    }
    
    if configs_to_run:
        df = process_corpora(
            corpus_configs=configs_to_run,
            output_dir=args.output_dir,
            output_format=args.format,
            max_files_per_corpus=args.max_files
        )
        
        if not df.empty:
            print("\n=== SUMMARY ===")
            print(f"Total rows: {len(df)}")
            print("\nCorpus breakdown:")
            print(df.groupby(['corpus', 'lang_prof']).size())
    else:
        print("No corpora selected.")