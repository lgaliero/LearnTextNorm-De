import re
import os
import csv
import copy
import spacy
import argparse
import logging
import pandas as pd
from configs import Paths, ExtractionParams

import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

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
