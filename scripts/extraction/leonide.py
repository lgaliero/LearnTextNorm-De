
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
from .data_models import TextBuilder
from .constants import QUOTE_CHARS, ABBREVIATIONS
from .xml_helpers import (
    strip_namespace,
    has_leading_whitespace,
    has_trailing_whitespace,
    has_sentence_ending
)
from .logger import debug

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
    