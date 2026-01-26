# At the top of pipeline.py
import os
import re
import csv
import pandas as pd
from typing import List, Dict, Optional
from configs import Paths, ExtractionParams
from .logger import debug
from .data_models import SentencePair
from .pair_builder import PairBuilder
from .file_formats import NormWriter
from .xml_helpers import inject_spaces_between_tags, strip_namespace
from .text_utils import tokenize_preserve_abbrev
from .constants import ABBREV_PATTERN, QUOTE_CHARS
import xml.etree.ElementTree as ET

class TextExtractor:
    """Handles extraction pipeline from XML content."""
    
    def __init__(self, corpus_type: str):
        self.corpus_type = corpus_type
    
    def extract(self, xml_content: str) -> List[SentencePair]:
        """Main extraction function."""
        # Inject space wrappers
        xml_content = inject_spaces_between_tags(xml_content)

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"[ERROR] XML Parse Error: {e}")
            return []

        if self.corpus_type == "LEONIDE":
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
                
                para_pairs = PairBuilder.from_leonide(para, unique_paras)
                
                # Check for incomplete sentence at end (existing logic)
                if para_pairs and i + 1 < len(unique_paras):
                    last_pair = para_pairs[-1]
                    src_incomplete = last_pair.src and not last_pair.src.rstrip().endswith(('.', '!', '?'))
                    tgt_incomplete = last_pair.tgt and not last_pair.tgt.rstrip().endswith(('.', '!', '?'))
                    
                    if src_incomplete or tgt_incomplete:
                        next_pairs = PairBuilder.from_leonide(unique_paras[i + 1])
                        
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
            if "Kolipsi_1" in self.corpus_type or "Kolipsi-1" in self.corpus_type:
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
                pairs = PairBuilder.from_kolipsi(ex)
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
        # Inject space wrappers
        xml_content = inject_spaces_between_tags(xml_content)
        
        extractor = TextExtractor(corpus_type)
        pairs = extractor.extract(xml_content)
        
        # Clean pairs
        cleaned = clean_sentence_pairs(pairs)
        
        return cleaned
    except Exception as e:
        print(f"     ERROR: {e}")
        import traceback
        traceback.print_exc()
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

            with NormWriter(out_path) as norm_writer:
                print(f"[DEBUG] NormWriter opened successfully")
                debug(f"[DEBUG NORM] Processing {len(corpus_pairs_with_files)} files...")
                current_line = 1
                
                for xml_filename, pairs in corpus_pairs_with_files:
                    for pair_idx, pair in enumerate(pairs):
                        sent_start_line = current_line 
                        debug(f"[DEBUG NORM] Pair {pair_idx} has {len(pair.orth_mappings)} mappings: {pair.orth_mappings[:3] if len(pair.orth_mappings) > 3 else pair.orth_mappings}")
                        
                        debug(f"[DEBUG NORM BEFORE TOKEN] SRC='{pair.src}', TGT='{pair.tgt}'")
                        
                        # Build mapping dict PER PAIR
                        mapping_dict = {orig: tgt_map for orig, tgt_map in pair.orth_mappings}
                        mapping_list = list(pair.orth_mappings)
                        used_mapping_indices = set()
                        
                        # Count how many src words map to the same target
                        target_counts = {}
                        for orig, tgt_map in pair.orth_mappings:
                            target_counts[tgt_map] = target_counts.get(tgt_map, 0) + 1
                        
                        # Tokenize
                        src_words = tokenize_preserve_abbrev(pair.src)
                        tgt_words = tokenize_preserve_abbrev(pair.tgt)
                        
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
                            if ABBREV_PATTERN.match(word):
                                return word, ""
                            match = re.match(r'^(.*?)([.,!?;:]+)$', word)
                            if match:
                                return match.group(1), match.group(2)
                            return word, ""
                        
                        def split_punct_for_output(word):
                            """Split word into base + punctuation for NORM output."""
                            if not word:
                                return [""]
                            if ABBREV_PATTERN.match(word):
                                return [word]
                            if word in QUOTE_CHARS:
                                return [word]
                            match = re.match(r'^(.*?)([.,!?;:"„""]+)$', word)
                            if match:
                                base = match.group(1)
                                punct = match.group(2)
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
                                                # Check if the EXACT sequence exists in mapping (not just similar words)
                                                if remaining_clean == orig_words_clean:
                                                    # Verify this mapping is for THIS occurrence by checking target alignment
                                                    tgt_val_words = tgt_val.split()
                                                    expected_tgt_clean = tgt_val_words[0].rstrip('.,!?;:') if tgt_val_words else ""
                                                    # Only match if current tgt position matches expected target
                                                    if tgt_i < len(tgt_words) and tgt_words[tgt_i].rstrip('.,!?;:') == expected_tgt_clean:
                                                        src_group = [src_words[src_i + j] for j in range(len(orig_words_clean))]
                                                        # Split punctuation from target before writing
                                                        tgt_parts = split_punct_for_output(tgt_val)
                                                        norm_writer.write(f"{' '.join(src_group)}\t{tgt_parts[0]}\n")
                                                        current_line += 1
                                                        for punct_part in tgt_parts[1:]:
                                                            norm_writer.write(f"\t{punct_part}\n")
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
                                                            norm_writer.write(f"{' '.join(src_group)}\t{tgt_parts[0]}\n")
                                                            current_line += 1
                                                            for punct_part in tgt_parts[1:]:
                                                                norm_writer.write(f"\t{punct_part}\n")
                                                                current_line += 1
                                                            src_i += len(orig_words_clean)
                                                            tgt_i += len(tgt_val_words)
                                                            found_multiword = True
                                                            break
                                        
                                if found_multiword:
                                    continue

                                # Check if current word is start of a spaced abbreviation
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
                                                # Split punctuation from target before writing
                                                tgt_parts = split_punct_for_output(tgt_val)
                                                norm_writer.write(f"{lookahead_text}\t{tgt_parts[0]}\n")
                                                current_line += 1
                                                for punct_part in tgt_parts[1:]:
                                                    norm_writer.write(f"\t{punct_part}\n")
                                                    current_line += 1
                                                src_i += len(lookahead_words)
                                                tgt_i += len(tgt_val.split())
                                                found_multiword = True
                                                break
                                            elif lookahead_text == orig_key:
                                                # Split punctuation from target before writing
                                                tgt_parts = split_punct_for_output(tgt_val)
                                                norm_writer.write(f"{lookahead_text}\t{tgt_parts[0]}\n")
                                                current_line += 1
                                                for punct_part in tgt_parts[1:]:
                                                    norm_writer.write(f"\t{punct_part}\n")
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
                                
                                # Check if current src word has a single-word mapping (fast dict lookup)
                                if src_word_clean in mapping_dict:
                                    # Find the FIRST unused occurrence of this mapping in the list
                                    matching_idx = None
                                    for idx, (orig, tgt_map) in enumerate(mapping_list):
                                        if idx not in used_mapping_indices and orig == src_word_clean:
                                            matching_idx = idx
                                            break
                                    
                                    # If all occurrences already used, treat as regular word
                                    if matching_idx is None:
                                        norm_writer.write(f"{src_word}\t{tgt_word}\n")
                                        src_i += 1
                                        tgt_i += 1
                                        continue
                                    
                                    # Get the expected target from THIS specific mapping occurrence
                                    expected_tgt = mapping_list[matching_idx][1]
                                    expected_tgt_clean = expected_tgt.rstrip('.,!?;:')
                                    # Extract punctuation from src_word to restore it later
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
                                            
                                            # Split punctuation from tgt_word before writing
                                            tgt_parts = split_punct_for_output(tgt_word)
                                            norm_writer.write(f"{' '.join(src_group)}\t{tgt_parts[0]}\n")
                                            current_line += 1
                                            # Write remaining punctuation on separate lines
                                            for punct_part in tgt_parts[1:]:
                                                norm_writer.write(f"\t{punct_part}\n")
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
                                        # Split punctuation before writing
                                        src_parts = split_punct_for_output(src_word)
                                        tgt_parts = split_punct_for_output(tgt_with_punct)
                                        max_parts = max(len(src_parts), len(tgt_parts))
                                        for i in range(max_parts):
                                            s = src_parts[i] if i < len(src_parts) else ""
                                            t = tgt_parts[i] if i < len(tgt_parts) else ""
                                            norm_writer.write_word_pair(s, t)
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
                                            # Split punctuation before writing
                                            src_parts = split_punct_for_output(src_word)
                                            tgt_parts = split_punct_for_output(tgt_with_punct)
                                            max_parts = max(len(src_parts), len(tgt_parts))
                                            for i in range(max_parts):
                                                s = src_parts[i] if i < len(src_parts) else ""
                                                t = tgt_parts[i] if i < len(tgt_parts) else ""
                                                norm_writer.write_word_pair(s, t)
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
                                    norm_writer.write_word_pair(s, t)
                                    current_line += 1
                                
                                src_i += 1
                                tgt_i += 1
                            
                            # Handle remaining words
                            while src_i < len(src_words):
                                src_parts = split_punct_for_output(src_words[src_i])
                                for part in src_parts:
                                    norm_writer.write_word_pair(part, "")
                                    current_line += 1
                                src_i += 1
                            
                            while tgt_i < len(tgt_words):
                                tgt_parts = split_punct_for_output(tgt_words[tgt_i])
                                for part in tgt_parts:
                                    norm_writer.write_word_pair("", part)
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
                                    norm_writer.write_word_pair(s, t)
                                    current_line += 1
                        
                        # End sentence
                        norm_writer.write_blank_line() 
                        sent_end_line = current_line  # FIXED - blank line IS the end marker
                        current_line += 1  # FIXED - increment AFTER setting sent_end_line
                        norm_writer.end_sentence(corpus_name, xml_filename, pair_idx + 1, sent_start_line, sent_end_line)  # FIXED - added sent_end_line parameter 
            print(f"\n[DEBUG] About to write NORM for {corpus_name}")
            print(f"[DEBUG] output_format = '{output_format}'")
            print(f"[DEBUG] Number of files with pairs: {len(corpus_pairs_with_files)}")
            total_pairs = sum(len(pairs) for _, pairs in corpus_pairs_with_files)
            print(f"[DEBUG] Total pairs collected: {total_pairs}")
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
                line_start, line_end = norm_writer.line_map.get((corpus_name, xml_filename, sent_num), (None, None))
                
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
