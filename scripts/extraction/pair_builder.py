# scripts/extraction/pair_builder.py
from typing import List
from .data_models import SentencePair
from .kolipsi import extract_kolipsi
from .leonide import extract_leonide  # ← Make sure this is here
from .sentencizer_de import spacy_sent
from .text_utils import strip_quotes_preserve_original, restore_quotes_to_sentence
from .logger import debug
import re

class PairBuilder:
    """Handles sentence pair extraction from parsed XML elements."""
    
    @staticmethod
    def from_kolipsi(element) -> List[SentencePair]:
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

    @staticmethod
    def from_leonide(paragraph, all_paragraphs=None) -> List[SentencePair]:
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

            debug(f"[DEBUG SRC (cleaned)]: '{src[:200]}'")
            debug(f"[DEBUG TGT (cleaned)]: '{tgt[:200]}'")

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
