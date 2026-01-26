import re
import spacy
from typing import List
from extraction.logger import debug

# Create a blank German pipeline
nlp = spacy.blank("de")

# Add Sentencizer if not present
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

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

    
    # Split at numbered markers IMMEDIATELY - before ANY other processing
    text = re.sub(r'(\S)\s*\d+\)\s*', r'\1<SPLIT>', text)

    # Protect ellipsis inside quotes from being treated as sentence boundary
    # Pattern: „ Text... WORD → should NOT split
    text = re.sub(r'(„[^"]*?)\.\.\.(\s+)([A-ZÄÖÜ])', r'\1ELLIPSISMARKER\2\3', text)

    # Force split at period + space + uppercase (sentence boundaries)
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
        # Force sentence boundary at chunk end (from numbered list splits)
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
    
    # Build regex pattern from actual QUOTE_CHARS set
    quote_pattern = '|'.join(re.escape(q) for q in QUOTE_CHARS)
    
    # Now tokenize: separate punctuation from words
    # Add space before punctuation (except within placeholders)
    tokenized = re.sub(r'([a-zA-ZäöüÄÖÜß0-9_])([.,!?;:)\]])', r'\1 \2', protected)
    # Add space after punctuation
    tokenized = re.sub(r'([.,!?;:])([a-zA-ZäöüÄÖÜß0-9_])', r'\1 \2', tokenized)
    # Handle opening quotes/parentheses
    tokenized = re.sub(r'([\[(])([a-zA-ZäöüÄÖÜß0-9_])', r'\1 \2', tokenized)
    
    # Use .format() instead of f-strings to preserve backreferences
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
