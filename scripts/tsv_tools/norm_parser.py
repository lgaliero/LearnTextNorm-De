"""
NORM file parsing utilities.
Handles parsing of normalized text format files.
"""

from typing import List, Tuple
import re

def parse_norm_file_simple(norm_path: str) -> List[Tuple[str, str, int, int]]:
    """
    Parse NORM file into list of (src_sentence, tgt_sentence, line_start, line_end) tuples.
    
    Args:
        norm_path: Path to NORM file
    
    Returns:
        List of (src, tgt, line_start, line_end) where line_end is the blank line
    """
    sentences = []
    current_src = []
    current_tgt = []
    sent_start_line = 1
    current_line = 1
    
    with open(norm_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Skip comments
            if line.startswith('#'):
                current_line += 1
                continue
            
            # Blank line = sentence boundary
            if not line.strip():
                if current_src or current_tgt:
                    src_sent = ' '.join(current_src).strip()
                    tgt_sent = ' '.join(current_tgt).strip()
            
                    sent_end_line = current_line  # Blank line is the end
                    sentences.append((src_sent, tgt_sent, sent_start_line, sent_end_line))
                    current_src = []
                    current_tgt = []
                    sent_start_line = current_line + 1  # Next sentence starts after blank
                current_line += 1
                continue
            
            # Split on tab OR multiple spaces (2+) to handle editor conversions
            parts = re.split(r'\t|\s{2,}', line)
            
            if len(parts) == 1:
                word = parts[0].strip()
                if word:
                    current_src.append(word)
            elif len(parts) >= 2:
                src_word = parts[0].strip()
                tgt_word = parts[1].strip()
                
                if src_word:
                    current_src.append(src_word)
                if tgt_word:
                    current_tgt.append(tgt_word)
            
            current_line += 1
    
    # Handle last sentence if file doesn't end with blank line
    if current_src or current_tgt:
        src_sent = ' '.join(current_src).strip()
        tgt_sent = ' '.join(current_tgt).strip()
        
        sent_end_line = current_line  # Last line of file
        sentences.append((src_sent, tgt_sent, sent_start_line, sent_end_line))
    
    return sentences


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: strip, lowercase, collapse whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return ' '.join(text.strip().lower().split())
