"""
Raw corpus statistics computation.
Analyzes XML files before extraction/filtering.
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict


def count_sentences_in_xml(xml_path: str, corpus_type: str) -> int:
    """
    Count sentences in raw XML using sentence-ending punctuation.
    
    Args:
        xml_path: Path to XML file
        corpus_type: Corpus name for logging
        
    Returns:
        Approximate sentence count based on punctuation
    """
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            xml_content = f.read()
        
        # Simple XML parsing without full extraction pipeline
        root = ET.fromstring(xml_content)
        
        # Get all text content
        text = ''.join(root.itertext())
        
        # Count sentence-ending punctuation
        return len(re.findall(r'[.!?]+', text))
    
    except Exception as e:
        print(f"    ERROR reading {os.path.basename(xml_path)}: {e}")
        return 0


def count_tokens_in_xml(xml_path: str, tokenize_func) -> Dict[str, int]:
    """
    Count tokens in raw XML.
    
    Args:
        xml_path: Path to XML file
        tokenize_func: Tokenization function to use
        
    Returns:
        Dict with 'src' and 'tgt' token counts (same for raw XML)
    """
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            xml_content = f.read()
        
        root = ET.fromstring(xml_content)
        text = ''.join(root.itertext())
        tokens = tokenize_func(text)
        
        # Return same count for src and tgt (raw XML doesn't separate cleanly)
        return {'src': len(tokens), 'tgt': len(tokens)}
    
    except Exception as e:
        print(f"    ERROR reading {os.path.basename(xml_path)}: {e}")
        return {'src': 0, 'tgt': 0}


def compute_raw_stats(corpus_configs: Dict[str, Dict], tokenize_func) -> Dict[str, Dict]:
    """
    Compute statistics from raw XML files (before extraction/filtering).
    
    Args:
        corpus_configs: Dict mapping corpus name to config with 'base_dir' key
        tokenize_func: Tokenization function to use (e.g., tokenize_for_stats)
        
    Returns:
        Dict mapping corpus name to stats dict with keys:
        - files: Number of XML files
        - sentences: Approximate sentence count
        - tokens_src: Token count (source)
        - tokens_tgt: Token count (target, same as src for raw)
    """
    stats = {}
    
    for corpus_name, cfg in corpus_configs.items():
        base_dir = cfg.get("base_dir", cfg.get("xml_dir", ""))
        
        if not os.path.isdir(base_dir):
            print(f"  WARNING: Directory not found: {base_dir}")
            stats[corpus_name] = {
                'files': 0,
                'sentences': 0,
                'tokens_src': 0,
                'tokens_tgt': 0
            }
            continue
        
        # Find all XML files
        xml_files = []
        for root_dir, dirs, files in os.walk(base_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in sorted(files):
                if f.lower().endswith(".xml") and not f.endswith(".pretty"):
                    xml_files.append(os.path.join(root_dir, f))
        
        # Count statistics
        total_sentences = 0
        total_tokens_src = 0
        total_tokens_tgt = 0
        
        for xml_path in xml_files:
            sentences = count_sentences_in_xml(xml_path, corpus_name)
            tokens = count_tokens_in_xml(xml_path, tokenize_func)
            
            total_sentences += sentences
            total_tokens_src += tokens['src']
            total_tokens_tgt += tokens['tgt']
        
        stats[corpus_name] = {
            'files': len(xml_files),
            'sentences': total_sentences,
            'tokens_src': total_tokens_src,
            'tokens_tgt': total_tokens_tgt
        }
    
    return stats
