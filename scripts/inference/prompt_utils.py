"""
Utilities for loading and managing prompt examples from JSON files.
"""

import json
from typing import List, Dict, Optional, Tuple

Example = Tuple[str, str]


def load_examples_json(json_path: str) -> List[Dict]:
    """
    Load examples from JSON file.
    
    Args:
        json_path: Path to JSON file with examples
        
    Returns:
        List of example dictionaries
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []


def find_examples_for_sentence(sentence: str, examples_data: List[Dict]) -> Optional[Dict]:
    """
    Find the example entry matching the given test sentence.
    
    Args:
        sentence: Test sentence to find examples for
        examples_data: List of example dictionaries
        
    Returns:
        Dictionary with examples if found, None otherwise
    """
    for entry in examples_data:
        if entry.get('test_source') == sentence:
            return entry
    return None


def get_examples_interactively() -> List[Example]:
    """
    Prompt user to input two example pairs interactively.
    
    Returns:
        List of (source, target) tuples
    """
    examples = []
    
    print("\n--- Enter 2 example pairs ---")
    for i in range(2):
        print(f"\nExample {i+1}:")
        src = input(f"  Source {i+1} > ").strip()
        tgt = input(f"  Target {i+1} > ").strip()
        
        if not src or not tgt:
            print("  Warning: Empty example, skipping...")
            continue
            
        examples.append((src, tgt))
    
    return examples


def extract_examples_from_entry(entry: Dict, count: int = 2) -> List[Example]:
    """
    Extract example pairs from an entry dictionary.
    
    Args:
        entry: Dictionary with 'examples' key
        count: Number of examples to extract
        
    Returns:
        List of (source, target) tuples
    """
    return [
        (ex['source'], ex['target']) 
        for ex in entry.get('examples', [])[:count]
    ]
