"""
File I/O utilities for saving inference outputs.
"""

import os


def append_to_tgt(text: str, mode: str, model: str, paths_config: dict):
    """
    Append model output to the appropriate target file.
    
    Args:
        text: Output text to save
        mode: Inference mode ('baseline' or '2-shot-json')
        model: Model identifier (e.g., 'llama3.2:latest')
        paths_config: Dictionary with path mappings (LLAMA_0, GPT_0, etc.)
        
    Raises:
        ValueError: If model or mode is unknown
    """
    model_short = model.split(':')[0].replace('.', '_')
    
    # Determine output path based on model and mode
    if mode == "baseline":
        if 'llama' in model_short:
            path = paths_config.get('LLAMA_0')
        elif 'gpt' in model_short:
            path = paths_config.get('GPT_0')
        elif 'gemma' in model_short:
            path = paths_config.get('GEMMA_0')
        else:
            raise ValueError(f"Unknown model: {model}")
    elif mode == "2-shot-json":
        if 'llama' in model_short:
            path = paths_config.get('LLAMA_2')
        elif 'gpt' in model_short:
            path = paths_config.get('GPT_2')
        elif 'gemma' in model_short:
            path = paths_config.get('GEMMA_2S')
        else:
            raise ValueError(f"Unknown model: {model}")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if not path:
        raise ValueError(f"No path configured for model={model}, mode={mode}")
    
    # Collapse to single line
    single_line = ' '.join(text.split())
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    with open(path, "a+", encoding="utf-8") as f:
        f.write(single_line + "\n")


def load_sentences_from_file(filepath: str) -> list:
    """
    Load sentences from a file (one per line).
    
    Args:
        filepath: Path to input file
        
    Returns:
        List of sentences (strings)
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
