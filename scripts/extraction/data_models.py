import re
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SentencePair:
    """Represents a source-target sentence pair with metadata."""
    src: str
    tgt: str
    has_correction: bool
    has_foreign: bool
    orth_mappings: List[Tuple[str, str, int]] = None
    
    def __post_init__(self):
        if self.orth_mappings is None:
            self.orth_mappings = []
    
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
        # Preserve all original punctuation spacing
        return text.strip()