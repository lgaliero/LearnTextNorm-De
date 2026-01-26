import re
from .data_models import SentencePair
from typing import Tuple, Optional
from .constants import ABBREV_PATTERNS_NORM, ABBREVIATIONS, ABBREV_PATTERN, QUOTE_CHARS

def has_leading_whitespace(text: Optional[str]) -> bool:
    """Check if text starts with whitespace in original XML."""
    return text is not None and len(text) > 0 and text[0].isspace()

def has_trailing_whitespace(text: Optional[str]) -> bool:
    """Check if text ends with whitespace in original XML."""
    return text is not None and len(text) > 0 and text[-1].isspace()

def strip_namespace(tag: str) -> str:
    """Remove XML namespace from tag."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def inject_spaces_between_tags(xml_string: str) -> str:
    """Inject explicit SPACEWRAPPER nodes for meaningful spaces in XML."""
    injected_count = 0
    
    def replacer_text_space_tag(match):
        nonlocal injected_count
        before_gt = match.group(1)
        text = match.group(2)
        spaces = match.group(3)
        after_lt = match.group(4)
        
        if '\n' in spaces or not text.strip():
            return match.group(0)
        
        injected_count += 1
        return f'{before_gt}{text}<SPACEWRAPPER> </SPACEWRAPPER>{after_lt}'
    
    result = re.sub(r'(>)([^<\n]*?\S[^<\n]*?)([ \t]+)(<)', replacer_text_space_tag, xml_string)
    
    def replacer_tag_space_tag(match):
        nonlocal injected_count
        before = match.group(1)
        space = match.group(2)
        after = match.group(3)
        
        if '\n' in space:
            return match.group(0)
        
        injected_count += 1
        return f'{before}<SPACEWRAPPER> </SPACEWRAPPER>{after}'
    
    result = re.sub(r'(>)([ \t]+)(<)', replacer_tag_space_tag, result)
    
    return result
