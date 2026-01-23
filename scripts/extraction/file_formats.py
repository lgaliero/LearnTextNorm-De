from typing import Dict, Tuple, Optional
from pathlib import Path

class NormWriter:
    """
    Handles writing sentence pairs to .norm format (verticalized word alignment).
    
    Format:
        word1_src    word1_tgt
        word2_src    word2_tgt
        <blank line>
    
    Tracks line numbers for each sentence: (corpus, xml_file, sent_num) -> (start, end)
    """
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.current_line = 1
        self.line_map: Dict[Tuple[str, str, int], Tuple[int, int]] = {}
        self._file_handle = None
    
    def __enter__(self):
        self._file_handle = open(self.output_path, 'w', encoding='utf-8')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle:
            self._file_handle.close()
    
    def write_word_pair(self, src_word: str, tgt_word: str):
        """Write a single word alignment pair."""
        self._file_handle.write(f"{src_word}\t{tgt_word}\n")
        self.current_line += 1
    
    def write_blank_line(self):
        """Write blank line separator between sentences."""
        self._file_handle.write("\n")
        self.current_line += 1
    
    def start_sentence(self) -> int:
        """Mark the start of a new sentence, return starting line number."""
        return self.current_line
    
    def end_sentence(self, corpus: str, xml_file: str, sent_num: int, start_line: int):
        """
        Mark the end of a sentence and record line mapping.
        
        Args:
            corpus: Corpus name
            xml_file: Source XML filename
            sent_num: Sentence number within file
            start_line: Line number where sentence started
        """
        end_line = self.current_line  # Blank line is the end marker
        self.line_map[(corpus, xml_file, sent_num)] = (start_line, end_line)
    
    def get_line_mapping(self, corpus: str, xml_file: str, sent_num: int) -> Tuple[Optional[int], Optional[int]]:
        """Retrieve line range for a specific sentence."""
        return self.line_map.get((corpus, xml_file, sent_num), (None, None))
    
    def get_all_mappings(self) -> Dict[Tuple[str, str, int], Tuple[int, int]]:
        """Return complete line mapping dictionary."""
        return self.line_map.copy()