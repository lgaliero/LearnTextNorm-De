from typing import Dict, Tuple, Optional
from pathlib import Path

class NormWriter:
    """
    Handles writing sentence pairs to .norm format (verticalized word alignment).
    
    Wraps a file handle and tracks line numbers for each sentence.
    """
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.line_map: Dict[Tuple[str, str, int], Tuple[int, int]] = {}
        self._file_handle = None
    
    def __enter__(self):
        self._file_handle = open(self.output_path, 'w', encoding='utf-8')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle:
            self._file_handle.close()
    
    def write(self, text: str):
        """Write raw text to file (matches fh.write())."""
        self._file_handle.write(text)

    def write_word_pair(self, src_word: str, tgt_word: str):
        """Write a single word alignment pair."""
        self._file_handle.write(f"{src_word}\t{tgt_word}\n")
        # NO line tracking here - caller handles current_line


    def write_blank_line(self):
        """Write blank line separator between sentences."""
        self._file_handle.write("\n")

    def end_sentence(self, corpus: str, xml_file: str, sent_num: int, start_line: int, end_line: int):
        """Store line mapping for a sentence."""
        self.line_map[(corpus, xml_file, sent_num)] = (start_line, end_line)