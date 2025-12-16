"""
Configuration file for XML extraction pipeline.
Contains corpus paths and extraction parameters.
"""

# Corpus configurations
CORPUS_CONFIGS = {
    'LEONIDE': {
        'base_dir': 'corpora/LEONIDE/pepper-xml-v1.1/data/DE',
        'lang_prof': 'L2'
    },
    'Kolipsi_1_L2': {
        'base_dir': 'corpora/Kolipsi_1/xmlmind-v1.1/data/annotations/L2/DE/files_split_by_exercises',
        'lang_prof': 'L2'
    },
    'Kolipsi_1_L1': {
        'base_dir': 'corpora/Kolipsi_1/xmlmind-v1.1/data/annotations/L1/DE/files_split_by_exercises',
        'lang_prof': 'L1'
    },
    'Kolipsi_2': {
        'base_dir': 'corpora/Kolipsi_2/xmlmind-v1.1/data/annotations/DE/files_split_by_exercises',
        'lang_prof': 'L2'
    }
}

# Active corpora to process (empty list = process none)
ACTIVE_CORPORA = ['LEONIDE', 'Kolipsi_1_L2', 'Kolipsi_1_L1', 'Kolipsi_2']

# Output settings
OUTPUT_DIR = 'output'
OUTPUT_FORMAT = 'norm'  # Options: "txt", "csv", "norm", or "both"

# Processing limits
MAX_FILES_PER_CORPUS = None  # None = process all files, or set to integer to limit

# Sentencizer settings (if needed in future)
SENTENCIZER_KWARGS = None