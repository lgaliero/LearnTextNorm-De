"""
Configuration file for XML extraction, computing statistics and more.
Contains:
1. Corpus paths and extraction parameters
2. Stats display options
"""

# =======================
# XML EXTRACTION CONFIGS
# =======================

CORPUS_CONFIGS = {
    'LEONIDE': {
        'base_dir': '../corpora/LEONIDE/pepper-xml-v1.1/data/DE',
        'lang_prof': 'L2'
    },
    'Kolipsi_1_L2': {
        'base_dir': '../corpora/Kolipsi_1/xmlmind-v1.1/data/annotations/L2/DE/files_split_by_exercises',
        'lang_prof': 'L2'
    },
    'Kolipsi_1_L1': {
        'base_dir': '../corpora/Kolipsi_1/xmlmind-v1.1/data/annotations/L1/DE/files_split_by_exercises',
        'lang_prof': 'L1'
    },
    'Kolipsi_2': {
        'base_dir': '../corpora/Kolipsi_2',
        'lang_prof': 'L2'
    }
}
#Corpora to process (empty list = process none)
ACTIVE_CORPORA = ['LEONIDE', 'Kolipsi_1_L2', 'Kolipsi_1_L1', 'Kolipsi_2']

# Output settings
OUTPUT_DIR = '../output/extraction'
OUTPUT_FORMAT = 'both'  # Options: "csv", "norm", or "both"

# Processing limits
MAX_FILES_PER_CORPUS = None  # None = process all files, or set to integer to limit

# Sentencizer settings (if needed in future)
SENTENCIZER_KWARGS = None



# =======================
# COMPUTING STATISTICS
# =======================

CSV_PATH = "../output/extraction/all_corpora.csv"

#1 Display overview
MAIN_STATS = True              # Main corpus statistics table

#2 Display sentence count by subcorpus
SUBCORPUS_STATS = True

#3 Correction breakdown by subcorpus
CORRECTION_BREAKDOWN = True    # Correction stats by subcorpus
CORRECTION_SUMMARY = True      # Overall correction summary

#4
CORRECTED_ONLY_STATS =  False   # Detailed stats for corrected pairs only

#5
STATS_PER_TEXT_TYPE = True
TEXT_TYPE_SENTENCE_LEV = False
TEXT_TYPE_DOCUMENT_LEV = False
TEXT_TYPE_COMBO_LEV = True  # Stats for text type

# TEST SET CREATION
SET_SPLITS = "../output/data_splits"
