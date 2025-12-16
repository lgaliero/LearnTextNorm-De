
"""
Configuration file for XML extraction pipeline.
Contains 
1. Corpus paths and extraction parameters.
2. Statistics display options
3. Other parameters
"""

# Corpus configurations
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
OUTPUT_DIR = 'output'
OUTPUT_FORMAT = 'both'  # Options: "txt", "csv", "norm", or "both"

# Processing limits
MAX_FILES_PER_CORPUS = None  # None = process all files, or set to integer to limit

# Sentencizer settings (if needed in future)
SENTENCIZER_KWARGS = None


# For Stats computing
# CONTROL PANEL - CUSTOMIZE YOUR OUTPUT HERE

# 1. Choose data source
#SOURCE = "csv"  # Options: "txt", "csv", or "both"

# 2. Choose which DataFrames to display (True = show, False = hide)
SHOW_MAIN_STATS = True              # Main corpus statistics table
SHOW_CORRECTION_BREAKDOWN = True    # Correction stats by subcorpus
SHOW_CORRECTION_SUMMARY = True      # Overall correction summary
SHOW_CORRECTED_ONLY_STATS =  True   # Detailed stats for corrected pairs only
SHOW_STATS_PER_TEXT_TYPE = True     #