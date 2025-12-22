"""
Configuration file for XML extraction, computing statistics and more.
Contains:
1. Corpus paths and extraction parameters
2. Stats display options
"""
class Paths: 
    EXTRACT_OUT = '../output/extraction'  
    EXTRACT_CSV = "../output/extraction/all_corpora.csv"
    SET_SPLITS = "../output/data_split"
    MODELS = "../output/results"
    LLM_BASE = "../output/results/llm_prompting/LLaMA3_2_base.tgt"
    LLM_2S = "../output/results/llm_prompting/LLaMA3_2_2S.tgt"

# =======================
# XML EXTRACTION CONFIGS
# =======================
class ExtractionParams:
    """Configuration for extraction process."""
    CORPORA = {
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
    ACTIVE_CORPORA = ['LEONIDE']  # Corpora to process (empty list = process none)
    OUTPUT_FORMAT = 'norm'         # Output settings - Options: "csv", "norm", or "both"
    EXCLUDE = ["DE_pic_2_57Y25A14_59.xml"," DE_pic_2_57Y25A03_59.xml", "DE_pic_3_67Y25A21_112.xml"]
    MAX_FILES_PER_CORPUS = None    # Processing limits - None = process all files, or set to integer to limit
    SENTENCIZER_KWARGS = None      # Sentencizer settings (if needed in future)


# =======================
# COMPUTING STATISTICS
# =======================
class StatsDisplay:
    MAIN_STATS = True               #1 Display overview # Main corpus statistics table
    SUBCORPUS_STATS = True          #2 Display sentence count by subcorpus
    CORRECTION_BREAKDOWN = True     #3A Correction breakdown by subcorpus # Correction stats by subcorpus
    CORRECTION_SUMMARY = True       #3B Correction breakdown by subcorpus # Overall correction summary
    CORRECTED_ONLY_STATS =  False   #4 Detailed stats for corrected pairs only
    STATS_PER_TEXT_TYPE = True      #5A
    TEXT_TYPE_SENTENCE_LEV = False  #5B
    TEXT_TYPE_DOCUMENT_LEV = False  #5C
    TEXT_TYPE_COMBINED = True       #6 Stats for text type

# =======================
# TEST SET CREATION
# =======================
class DataSplits:
    TEST = 0.10
    DEV = 0.10
    TRAIN = 0.80



# =======================
# LLM TESTING (via institution API)
# =======================
class ApiConfig:
    HOST = "http://51.124.247.170:80"
    MODEL = "llama3.2:latest"
    MODE = "baseline"  # or "2-shot"
    SYS_BASELINE = "Du bekommst deutsche Sätze, die von Lernenden aus Mittel- und Oberschulen geschrieben wurden. Korrigiere nur orthographische Fehler, falls vorhanden (falsche Buchstaben, Groß- und Kleinschreibung, Umlaute, ß/ss). Gib immer nur den vollständigen Satz zurück, ohne Kommentare oder Labels - auch wenn der Satz unverständlich ist. Weitere Änderungen oder Ergänzungen des Ausgangstexts sind nicht erlaubt."
    SYS_2_SHOT = "" #to be uptadted

# 2 shots to be added soon