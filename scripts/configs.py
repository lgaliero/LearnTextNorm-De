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
    ACTIVE_CORPORA = ['LEONIDE', 'Kolipsi_1_L2', 'Kolipsi_1_L1', 'Kolipsi_2']   # Corpora to process (empty list = process none)
    OUTPUT_FORMAT = 'both'                                                      # Output settings - Options: "csv", "norm", or "both"
    MAX_FILES_PER_CORPUS = None                                                 # Processing limits - None = process all files, or set to integer to limit
    SENTENCIZER_KWARGS = None                                                   # Sentencizer settings (if needed in future)


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
    TEXT_TYPE_COMBINED = True      #6 Stats for text type

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
    SYS_BASELINE = "Du bekommst deutsche Sätze, einen nach dem anderen. Alle stammen von Lernenden aus Mittel- und Oberschulen; jede Instanz steht außerhalb ihres ursprünglichen Kontexts. Es liegen keine Angaben zu Alter oder Muttersprache vor. Korrigiere ausschließlich Rechtschreibfehler (falsch geschriebene Wörter) und gib den vollständigen Satz in korrigierter Form zurück. Halte dich so weit wie möglich am Inhalt des Ausgangstexts. Füge keine weiteren Ergänzungen hinzu. Grammatikfehler (Wortstellung, Interpunktion, Kasus, falsche Wortarten) nicht korrigieren. Keine Erklärungen, Kommentare, Hinweise, Labels oder Titel, auch wenn der Satz unverständlich ist."
    SYS_2_SHOT = "" #to be uptadted

# 2 shots to be added soon