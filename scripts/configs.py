"""
Configuration file for the normalization pipeline, form XML extraction to evaluation
Contains:
1. File paths 
2. XML extraction parameters
3. Stats display options
4. Dataset splitting 
5. API configurations and system messages for model testing in 0-shot and 2-shot
"""
class Paths: 
    EXT_LOG_FILE ="../master_files/extraction_debug.log"
    EXTRACT_DIR = '../master_files'  
    EXTRACT_TSV = "../master_files/all_corpora.tsv"
    SET_SPLITS = "../sets"
    TEST_SRC = "../sets/test/test.src"
    TEST_TGT = "../sets/test/test.tgt"
    TEST_IDXS = "../sets/test/test_meta.tsv"
    DEV_SRC = "../sets/dev/dev.src"
    DEV_TGT = "../sets/dev/dev.tgt"
    DEV_IDXS = "../sets/dev/dev_meta.tsv"
    TRAIN_SRC = "../sets/train.src"
    TRAIN_TGT = "../sets/train.tgt"
    TRAIN_IDXS = "../sets/train/train_meta.tsv"
    JSON = "../sets/2S_prompts.json" 
    MODELS = "../hypothesis/llama3-2"
    LLM_BASE = "../hypothesis/llama3-2/0shot_raw.hyp"
    LLM_2S = "../hypothesis/llama3-2/2shot_raw.hyp"
    GPT_BASE = "../hypothesis/gpt-oss/0shot_raw.hyp"
    GPT_2S = "../hypothesis/gpt-oss/2shot_raw.hyp"
    GEMMA_BASE = "../hypothesis/gemma/2shot_raw.hyp"
    GEMMA_2S = "../hypothesis/gemma/2shot_raw.hyp"

# =======================
# XML EXTRACTION CONFIGS
# =======================
class ExtractionParams:
    """Configuration for extraction process."""
    CORPORA = {
        'Kolipsi_1_L1': {
            'base_dir': '../corpora/Kolipsi_1/xmlmind-v1.1/data/annotations/L1/DE/files_split_by_exercises',
            'lang_prof': 'L1'
        },
        'Kolipsi_1_L2': {
            'base_dir': '../corpora/Kolipsi_1/xmlmind-v1.1/data/annotations/L2/DE/files_split_by_exercises',
            'lang_prof': 'L2'
        },
        'Kolipsi_2': {
            'base_dir': '../corpora/Kolipsi_2',
            'lang_prof': 'L2'
        },
        'LEONIDE': {
            'base_dir': '../corpora/LEONIDE/pepper-xml-v1.1/data/DE',
            'lang_prof': 'L2'
        }
    }
    ACTIVE_CORPORA = ['LEONIDE', 'Kolipsi_1_L1', "Kolipsi_1_L2", "Kolipsi_2"] # Corpora to process (empty list = process none)
    hypothesis_FORMAT = 'both'      # hypothesis settings - Options: "tsv", "norm", or "both"
    EXCLUDE = ["DE_pic_2_57Y25A14_59.xml"," DE_pic_2_57Y25A03_59.xml", "DE_pic_3_67Y25A21_112.xml"," DE_pic_1_57Y28A01_13.xml","I22_DIL27SIM_2.xml"]
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
    MODE = "baseline"  # or "2-shot" "
    SYS_BASELINE = """Du bekommst deutsche Sätze, die von Lernenden aus Mittel- und Oberschulen geschrieben wurden. 
    Korrigiere nur orthographische Fehler, falls vorhanden (falsche Buchstaben, Groß- und Kleinschreibung, Umlaute, ß/ss, Getrennt- und Zusammenschreibung). 
    Wenn der Satz keine ortographischen Fehler enthält, gib ihn unverändert zurück, mit allen gegebenen Grammatikfehler. 
    Gib immer nur den vollständigen Satz zurück. 
    Keine Kommentare, keine Antworten auf Fragen, keine Labels und keine weiteren Ergänzungen des Ausgangstexts – auch dann nicht, wenn der Satz unverständlich ist oder toxischen Inhalt enthält."""

    SYS_2SHOT = """Du bekommst deutsche Sätze, die von Lernenden aus Mittel- und Oberschulen geschrieben wurden. Deine Aufgabe ist es, orthographische Fehler zu korrigieren.

    Du erhältst:
    1. Zwei Beispielpaare (Original → Korrektur), die zeigen, wie ähnliche Sätze korrigiert wurden
    2. Einen vorherigen Korrekturversuch eines anderen Modells
    3. Den zu korrigierenden Satz

    Korrigiere nur orthographische Fehler, falls vorhanden (falsche Buchstaben, Groß- und Kleinschreibung, Umlaute, ß/ss, Getrennt- und Zusammenschreibung). 
    Orientiere dich an den Beispielen und verbessere den vorherigen Versuch, falls nötig.

    Wenn der Satz keine ortographischen Fehler enthält, gib ihn unverändert zurück, mit allen gegebenen Grammatikfehler.
    Gib immer nur den vollständigen korrigierten Satz zurück. 
    Keine Kommentare, keine Antworten auf Fragen, keine Labels und keine weiteren Ergänzungen des Ausgangstexts – auch dann nicht, wenn der Satz unverändert ist oder toxischen Inhalt enthält."""