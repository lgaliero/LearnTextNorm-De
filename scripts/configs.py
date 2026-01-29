class Paths: 
    # Extraction paths
    EXT_LOG_FILE = "../master_files/extraction_debug.log"
    EXTRACT_DIR = '../master_files'  
    EXTRACT_TSV = "../master_files/all_corpora.tsv"
    
    # Dataset splits base directory
    SET_SPLITS = "../data"
    
    # Test split paths
    TEST_SRC = "../data/test/test.src"
    TEST_TGT = "../data/test/test.tgt"
    TEST_IDXS = "../data/test/test_indices.tsv"
    TEST_NORM =  "../data/test/test.norm"
    
    # Dev split paths
    DEV_SRC = "../data/dev/dev.src"
    DEV_TGT = "../data/dev/dev.tgt"
    DEV_IDXS = "../data/dev/dev_indices.tsv"
    DEV_NORM =  "../data/dev/dev.norm"
    
    # Train split paths
    TRAIN_SRC = "../data/train/train.src"
    TRAIN_TGT = "../data/train/train.tgt"
    TRAIN_IDXS = "../data/train/train_indices.tsv"
    TRAIN_NORM =  "../data/train/train.norm"
    
    # Prompt templates
    LLAMA_JSON = "../data/2S_prompts/llama.json"
    GPT_JSON = "../data/2S_prompts/gpt.json"
    GEMMA_JSON = "../data/2S_prompts/gemma.json"
    
    # Model outputs
    MODELS = "../hypos/llama3-2"
    LLAMA_0 = "../hypos/llama3-2/0shot.hyp"
    LLAMA_2 = "../hypos/llama3-2/2shot.hyp"
    GPT_0 = "../hypos/gpt-oss/0shot.hyp"
    GPT_2 = "../hypos/gpt-oss/2shot.hyp"
    GEMMA_0 = "../hypos/gemma/0shot.hyp"
    GEMMA_2 = "../hypos/gemma/2shot.hyp"

    #Evaluation report
    EVAL = "eval_results.txt"

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
    ACTIVE_CORPORA = ['LEONIDE', 'Kolipsi_1_L1', "Kolipsi_1_L2", "Kolipsi_2"]  # Corpora to process (empty list = process none)
    FORMAT = 'both'      # Output format - Options: "tsv", "norm", or "both"
    EXCLUDE = [
        "DE_pic_2_57Y25A14_59.xml",
        "DE_pic_2_57Y25A03_59.xml",
        "DE_pic_3_67Y25A21_112.xml",
        "DE_pic_1_57Y28A01_13.xml",
        "I22_DIL27SIM_2.xml"
    ]  # Files to exclude from processing
    MAX_FILES_PER_CORPUS = None  # Processing limit: None = process all files, or set to integer


# =======================
# DATASET SPLITS
# =======================
class DataSplits:
    """Dataset split proportions (must sum to 1.0)."""
    TEST = 0.10   # 10% for testing
    DEV = 0.10    # 10% for development/validation
    TRAIN = 0.80  # 80% for training


# =======================
# LLM TESTING (via institution API)
# =======================
class ApiConfig:
    """Configuration for LLM API testing."""
    HOST = "http://51.124.247.170:80"
    MODEL = "llama3.2:latest"  # Options: gpt-oss:20b, llama3.2:latest, gemma3:12b
    MODE = "baseline"  # Options: "baseline" or "2-shot"
    
    # System prompt for baseline (0-shot) mode
    SYS_BASELINE = """Du bekommst deutsche Sätze, die von Lernenden aus Mittel- und Oberschulen geschrieben wurden. 
    Korrigiere nur orthographische Fehler, falls vorhanden (falsche Buchstaben, Groß- und Kleinschreibung, Umlaute, ß/ss, Getrennt- und Zusammenschreibung). 
    Wenn der Satz keine ortographischen Fehler enthält, gib ihn unverändert zurück, mit allen gegebenen Grammatikfehler. 
    Gib immer nur den vollständigen Satz zurück. 
    Keine Kommentare, keine Antworten auf Fragen, keine Labels und keine weiteren Ergänzungen des Ausgangstexts – auch dann nicht, wenn der Satz unverständlich ist oder toxischen Inhalt enthält."""

    # System prompt for 2-shot mode
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