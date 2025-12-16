import spacy
import pandas as pd
from IPython.display import display

# Load spaCy with sentencizer
def load_spacy(model="de_core_news_sm"):
    try:
        nlp = spacy.load(model, disable=["tagger", "parser", "ner", "lemmatizer"])
    except:
        nlp = spacy.blank("de")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

nlp = load_spacy()
nlp.max_length = 2_000_000

# Process one corpus file (TXT)
def process_corpus_spacy(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if not text:
        return {"n_sentences": 0, "words": 0, "unique_tokens": 0, "avg_words_per_sentence": 0}
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    tokens = [tok.text for tok in doc if tok.is_alpha]
    num_sent = len(sentences)
    num_words = len(tokens)
    unique_tokens = len(set(tokens))
    avg_words_per_sentence = num_words / num_sent if num_sent else 0
    
    return {
        "n_sentences": num_sent,
        "words": num_words,
        "unique_tokens": unique_tokens,
        "avg_words_per_sentence": round(avg_words_per_sentence, 2)
    }

# Process CSV with spaCy row-by-row (much faster than concatenating)
def process_csv_stats_spacy_optimized(df_subset):
    """
    Process CSV data with spaCy row-by-row to avoid memory issues.
    Each sentence is already split in the CSV, so we process individually.
    """
    total_pairs = len(df_subset)
    corrected_pairs = df_subset['corrected'].sum()
    left_as_is = total_pairs - corrected_pairs
    
    # Count sentences (each row = 1 sentence pair = 2 sentences)
    total_sentences = total_pairs * 2
    corrected_sentences = corrected_pairs * 2
    uncorrected_sentences = left_as_is * 2
    
    all_tokens = []
    
    # Process src column row-by-row
    for idx, text in enumerate(df_subset['src'].fillna('')):
        if text and str(text).strip():
            try:
                doc = nlp(str(text))
                tokens = [tok.text for tok in doc if tok.is_alpha]
                all_tokens.extend(tokens)
            except Exception as e:
                # Skip problematic rows
                continue
    
    # Process tgt column row-by-row
    for idx, text in enumerate(df_subset['tgt'].fillna('')):
        if text and str(text).strip():
            try:
                doc = nlp(str(text))
                tokens = [tok.text for tok in doc if tok.is_alpha]
                all_tokens.extend(tokens)
            except Exception as e:
                # Skip problematic rows
                continue
  
    print()  # New line after progress indicators
    
    num_words = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    avg_words_per_sentence = num_words / total_sentences if total_sentences else 0
    
    return {
        "n_sentence_pairs": total_pairs,
        "n_sentences": total_sentences,
        "words": num_words,
        "unique_tokens": unique_tokens,
        "avg_words_per_sentence": round(avg_words_per_sentence, 2),
        "corrected_pairs": int(corrected_pairs),
        "left_as_is": int(left_as_is),
        "corrected_pairs_pct": f"{round(corrected_pairs / total_pairs * 100, 2)}%" if total_pairs else "0%",
        "corrected_sentences": int(corrected_sentences),
        "uncorrected_sentences": int(uncorrected_sentences),
        "corrected_sentences_pct": round(corrected_sentences / total_sentences * 100, 2) if total_sentences else 0
    }

def compute_corpus_stats(csv_path="output/all_corpora.csv"):
    """
    Compute statistics on corpus data.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with statistics
    """

    results = []
    try:
        df_csv = pd.read_csv(csv_path, encoding="utf-8")
        
        # Individual corpora from CSV
        corpus_names = sorted(df_csv['corpus'].unique())
        
        for corpus_name in corpus_names:
            df_subset = df_csv[df_csv['corpus'] == corpus_name]
            stats = process_csv_stats_spacy_optimized(df_subset)
            stats["corpus"] = corpus_name
            results.append(stats)
        
        # Whole CSV corpus
        all_csv_stats = process_csv_stats_spacy_optimized(df_csv)
        all_csv_stats["corpus"] = "WHOLE_CORPUS"
        results.append(all_csv_stats)

    except FileNotFoundError:
        print(f"✗ CSV file not found: {csv_path}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Reorder columns for better readability
    base_cols = ["corpus"]
    
    if "n_sentence_pairs" in df_results.columns:
        other_cols = ["n_sentence_pairs", "n_sentences", "words", "unique_tokens", 
                     "avg_words_per_sentence", "corrected_pairs", "left_as_is", 
                     "corrected_pairs_pct"]
    else:
        other_cols = ["n_sentences", "words", "unique_tokens", "avg_words_per_sentence"]
    
    available_cols = base_cols + [col for col in other_cols if col in df_results.columns]
    df_results = df_results[available_cols]
    
    return df_results

def compute_corrected_only_stats(csv_path="LearnTextNorm-Deoutput/all_corpora.csv"):
    """
    Compute statistics for corrected pairs only.
    
    Returns:
        DataFrame with corrected-only statistics
    """
    try:
        df_csv_full = pd.read_csv(csv_path, encoding="utf-8")
        df_corrected_only = df_csv_full[df_csv_full['corrected'] == True]
        
        if len(df_corrected_only) == 0:
            print("No corrected pairs found in the dataset.")
            return pd.DataFrame()
        
        corrected_stats = []
        corpus_names = sorted(df_corrected_only['corpus'].unique())
        
        # Per-subcorpus stats for corrected pairs only
        for corpus_name in corpus_names:
            df_subset = df_corrected_only[df_corrected_only['corpus'] == corpus_name]
            
            all_tokens = []
            
            # Process src
            for text in df_subset['src'].fillna(''):
                if text and str(text).strip():
                    try:
                        doc = nlp(str(text))
                        tokens = [tok.text for tok in doc if tok.is_alpha]
                        all_tokens.extend(tokens)
                    except:
                        continue
            
            # Process tgt
            for text in df_subset['tgt'].fillna(''):
                if text and str(text).strip():
                    try:
                        doc = nlp(str(text))
                        tokens = [tok.text for tok in doc if tok.is_alpha]
                        all_tokens.extend(tokens)
                    except:
                        continue
            
            num_words = len(all_tokens)
            unique_tokens = len(set(all_tokens))
            total_sentences = len(df_subset) * 2  # Each pair = 2 sentences
            avg_words = num_words / total_sentences if total_sentences else 0
            
            corrected_stats.append({
                'corpus': corpus_name,
                'corrected_pairs': len(df_subset),
                'words': num_words,
                'unique_tokens': unique_tokens,
                'avg_words_per_sentence': round(avg_words, 2)
            })
        
        # Whole corpus corrected pairs
        print(f"  Processing ALL corrected pairs ({len(df_corrected_only):,} rows)...")
        all_tokens_corrected = []
        
        for text in df_corrected_only['src'].fillna(''):
            if text and str(text).strip():
                try:
                    doc = nlp(str(text))
                    tokens = [tok.text for tok in doc if tok.is_alpha]
                    all_tokens_corrected.extend(tokens)
                except:
                    continue
        
        for text in df_corrected_only['tgt'].fillna(''):
            if text and str(text).strip():
                try:
                    doc = nlp(str(text))
                    tokens = [tok.text for tok in doc if tok.is_alpha]
                    all_tokens_corrected.extend(tokens)
                except:
                    continue
        
        num_words_all = len(all_tokens_corrected)
        unique_tokens_all = len(set(all_tokens_corrected))
        total_sentences_all = len(df_corrected_only) * 2
        avg_words_all = num_words_all / total_sentences_all if total_sentences_all else 0
        
        corrected_stats.append({
            'corpus': 'ALL_CORRECTED',
            'corrected_pairs': len(df_corrected_only),
            'words': num_words_all,
            'unique_tokens': unique_tokens_all,
            'avg_words_per_sentence': round(avg_words_all, 2)
        })
        
        return pd.DataFrame(corrected_stats)
        
    except FileNotFoundError:
        print(f"✗ CSV file not found: {csv_path}")
        return pd.DataFrame()

# MAIN EXECUTION 
if __name__ == "__main__":
    import stats_config
    print("\n" + "="*80)
    print(f"CORPUS STATISTICS")
    print("="*80)

    # 1. Main Statistics
    if stats_config.SHOW_MAIN_STATS:
        df_stats = compute_corpus_stats(csv_path="output/all_corpora.csv")
        display(df_stats)


    # 2. Correction Breakdown by Subcorpus
    if stats_config.SHOW_CORRECTION_BREAKDOWN:
        print("\n" + "="*80)
        print("CORRECTION STATISTICS BREAKDOWN")
        print("="*80)
        
        try:
            df_csv_full = pd.read_csv("output/all_corpora.csv", encoding="utf-8")
            
            print("\n--- By Subcorpus ---")
            correction_by_corpus = df_csv_full.groupby('corpus')['corrected'].agg([
                ('total_pairs', 'count'),
                ('corrected_pairs', 'sum'),
                ('left_as_is', lambda x: (~x).sum()),
                ('corrected_pct', lambda x: f"{round(x.sum() / len(x) * 100, 2)}%")
            ]).reset_index()
            
            display(correction_by_corpus)
            
        except FileNotFoundError:
            print("✗ CSV file not found for correction analysis")
    
    # 3. Overall Correction Summary
    if stats_config.SHOW_CORRECTION_SUMMARY:
        try:
            if 'df_csv_full' not in locals():
                df_csv_full = pd.read_csv("output/all_corpora.csv", encoding="utf-8")
            
            print("\n--- Whole Corpus ---")
            total_pairs = len(df_csv_full)
            corrected_pairs = df_csv_full['corrected'].sum()
            left_as_is = total_pairs - corrected_pairs
            
            overall_stats = pd.DataFrame([{
                'Metric': 'Total Sentence Pairs',
                'Count': total_pairs,
                'Percentage': '100.00%'
            }, {
                'Metric': 'Corrected Pairs (True)',
                'Count': int(corrected_pairs),
                'Percentage': f"{corrected_pairs/total_pairs*100:.2f}%"
            }, {
                'Metric': 'Left-As-Is Pairs (False)',
                'Count': int(left_as_is),
                'Percentage': f"{left_as_is/total_pairs*100:.2f}%"
            }])
            
            display(overall_stats)
            
        except FileNotFoundError:
            print("✗ CSV file not found for correction analysis")
        
    # 4. Corrected Pairs Only - Detailed Stats
    if stats_config.SHOW_CORRECTED_ONLY_STATS:
        print("\n" + "="*80)
        print("CORRECTED PAIRS ONLY - DETAILED STATISTICS")
        print("="*80)
        
        df_corrected_stats = compute_corrected_only_stats(csv_path="output/all_corpora.csv")
        if not df_corrected_stats.empty:
            display(df_corrected_stats)

    # 5. Text Type Breakdown
    if stats_config.SHOW_STATS_PER_TEXT_TYPE:
        print("\n" + "="*80)
        print("TEXT TYPE BREAKDOWN")
        print("="*80)
        
        try:
            if 'df_csv_full' not in locals():
                df_csv_full = pd.read_csv("output/all_corpora.csv", encoding="utf-8")
            
            # Sentence-level breakdown
            print("\n--- Sentence-Level Statistics ---")
            sentence_level = df_csv_full.groupby('text_type').agg({
                'src': 'count',
                'corrected': ['sum', lambda x: f"{x.sum()/len(x)*100:.2f}%"]
            }).reset_index()
            sentence_level.columns = ['text_type', 'total_sentences', 'corrected_sentences', 'corrected_pct']
            display(sentence_level)
            
            # Document-level breakdown
            print("\n--- Document-Level Statistics ---")
            # Get unique xml_file + text_type combinations
            unique_docs = df_csv_full.groupby(['xml_file', 'text_type']).size().reset_index(name='sentences_in_doc')
            doc_level = unique_docs.groupby('text_type').agg({
                'xml_file': 'count',
                'sentences_in_doc': ['sum', 'mean']
            }).reset_index()
            doc_level.columns = ['text_type', 'document_count', 'total_sentences', 'avg_sentences_per_doc']
            doc_level['avg_sentences_per_doc'] = doc_level['avg_sentences_per_doc'].round(2)
            display(doc_level)
            
            # Combined breakdown by corpus and text type
            print("\n--- By Corpus and Text Type ---")
            corpus_text_breakdown = df_csv_full.groupby(['corpus', 'text_type']).agg({
                'src': 'count',
                'corrected': ['sum', lambda x: f"{x.sum()/len(x)*100:.2f}%"]
            }).reset_index()
            corpus_text_breakdown.columns = ['corpus', 'text_type', 'total_sentences', 'corrected_sentences', 'corrected_pct']
            display(corpus_text_breakdown)
            
        except FileNotFoundError:
            print("✗ CSV file not found for text type analysis")
        except KeyError as e:
            print(f"✗ Column not found: {e}. Make sure 'text_type' column exists in CSV.")