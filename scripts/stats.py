import spacy
import pandas as pd
from configs import Paths, StatsDisplay
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

# Process TSV with spaCy row-by-row (much faster than concatenating)
def process_tsv(df_subset):
    """
    Process TSV data with spaCy row-by-row to avoid memory issues.
    Each sentence is already split in the TSV, so we process individually.
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

def compute_stats(tsv_path=Paths.EXTRACT_TSV):
    """
    Compute statistics on corpus data.
    
    Args:
        tsv_path: Path to TSV file
    
    Returns:
        DataFrame with statistics
    """

    results = []
    try:
        df_tsv = pd.read_csv(tsv_path, encoding="utf-8", sep="\t")
        
        # Individual corpora from TSV
        corpus_names = sorted(df_tsv['corpus'].unique())
        
        for corpus_name in corpus_names:
            df_subset = df_tsv[df_tsv['corpus'] == corpus_name]
            stats = process_tsv(df_subset)
            stats["corpus"] = corpus_name
            results.append(stats)
        
        # Whole TSV corpus
        all_csv_stats = process_tsv(df_tsv)
        all_csv_stats["corpus"] = "WHOLE_CORPUS"
        results.append(all_csv_stats)

    except FileNotFoundError:
        print(f"✗ TSV file not found: {tsv_path}")
    
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

def correction_stats_only(tsv_path=Paths.EXTRACT_TSV):
    """
    Compute statistics for corrected pairs only.
    
    Returns:
        DataFrame with corrected-only statistics
    """
    try:
        df_tsv_full = pd.read_csv(tsv_path, encoding="utf-8",sep="\t")
        df_corrected_only = df_tsv_full[df_tsv_full['corrected'] == True]
        
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
        print(f"✗ TSV file not found: {tsv_path}")
        return pd.DataFrame()

# MAIN EXECUTION 
if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"CORPUS STATISTICS")
    print("="*80)

    # 1. Main Statistics
    if StatsDisplay.MAIN_STATS:
        print("\n" + "="*80)
        print(f"GENERAL OVERVIEW")
        print("="*80)
        df_stats = compute_stats(tsv_path=Paths.EXTRACT_TSV)
        display(df_stats)

    # 2. Sentence Count by Subcorpus
    if StatsDisplay.SUBCORPUS_STATS:
        print("\n" + "="*80)
        print("SENTENCE COUNT BY SUBCORPUS")
        print("="*80)

        try:
            if 'df_tsv_full' not in locals():
                df_tsv_full = pd.read_csv(Paths.EXTRACT_TSV, encoding="utf-8", sep="\t")
            
            total_sentences = len(df_tsv_full)
            
            sentence_count_by_corpus = df_tsv_full.groupby('corpus').size().reset_index(name='sentence_count')
            sentence_count_by_corpus['percentage'] = (sentence_count_by_corpus['sentence_count'] / total_sentences * 100).round(2).astype(str) + '%'
            
            # Add total row
            total_row = pd.DataFrame([{
                'corpus': 'WHOLE_CORPUS',
                'sentence_count': total_sentences,
                'percentage': '100.00%'
            }])
            sentence_count_by_corpus = pd.concat([sentence_count_by_corpus, total_row], ignore_index=True)
            
            display(sentence_count_by_corpus)
            
        except FileNotFoundError:
            print("✗ TSV file not found for sentence count analysis")


    # 3. Correction Breakdown by Subcorpus
    if StatsDisplay.CORRECTION_BREAKDOWN:
        print("\n" + "="*80)
        print("CORRECTION STATISTICS BREAKDOWN")
        print("="*80)
        
        try:
            df_tsv_full = pd.read_csv(Paths.EXTRACT_TSV, encoding="utf-8", sep="\t")
            
            print("\n--- By Subcorpus ---")
            correction_by_corpus = df_tsv_full.groupby('corpus')['corrected'].agg([
                ('total_pairs', 'count'),
                ('corrected_pairs', 'sum'),
                ('left_as_is', lambda x: (~x).sum()),
                ('corrected_pct', lambda x: f"{round(x.sum() / len(x) * 100, 2)}%")
            ]).reset_index()
            
            display(correction_by_corpus)
            
        except FileNotFoundError:
            print("✗ TSV file not found for correction analysis")
    
    # 4. Overall Correction Summary
    if StatsDisplay.CORRECTION_SUMMARY:
        try:
            if 'df_tsv_full' not in locals():
                df_tsv_full = pd.read_csv(Paths.EXTRACT_TSV, encoding="utf-8", sep="\t")
            
            print("\n--- Whole Corpus ---")
            total_pairs = len(df_tsv_full)
            corrected_pairs = df_tsv_full['corrected'].sum()
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
            print("✗ TSV file not found for correction analysis")
        
    # 5. Corrected Pairs Only - Detailed Stats
    if StatsDisplay.CORRECTED_ONLY_STATS:
        print("\n" + "="*80)
        print("CORRECTED PAIRS ONLY - DETAILED STATISTICS")
        print("="*80)
        
        df_corrected_stats = correction_stats_only(tsv_path=Paths.EXTRACT_TSV)
        if not df_corrected_stats.empty:
            display(df_corrected_stats)

    # 5. Text Type Breakdown
    if StatsDisplay.STATS_PER_TEXT_TYPE:
        print("\n" + "="*80)
        print("TEXT TYPE BREAKDOWN")
        print("="*80)
        
        try:
            if 'df_tsv_full' not in locals():
                df_tsv_full = pd.read_csv(Paths.EXTRACT_TSV, encoding="utf-8", sep="\t")
            
            total_sentences_overall = len(df_tsv_full)
            
            # 5A. Sentence-level breakdown
            if StatsDisplay.TEXT_TYPE_SENTENCE_LEV:
                print("\n--- Sentence-Level Statistics ---")
                sentence_level = df_tsv_full.groupby('text_type').size().reset_index(name='sentence_count')
                sentence_level['percentage'] = (sentence_level['sentence_count'] / total_sentences_overall * 100).round(2).astype(str) + '%'
                
                # Add total row
                total_row = pd.DataFrame([{
                    'text_type': 'TOTAL',
                    'sentence_count': total_sentences_overall,
                    'percentage': '100.00%'
                }])
                sentence_level = pd.concat([sentence_level, total_row], ignore_index=True)
                display(sentence_level)
                

            # 5B. Document-level breakdown
            if StatsDisplay.TEXT_TYPE_DOCUMENT_LEV:
                print("\n--- Document-Level Statistics ---")
                # Get unique xml_file + text_type combinations
                unique_docs = df_tsv_full.groupby(['xml_file', 'text_type']).size().reset_index(name='sentences_in_doc')
                total_docs = len(unique_docs)
                
                doc_level = unique_docs.groupby('text_type').agg({
                    'xml_file': 'count',
                    'sentences_in_doc': ['sum', 'mean']
                }).reset_index()
                doc_level.columns = ['text_type', 'document_count', 'total_sentences', 'avg_sentences_per_doc']
                doc_level['percentage'] = (doc_level['document_count'] / total_docs * 100).round(2).astype(str) + '%'
                doc_level['avg_sentences_per_doc'] = doc_level['avg_sentences_per_doc'].round(2)
                
                # Add total row
                total_doc_row = pd.DataFrame([{
                    'text_type': 'TOTAL',
                    'document_count': total_docs,
                    'total_sentences': unique_docs['sentences_in_doc'].sum(),
                    'avg_sentences_per_doc': (unique_docs['sentences_in_doc'].sum() / total_docs).round(2),
                    'percentage': '100.00%'
                }])
                doc_level = pd.concat([doc_level, total_doc_row], ignore_index=True)
                display(doc_level)
            
            # 5C. Combined breakdown by corpus and text type
            if StatsDisplay.TEXT_TYPE_COMBINED:
                print("\n--- By Corpus and Text Type ---")
                print(df_tsv_full['text_type'].value_counts())
                corpus_text_breakdown = df_tsv_full.groupby(['corpus', 'text_type']).size().reset_index(name='sentence_count')
                
                # Calculate percentages within each corpus
                corpus_totals = df_tsv_full.groupby('corpus').size().reset_index(name='corpus_total')
                corpus_text_breakdown = corpus_text_breakdown.merge(corpus_totals, on='corpus')
                corpus_text_breakdown['percentage'] = (corpus_text_breakdown['sentence_count'] / corpus_text_breakdown['corpus_total'] * 100).round(2).astype(str) + '%'
                corpus_text_breakdown = corpus_text_breakdown[['corpus', 'text_type', 'sentence_count', 'percentage']]
                
                # Add WHOLE_CORPUS totals
                whole_corpus_breakdown = df_tsv_full.groupby('text_type').size().reset_index(name='sentence_count')
                whole_corpus_breakdown['corpus'] = 'WHOLE_CORPUS'
                whole_corpus_breakdown['percentage'] = (whole_corpus_breakdown['sentence_count'] / total_sentences_overall * 100).round(2).astype(str) + '%'
                whole_corpus_breakdown = whole_corpus_breakdown[['corpus', 'text_type', 'sentence_count', 'percentage']]
                
                corpus_text_breakdown = pd.concat([corpus_text_breakdown, whole_corpus_breakdown], ignore_index=True)
                display(corpus_text_breakdown)
            
        except FileNotFoundError:
            print("✗ TSV file not found for text type analysis")
        except KeyError as e:
            print(f"✗ Column not found: {e}. Make sure 'text_type' column exists in TSV.")