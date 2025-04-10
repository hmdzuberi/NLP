import os
import numpy as np
import math
from collections import Counter

# --- Import necessary functions ---

try:
    # From the main Naive Bayes implementation file
    from pa3 import (
        create_vocabulary,       # To get vocab map
        tokenize_and_process,    # To process docs for IDF and TF-IDF
        PorterStemmer,           # For stemming consistency
        train_naive_bayes,       # To train the NB model
        predict_naive_bayes,     # To make predictions
        calculate_metrics        # To evaluate results
    )
except ImportError:
    print("ERROR: Failed to import functions from pa3.py. Make sure it's accessible.")
    exit()

def compute_tf(doc_tokens):
    """
    Computes Term Frequency (TF) for a single document.
    TF(t, d) = count(t, d) / total_tokens_in_d

    Args:
        doc_tokens (list): A list of tokens in the document.

    Returns:
        dict: A dictionary mapping each token to its TF score.
    """
    total_tokens = len(doc_tokens)
    if total_tokens == 0:
        return {}
    token_counts = Counter(doc_tokens)
    tf_map = {token: count / total_tokens for token, count in token_counts.items()}
    return tf_map

def compute_idf(corpus_docs_tokens):
    """
    Computes Inverse Document Frequency (IDF) for a corpus.
    IDF(t, D) = log(N / df(t)) where N is total docs, df(t) is doc frequency of term t.

    Args:
        corpus_docs_tokens (list): A list where each element is a list of tokens
                                   representing a document in the training corpus.

    Returns:
        dict: A dictionary mapping each token (term) to its IDF score.
    """
    N = len(corpus_docs_tokens)
    if N == 0:
        return {}

    # Calculate document frequency (df) for each term
    df_map = Counter()
    for doc_tokens in corpus_docs_tokens:
        # Use set to count each term only once per document
        unique_tokens_in_doc = set(doc_tokens)
        df_map.update(unique_tokens_in_doc)

    idf_map = {}
    for token, df in df_map.items():
        # Standard IDF formula: log(N / df). Add 1 to N and df to avoid log(0) or division by zero issues slightly differently.
        # A common variant is log(N / (1 + df)) to handle terms present in all docs smoothly.
        # Another is log((1 + N) / (1 + df)) + 1 (used by scikit-learn default)
        # Let's use log(N / df) and assume df > 0 for terms in vocab.
        # If df is 0, the term isn't in the corpus, so it won't be in df_map.
        # If df = N, log(N/N) = log(1) = 0, which is correct.
        if df > 0:
             idf_map[token] = math.log(N / df)
        # else: Handle terms not seen? They won't be in df_map anyway.

    return idf_map

def create_tfidf_vectors(data_dir, vocab_map, idf_map, use_stemming):
    """
    Processes documents in a directory, calculates their TF-IDF vectors based
    on a pre-computed IDF map and vocabulary map.

    Args:
        data_dir (str): Path to the directory containing 'positive' and 'negative' subdirs.
        vocab_map (dict): Map from vocabulary word to index.
        idf_map (dict): Map from vocabulary word to its IDF score (computed from training data).
        use_stemming (bool): Whether to apply stemming to document tokens.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: TF-IDF feature vectors (documents x vocabulary size).
            - numpy.ndarray: Labels (1 for positive, 0 for negative).
    """
    vocab_size = len(vocab_map)
    feature_vectors = []
    labels = []
    stemmer = PorterStemmer() if use_stemming else None
    stem_tag = ", stemmed" if use_stemming else ""

    print(f"\nCreating TF-IDF vectors ({'stemmed' if use_stemming else 'no stemming'}) from: {data_dir}")

    if not os.path.isdir(data_dir):
        print(f"Error: Base directory not found at {data_dir}")
        return np.array([]), np.array([])

    # Pre-fetch IDF values for terms in our specific vocab_map
    # Use IDF=0 for terms in vocab_map but not in idf_map (shouldn't happen if idf_map derived from same vocab source)
    # Or handle OOV terms from test set later?
    # For now, assume idf_map covers vocab_map keys.

    for label_int, sub_dir in enumerate(['negative', 'positive']):
        current_dir = os.path.join(data_dir, sub_dir)
        if not os.path.isdir(current_dir):
            print(f"Warning: Subdirectory not found at {current_dir} during TF-IDF vectorization.")
            continue

        for filename in os.listdir(current_dir):
            filepath = os.path.join(current_dir, filename)
            if os.path.isfile(filepath):
                doc_vector = np.zeros(vocab_size, dtype=float) # TF-IDF uses floats
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Process document text (tokenize, stem/process)
                        processed_tokens = tokenize_and_process(text, use_stemming, stemmer)

                        # Calculate TF for the current document
                        tf_scores = compute_tf(processed_tokens)

                        # Calculate TF-IDF score for each term in the vocabulary
                        for token, tf in tf_scores.items():
                            if token in vocab_map:
                                index = vocab_map[token]
                                # Get IDF score (use 0 if somehow not in idf_map, though it should be)
                                idf = idf_map.get(token, 0)
                                doc_vector[index] = tf * idf

                    feature_vectors.append(doc_vector)
                    labels.append(label_int)
                except Exception as e:
                    print(f"Error processing file {filename} in {sub_dir} for TF-IDF vectorization: {e}")

    return np.array(feature_vectors), np.array(labels)

# --- Helper function to get processed docs --- #

def get_processed_docs(data_dir, use_stemming):
    """
    Reads documents from subdirectories, processes them, and returns a list of token lists.
    Needed for IDF calculation because the imported create_vocabulary doesn't return it.
    """
    all_docs_tokens = []
    stemmer = PorterStemmer() if use_stemming else None
    print(f"Re-processing documents in {data_dir} for IDF calculation...")

    if not os.path.isdir(data_dir):
        print(f"Error: Base directory not found at {data_dir}")
        return []

    for sub_dir in ['positive', 'negative']:
        current_dir = os.path.join(data_dir, sub_dir)
        if not os.path.isdir(current_dir):
            continue # Skip if subdir doesn't exist

        for filename in os.listdir(current_dir):
            filepath = os.path.join(current_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        processed_tokens = tokenize_and_process(text, use_stemming, stemmer)
                        all_docs_tokens.append(processed_tokens)
                except Exception as e:
                    print(f"Error re-processing file {filename} in {sub_dir}: {e}")
    return all_docs_tokens

# --- Main Execution Logic --- #

if __name__ == '__main__':
    train_data_dir = 'tweet/train'
    test_data_dir = 'tweet/test'
    results_file = 'results_tfidf.log' # Separate results file

    results_log_content = [] # Store lines for the log file

    print("===== Naive Bayes with TF-IDF Features =====")
    results_log_content.append("===== Naive Bayes with TF-IDF Features =====")

    # --- Process combinations (Stemming only) ---
    for use_stem in [False, True]:
        stem_label = "Stemmed" if use_stem else "No-Stemming"
        print(f"\n===== PROCESSING TF-IDF: {stem_label} =====")
        results_log_content.append(f"\n===== Model: {stem_label} + TF-IDF ====")

        # 1. Create Vocabulary Map from Training Data
        print(f"--- Step 1: Creating Vocabulary Map ({stem_label}) ---")
        # Use the imported create_vocabulary (ignore the returned vocab set for now)
        _, vocab_map = create_vocabulary(train_data_dir, use_stemming=use_stem)
        if not vocab_map:
            print(f"Failed to create vocabulary map for {stem_label}. Skipping.")
            results_log_content.append("ERROR: Failed to create vocabulary map.\n")
            continue
        vocab_size = len(vocab_map)
        print(f"Vocabulary Size: {vocab_size}")

        # 2. Get Processed Training Docs for IDF
        print(f"--- Step 2: Processing Training Docs for IDF ({stem_label}) ---")
        train_docs_tokens = get_processed_docs(train_data_dir, use_stemming=use_stem)
        if not train_docs_tokens:
            print(f"Failed to process training docs for IDF ({stem_label}). Skipping.")
            results_log_content.append("ERROR: Failed to process training docs for IDF.\n")
            continue
        print(f"Processed {len(train_docs_tokens)} training documents.")

        # 3. Compute IDF from Training Docs
        print(f"--- Step 3: Computing IDF ({stem_label}) ---")
        idf_map = compute_idf(train_docs_tokens)
        print(f"IDF map calculated for {len(idf_map)} terms.")

        # 4. Create TF-IDF Vectors for Training Data
        print(f"--- Step 4: Vectorizing Training Data (TF-IDF, {stem_label}) ---")
        features_train_tfidf, labels_train = create_tfidf_vectors(
            train_data_dir, vocab_map, idf_map, use_stemming=use_stem
        )
        print(f"Train TF-IDF features shape: {features_train_tfidf.shape}")
        print(f"Train labels shape: {labels_train.shape}")

        if features_train_tfidf.size == 0:
            print(f"Failed to vectorize training data for {stem_label}. Skipping.")
            results_log_content.append(f"ERROR: Failed to vectorize training data (TF-IDF, {stem_label}).\n")
            continue

        # 5. Train Naive Bayes (using TF-IDF features)
        # Note: Standard NB expects counts/binary, using TF-IDF might be suboptimal.
        print(f"--- Step 5: Training Classifier (TF-IDF, {stem_label}) ---")
        log_priors_tfidf, log_likelihoods_tfidf = train_naive_bayes(features_train_tfidf, labels_train)
        print("Training complete.")

        # 6. Create TF-IDF Vectors for Test Data
        print(f"--- Step 6: Vectorizing Test Data (TF-IDF, {stem_label}) ---")
        features_test_tfidf, labels_test = create_tfidf_vectors(
            test_data_dir, vocab_map, idf_map, use_stemming=use_stem
        )
        print(f"Test TF-IDF features shape: {features_test_tfidf.shape}")
        print(f"Test labels shape: {labels_test.shape}")

        if features_test_tfidf.size == 0:
            print(f"Failed to vectorize test data for {stem_label}. Skipping.")
            results_log_content.append(f"ERROR: Failed to vectorize test data (TF-IDF, {stem_label}).\n")
            continue

        # 7. Predict on Test Data
        print(f"--- Step 7: Predicting on Test Data (TF-IDF, {stem_label}) ---")
        predictions_tfidf = predict_naive_bayes(features_test_tfidf, log_priors_tfidf, log_likelihoods_tfidf)
        print("Prediction complete.")

        # 8. Evaluate
        print(f"--- Step 8: Evaluating (TF-IDF, {stem_label}) ---")
        accuracy_tfidf, conf_matrix_tfidf = calculate_metrics(labels_test, predictions_tfidf)
        print(f"Accuracy: {accuracy_tfidf:.4f}")
        print(f"Confusion Matrix: {conf_matrix_tfidf}")

        # Append results to log content
        results_log_content.append(f"Accuracy: {accuracy_tfidf:.4f}")
        results_log_content.append("Confusion Matrix:")
        results_log_content.append(f"  Predicted:")
        results_log_content.append(f"         Neg(0) Pos(1)")
        results_log_content.append(f"Actual Neg(0) [ {conf_matrix_tfidf['TN']:^5} | {conf_matrix_tfidf['FP']:^5} ]")
        results_log_content.append(f"Actual Pos(1) [ {conf_matrix_tfidf['FN']:^5} | {conf_matrix_tfidf['TP']:^5} ]")
        results_log_content.append("") # Add a blank line


    # 9. Save results to file
    print(f"\n--- Step 9: Saving TF-IDF Results to {results_file} ---")
    try:
        with open(results_file, 'w') as f:
            f.write("\n".join(results_log_content))
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to {results_file}: {e}")

    print("\n===== TF-IDF Processing Complete =====") 