import os
import re
import string
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK data (run this once)
# import nltk
# nltk.download('punkt')

def clean_text(text):
    """Removes HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def process_token(token, use_stemming, stemmer):
    """Processes a single token: applies stemming or capitalization rules."""
    if use_stemming:
        # Stemming usually done on lowercased words
        return stemmer.stem(token.lower())
    else:
        # Lowercase if starts with capital but not all caps
        if token and token[0].isupper() and not token.isupper():
            return token.lower()
        return token

def tokenize_and_process(text, use_stemming, stemmer):
    """Cleans text, tokenizes, and processes tokens."""
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    processed_tokens = [process_token(t, use_stemming, stemmer) for t in tokens]
    return processed_tokens

def create_vocabulary(data_dir, use_stemming):
    """
    Reads training data, processes tokens, and creates vocabulary and its map.

    Args:
        data_dir (str): Path to the directory containing 'positive' and 'negative' subdirectories.
        use_stemming (bool): Whether to apply stemming.

    Returns:
        tuple: A tuple containing:
            - set: Vocabulary.
            - dict: Map from word to index in the vocabulary list.
    """
    vocabulary = set()
    stemmer = PorterStemmer() if use_stemming else None
    stem_tag = " (stemmed)" if use_stemming else " (no stemming)"

    print(f"Creating vocabulary{stem_tag} from subdirectories of: {data_dir}")
    if not os.path.isdir(data_dir):
        print(f"Error: Base directory not found at {data_dir}")
        return set(), {}

    for sub_dir in ['positive', 'negative']:
        current_dir = os.path.join(data_dir, sub_dir)
        if not os.path.isdir(current_dir):
            print(f"Warning: Subdirectory not found at {current_dir}")
            continue

        print(f"Processing files in: {current_dir}")
        for filename in os.listdir(current_dir):
            filepath = os.path.join(current_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        processed_tokens = tokenize_and_process(text, use_stemming, stemmer)
                        vocabulary.update(processed_tokens)
                except Exception as e:
                    print(f"Error processing file {filename} in {sub_dir}: {e}")

    vocab_list = sorted(list(vocabulary))
    vocab_map = {word: i for i, word in enumerate(vocab_list)}
    print(f"Vocabulary size{stem_tag}: {len(vocab_list)}")
    return vocabulary, vocab_map

def vectorize_data(data_dir, vocab_map, representation_type, use_stemming):
    """
    Converts documents into Bag-of-Words feature vectors based on a given vocabulary map.

    Args:
        data_dir (str): Path to the directory containing 'positive' and 'negative' subdirectories.
        vocab_map (dict): Map from word to index.
        representation_type (str): 'frequency' or 'binary'.
        use_stemming (bool): Whether to apply stemming to document tokens.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Feature vectors (documents x vocabulary size).
            - numpy.ndarray: Labels (1 for positive, 0 for negative).
    """
    vocab_size = len(vocab_map)
    feature_vectors = []
    labels = []
    stemmer = PorterStemmer() if use_stemming else None
    stem_tag = ", stemmed" if use_stemming else ""

    print(f"\nVectorizing data ({representation_type}{stem_tag}) from: {data_dir}")

    if not os.path.isdir(data_dir):
        print(f"Error: Base directory not found at {data_dir}")
        return np.array([]), np.array([])

    for label_int, sub_dir in enumerate(['negative', 'positive']): # Assign 0 to negative, 1 to positive
        current_dir = os.path.join(data_dir, sub_dir)
        if not os.path.isdir(current_dir):
            print(f"Warning: Subdirectory not found at {current_dir} during vectorization.")
            continue

        # print(f"Processing files in: {current_dir}") # Optional: reduce verbosity
        for filename in os.listdir(current_dir):
            filepath = os.path.join(current_dir, filename)
            if os.path.isfile(filepath):
                doc_vector = np.zeros(vocab_size, dtype=int)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        processed_tokens = tokenize_and_process(text, use_stemming, stemmer)

                        if representation_type == 'binary':
                            for token in processed_tokens:
                                if token in vocab_map:
                                    index = vocab_map[token]
                                    doc_vector[index] = 1
                        elif representation_type == 'frequency':
                            token_counts = Counter(processed_tokens)
                            for token, count in token_counts.items():
                                if token in vocab_map:
                                    index = vocab_map[token]
                                    doc_vector[index] = count

                    feature_vectors.append(doc_vector)
                    labels.append(label_int)
                except Exception as e:
                    print(f"Error processing file {filename} in {sub_dir} for vectorization: {e}")

    return np.array(feature_vectors), np.array(labels)

def train_naive_bayes(features, labels):
    """
    Trains a Multinomial Naive Bayes classifier.

    Args:
        features (numpy.ndarray): Document-term matrix (docs x vocab_size).
                                  Can be frequency counts or binary presence.
        labels (numpy.ndarray): Corresponding labels (0 or 1) for each document.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Log prior probabilities for each class [log P(c=0), log P(c=1)].
            - numpy.ndarray: Log likelihood probabilities for each word given class (vocab_size x 2).
                           Column 0: log P(w | c=0), Column 1: log P(w | c=1)
    """
    n_docs, vocab_size = features.shape
    n_classes = 2 # We have positive (1) and negative (0)

    log_priors = np.zeros(n_classes)
    log_likelihoods = np.zeros((vocab_size, n_classes))

    for c in range(n_classes): # 0 for negative, 1 for positive
        docs_in_class = features[labels == c]
        n_docs_in_class = docs_in_class.shape[0]
        log_priors[c] = np.log(n_docs_in_class / n_docs)

        word_counts_in_class = np.sum(docs_in_class, axis=0)
        total_words_in_class = np.sum(word_counts_in_class)

        numerator = word_counts_in_class + 1
        denominator = total_words_in_class + vocab_size
        log_likelihoods[:, c] = np.log(numerator) - np.log(denominator)

    # print(f"Training complete. Log priors: {log_priors}") # Reduce verbosity
    return log_priors, log_likelihoods

def predict_naive_bayes(features, log_priors, log_likelihoods):
    """
    Predicts class labels for features using trained Naive Bayes parameters.

    Args:
        features (numpy.ndarray): Test features (n_test_docs x vocab_size).
        log_priors (numpy.ndarray): Log prior probabilities [log P(c=0), log P(c=1)].
        log_likelihoods (numpy.ndarray): Log likelihoods (vocab_size x 2).

    Returns:
        numpy.ndarray: Predicted labels (0 or 1) for each test document.
    """
    # Calculate score for each class for all documents
    # Score(c) = log P(c) + sum(feature_vector * log P(w|c))
    # Using matrix multiplication: features @ log_likelihoods
    # Result shape: (n_test_docs x 2)
    scores = log_priors + features @ log_likelihoods

    # Predict the class with the higher score (argmax along axis 1)
    predictions = np.argmax(scores, axis=1)
    return predictions

def calculate_metrics(true_labels, predictions):
    """
    Calculates accuracy and confusion matrix.

    Args:
        true_labels (numpy.ndarray): True labels (0 or 1).
        predictions (numpy.ndarray): Predicted labels (0 or 1).

    Returns:
        tuple: A tuple containing:
            - float: Accuracy.
            - dict: Confusion matrix {'TN': int, 'FP': int, 'FN': int, 'TP': int}.
    """
    accuracy = np.mean(true_labels == predictions)

    TN = np.sum((true_labels == 0) & (predictions == 0))
    FP = np.sum((true_labels == 0) & (predictions == 1))
    FN = np.sum((true_labels == 1) & (predictions == 0))
    TP = np.sum((true_labels == 1) & (predictions == 1))

    confusion_matrix = {'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP}
    return accuracy, confusion_matrix

if __name__ == '__main__':
    train_data_dir = 'tweet/train'
    test_data_dir = 'tweet/test'
    results_file = 'results.log'

    results_log_content = [] # Store lines for the log file

    # --- Process combinations ---
    for use_stem in [False, True]:
        stem_label = "Stemmed" if use_stem else "No-Stemming"
        print(f"\n===== PROCESSING: {stem_label} =====")
        results_log_content.append(f"===== {stem_label} =====")

        # 1. Create Vocabulary and Map from Training Data
        print(f"--- Step 1: Creating Vocabulary ({stem_label}) ---")
        vocab, vocab_map = create_vocabulary(train_data_dir, use_stemming=use_stem)
        if not vocab_map:
            print(f"Failed to create vocabulary for {stem_label}. Skipping.")
            results_log_content.append("ERROR: Failed to create vocabulary.\n")
            continue

        for rep_type in ['frequency', 'binary']:
            print(f"\n--- Step 2: Vectorizing Training Data ({rep_type}, {stem_label}) ---")
            features_train, labels_train = vectorize_data(
                train_data_dir, vocab_map, rep_type, use_stemming=use_stem
            )
            print(f"Train features shape: {features_train.shape}")
            print(f"Train labels shape: {labels_train.shape}")

            if features_train.size == 0:
                 print(f"Failed to vectorize training data for {stem_label}, {rep_type}. Skipping.")
                 results_log_content.append(f"ERROR: Failed to vectorize training data ({stem_label}, {rep_type}).\n")
                 continue

            # 3. Train Naive Bayes
            print(f"--- Step 3: Training Classifier ({rep_type}, {stem_label}) ---")
            log_priors, log_likelihoods = train_naive_bayes(features_train, labels_train)
            print("Training complete.")

            # 4. Vectorize Test Data
            print(f"--- Step 4: Vectorizing Test Data ({rep_type}, {stem_label}) ---")
            features_test, labels_test = vectorize_data(
                test_data_dir, vocab_map, rep_type, use_stemming=use_stem
            )
            print(f"Test features shape: {features_test.shape}")
            print(f"Test labels shape: {labels_test.shape}")

            if features_test.size == 0:
                 print(f"Failed to vectorize test data for {stem_label}, {rep_type}. Skipping.")
                 results_log_content.append(f"ERROR: Failed to vectorize test data ({stem_label}, {rep_type}).\n")
                 continue

            # 5. Predict on Test Data
            print(f"--- Step 5: Predicting on Test Data ({rep_type}, {stem_label}) ---")
            predictions = predict_naive_bayes(features_test, log_priors, log_likelihoods)
            print("Prediction complete.")

            # 6. Evaluate
            print(f"--- Step 6: Evaluating ({rep_type}, {stem_label}) ---")
            accuracy, conf_matrix = calculate_metrics(labels_test, predictions)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Confusion Matrix: {conf_matrix}")

            # Append results to log content
            results_log_content.append(f"\n--- Model: {stem_label} + {rep_type.capitalize()} ---")
            results_log_content.append(f"Accuracy: {accuracy:.4f}")
            results_log_content.append("Confusion Matrix:")
            results_log_content.append(f"  Predicted:")
            results_log_content.append(f"         Neg(0) Pos(1)")
            results_log_content.append(f"Actual Neg(0) [ {conf_matrix['TN']:^5} | {conf_matrix['FP']:^5} ]")
            results_log_content.append(f"Actual Pos(1) [ {conf_matrix['FN']:^5} | {conf_matrix['TP']:^5} ]")
            results_log_content.append("") # Add a blank line for readability


    # 7. Save results to file
    print(f"\n--- Step 7: Saving Results to {results_file} ---")
    try:
        with open(results_file, 'w') as f:
            f.write("\n".join(results_log_content))
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to {results_file}: {e}")

    print("\n===== All Processing Complete =====")
