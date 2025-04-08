import os
import re
import string
import numpy as np
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
import emoji
from bs4 import BeautifulSoup

# Download necessary NLTK data (run this once)
# import nltk
# nltk.download('punkt')

def clean_text(text):
    # Parse and remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Convert emoji characters to their text descriptions
    text = emoji.demojize(text)
    
    # Remove any URLs from the text using regex
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace by replacing multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_token(token, use_stemming, stemmer):
    # Keep punctuation as is
    if token in string.punctuation:
        return token
    
    if use_stemming:
        # Apply stemming and convert to lowercase
        return stemmer.stem(token.lower())
    else:
        # Only lowercase words that start with capital letter but aren't all caps
        if token and token[0].isupper() and not token.isupper():
            return token.lower()
        return token

def tokenize_and_process(text, use_stemming, stemmer):
    # Clean the text first
    cleaned_text = clean_text(text)
    # Split into tokens
    tokens = word_tokenize(cleaned_text)
    # Process each token
    processed_tokens = [process_token(t, use_stemming, stemmer) for t in tokens]
    return processed_tokens

def create_vocabulary(data_dir, use_stemming):
    # Initialize sets to store unique words from each class
    positive_words = set()
    negative_words = set()
    # Initialize stemmer if needed
    stemmer = SnowballStemmer('english') if use_stemming else None
    stem_tag = " (stemmed)" if use_stemming else " (no stemming)"

    print(f"Creating vocabulary{stem_tag} from subdirectories of: {data_dir}")
    if not os.path.isdir(data_dir):
        print(f"Error: Base directory not found at {data_dir}")
        return set(), {}

    # Process positive documents
    pos_dir = os.path.join(data_dir, 'positive')
    if os.path.isdir(pos_dir):
        for filename in os.listdir(pos_dir):
            filepath = os.path.join(pos_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        processed_tokens = tokenize_and_process(text, use_stemming, stemmer)
                        positive_words.update(processed_tokens)
                except Exception as e:
                    print(f"Error processing file {filename} in positive: {e}")

    # Process negative documents
    neg_dir = os.path.join(data_dir, 'negative')
    if os.path.isdir(neg_dir):
        for filename in os.listdir(neg_dir):
            filepath = os.path.join(neg_dir, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        processed_tokens = tokenize_and_process(text, use_stemming, stemmer)
                        negative_words.update(processed_tokens)
                except Exception as e:
                    print(f"Error processing file {filename} in negative: {e}")

    # Combine positive and negative words into vocabulary
    vocabulary = positive_words.union(negative_words)
    # Create sorted list and mapping for efficient lookup
    vocab_list = sorted(list(vocabulary))
    vocab_map = {word: i for i, word in enumerate(vocab_list)}
    
    # Create dictionaries to track which words belong to which class
    class_words = {
        1: {word: 1 for word in positive_words},  # Positive class
        0: {word: 0 for word in negative_words}   # Negative class
    }
    
    print(f"Vocabulary size{stem_tag}: {len(vocab_list)}")
    return vocabulary, vocab_map, class_words

def vectorize_data(data_dir, vocab_map, representation_type, use_stemming):
    vocab_size = len(vocab_map)
    feature_vectors = []
    labels = []
    stemmer = SnowballStemmer('english') if use_stemming else None
    stem_tag = ", stemmed" if use_stemming else ""

    print(f"\nVectorizing data ({representation_type}{stem_tag}) from: {data_dir}")

    if not os.path.isdir(data_dir):
        print(f"Error: Base directory not found at {data_dir}")
        return np.array([]), np.array([])

    # Process both positive and negative documents
    for label_int, sub_dir in enumerate(['negative', 'positive']):
        current_dir = os.path.join(data_dir, sub_dir)
        if not os.path.isdir(current_dir):
            print(f"Warning: Subdirectory not found at {current_dir} during vectorization.")
            continue

        for filename in os.listdir(current_dir):
            filepath = os.path.join(current_dir, filename)
            if os.path.isfile(filepath):
                # Initialize document vector based on representation type
                if representation_type == 'binary':
                    doc_vector = np.zeros(vocab_size, dtype=int)
                    unique_tokens = set()
                else:  # frequency
                    doc_vector = np.zeros(vocab_size, dtype=int)
                    token_counts = defaultdict(int)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        processed_tokens = tokenize_and_process(text, use_stemming, stemmer)

                        if representation_type == 'binary':
                            # For binary representation, just mark presence of words
                            for token in processed_tokens:
                                if token in vocab_map:
                                    unique_tokens.add(token)
                            for token in unique_tokens:
                                doc_vector[vocab_map[token]] = 1
                        else:  # frequency
                            # For frequency representation, count word occurrences
                            for token in processed_tokens:
                                if token in vocab_map:
                                    token_counts[token] += 1
                            for token, count in token_counts.items():
                                doc_vector[vocab_map[token]] = count

                    feature_vectors.append(doc_vector)
                    labels.append(label_int)
                except Exception as e:
                    print(f"Error processing file {filename} in {sub_dir} for vectorization: {e}")

    return np.array(feature_vectors), np.array(labels)

def train_naive_bayes(features, labels):
    n_docs, vocab_size = features.shape
    n_classes = 2  # Binary classification: positive (1) and negative (0)

    # Calculate class prior probabilities
    log_priors = np.zeros(n_classes)
    for c in range(n_classes):
        n_docs_in_class = np.sum(labels == c)
        log_priors[c] = np.log(n_docs_in_class / n_docs)

    # Calculate word likelihoods for each class
    log_likelihoods = np.zeros((vocab_size, n_classes))
    for c in range(n_classes):
        docs_in_class = features[labels == c]
        word_counts_in_class = np.sum(docs_in_class, axis=0)
        total_words_in_class = np.sum(word_counts_in_class)
        
        # Apply Laplace smoothing (add-1 smoothing)
        numerator = word_counts_in_class + 1
        denominator = total_words_in_class + vocab_size
        log_likelihoods[:, c] = np.log(numerator) - np.log(denominator)

    return log_priors, log_likelihoods

def predict_naive_bayes(features, log_priors, log_likelihoods):
    # Initialize scores for each class
    scores = np.zeros((features.shape[0], 2))
    
    for c in range(2):
        # Start with log prior probability
        scores[:, c] = log_priors[c]
        
        # Add log likelihood for each word in the document
        for i in range(features.shape[0]):
            word_indices = np.where(features[i] > 0)[0]
            for idx in word_indices:
                scores[i, c] += log_likelihoods[idx, c] * features[i, idx]

    # Predict the class with the highest score
    predictions = np.argmax(scores, axis=1)
    return predictions

def calculate_metrics(true_labels, predictions):
    # Calculate confusion matrix
    TN = np.sum((true_labels == 0) & (predictions == 0))
    FP = np.sum((true_labels == 0) & (predictions == 1))
    FN = np.sum((true_labels == 1) & (predictions == 0))
    TP = np.sum((true_labels == 1) & (predictions == 1))

    confusion_matrix = {'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP}

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Handle division by zero
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, confusion_matrix

if __name__ == '__main__':
    train_data_dir = 'tweet/train'
    test_data_dir = 'tweet/test'
    results_file = 'results2.log'

    results_log_content = [] # Store lines for the log file

    # --- Process combinations ---
    for use_stem in [False, True]:
        stem_label = "Stemmed" if use_stem else "No-Stemming"
        print(f"\n===== PROCESSING: {stem_label} =====")
        results_log_content.append(f"===== {stem_label} =====")

        # 1. Create Vocabulary and Map from Training Data
        print(f"--- Step 1: Creating Vocabulary ({stem_label}) ---")
        vocab, vocab_map, class_words = create_vocabulary(train_data_dir, use_stemming=use_stem)
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

            # 5. Predict on Test Data
            print(f"--- Step 5: Predicting on Test Data ({rep_type}, {stem_label}) ---")
            predictions = predict_naive_bayes(features_test, log_priors, log_likelihoods)
            print("Prediction complete.")

            # 6. Evaluate
            print(f"--- Step 6: Evaluating ({rep_type}, {stem_label}) ---")
            accuracy, precision, recall, f1, conf_matrix = calculate_metrics(labels_test, predictions)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Confusion Matrix: {conf_matrix}")

            # Append results to log content
            results_log_content.append(f"\n--- Model: {stem_label} + {rep_type.capitalize()} ---")
            results_log_content.append(f"Accuracy: {accuracy:.4f}")
            results_log_content.append(f"Precision: {precision:.4f}")
            results_log_content.append(f"Recall: {recall:.4f}")
            results_log_content.append(f"F1 Score: {f1:.4f}")
            results_log_content.append("Confusion Matrix:")
            results_log_content.append(f"  Predicted:")
            results_log_content.append(f"    Negative: {conf_matrix['TN']} (TN), {conf_matrix['FP']} (FP)")
            results_log_content.append(f"    Positive: {conf_matrix['FN']} (FN), {conf_matrix['TP']} (TP)")
            results_log_content.append("")

    # Write results to file
    with open(results_file, 'w') as f:
        f.write('\n'.join(results_log_content))
    print(f"\nResults written to {results_file}")
