# AIT 526 Programming Assignment 3

Team 12:

- Aakiff Panjwani

- Vansh Setpal

- Hamaad Zuberi



## Problem

In this assignment, we create a program to automatically identify if airline reviews express positive or negative sentiments by building a naïve Bayes classifier.



## Inputs and Examples

See: `results.log` and `results_tfidf.log`

Run the following code to execute the script.

```shell
python pa3.py
```



## Description

This script implements a Naive Bayes classifier for sentiment analysis on text data (presumably tweets, given the directory names). It tests four different configurations by combining two text preprocessing options (stemming vs. no stemming) and two feature representation methods (binary vs. frequency).

1. **Initialization:**

    - Define paths for training (`tweet/train`) and testing (`tweet/test`) data, and an output file (`results.log`).

    - Initialize an empty list (`results_log_content`) to store results for the log file.

2. **Iterate through Configurations:** The script loops through two main settings:

    - **Stemming:** Runs the entire process once with stemming disabled (`use_stem=False`) and once with stemming enabled (`use_stem=True`). Stemming reduces words to their root form (e.g., "running" -> "run").

    - **Feature Representation:** Within each stemming loop, it further loops through two types:

        - `frequency`: Represents documents based on the count of each word.

        - `binary`: Represents documents based on the presence or absence (1 or 0) of each word.

3. **Vocabulary Creation (**`create_vocabulary`**):**

    - For the current stemming setting, this function reads all files in the `positive` and `negative` subdirectories of the *training* data (`tweet/train`).

    - **Text Cleaning (**`clean_text`**):** Each file's content is cleaned by removing HTML tags, converting emojis to text descriptions, removing URLs, and normalizing whitespace.

    - **Tokenization and Processing (**`tokenize_and_process`**,** `process_token`**):** The cleaned text is tokenized (split into words/punctuation). Each token is then processed: punctuation is kept as is, words are potentially stemmed (if `use_stem` is `True`) and lowercased (conditionally, to avoid lowercasing all-caps words).

    - A set of all unique processed tokens from all training documents is created. This forms the `vocabulary`.

    - A sorted list of the vocabulary and a `vocab_map` (dictionary mapping each token to a unique integer index) are created.

4. **Data Vectorization (**`vectorize_data`**):**

    - This function converts the text documents (both training and testing sets, processed separately) into numerical feature vectors based on the `vocab_map` and the chosen `representation_type` (`binary` or `frequency`).

    - It iterates through the `positive` and `negative` subdirectories of the specified data path (either train or test).

    - Each file is cleaned and tokenized using the same process as in vocabulary creation (including the current `use_stem` setting).

    - A vector (initialized to zeros) is created for each document with a length equal to the vocabulary size.

    - **Binary Representation:** If a token from the vocabulary exists in the document, the corresponding index in the vector is set to 1.

    - **Frequency Representation:** The value at each index in the vector corresponds to the number of times the respective vocabulary token appears in the document.

    - The function returns two NumPy arrays: `feature_vectors` (one row per document) and `labels` (0 for negative, 1 for positive).

5. **Naive Bayes Training (**`train_naive_bayes`**):**

    - This function takes the *training* feature vectors and labels.

    - It calculates the log prior probability of each class (positive/negative) based on the proportion of training documents belonging to each class: ( $log P(c)$ ).

    - It calculates the log likelihood of each word in the vocabulary given each class: ( $log P(word|c)$ ). This involves counting word occurrences within each class and applying Laplace (add 1) smoothing to handle words that might not appear in documents of a specific class.

    - Returns the calculated `log_priors` and `log_likelihoods`.

6. **Naive Bayes Prediction (**`predict_naive_bayes`**):**

    - This function takes the *testing* feature vectors and the `log_priors` and `log_likelihoods` learned during training.

    - For each test document vector:

        - It calculates a score for each class (positive and negative) by summing the log prior of the class and the log likelihoods of the words present in the document (multiplied by their frequency if using frequency representation).

        $$
        score(c) = \log P(c) + \sum_{word \in doc} \text{count}(word) \times \log P(word|c)
        $$

        - The class with the higher score is assigned as the prediction for that document.

    - Returns an array of predictions for the test set.

7. **Evaluation (**`calculate_metrics`**):**

    - Takes the true labels from the test set and the predictions made in the previous step.

    - Calculates:

        - Confusion Matrix components ($TN, FP, FN, TP$).

        - Accuracy: ( $\frac{\text{TP} + \text{TN}}{\text{Total}}$ ).

        - Precision: ( $\frac{\text{TP}}{\text{TP} + \text{FP}}$ ).

        - Recall: ( $\frac{\text{TP}}{\text{TP} + \text{FN}}$ ).

        - F1 Score: ( $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ ).

    - Returns these metrics.

8. **Reporting:**

    - The calculated metrics for the current configuration (stemming + representation type) are printed to the console.

    - Formatted results are appended to the `results_log_content` list.

9. **Output:**

    - After all four configurations are processed, the collected results in `results_log_content` are written to the `results.log` file.

## Bonus

See: `results_tfidf.log` for results

While TF-IDF is often effective for text classification, it’s usually not the best fit for Naive Bayes classifiers. That’s because Naive Bayes is designed around binary or frequency-based data rather than weighted values.
Binary or frequency representations align naturally with Naive Bayes’ probabilistic assumptions, so they typically yield better performance. Switching to TF-IDF with Naive Bayes doesn't improve accuracy (as seen in `results_tfidf.log` when compared to `results.log`) because TF-IDF doesn’t match well with the underlying assumptions of the algorithm. To effectively leverage TF-IDF, you’d generally choose a different type of classifier instead.
