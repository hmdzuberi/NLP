# Import required libraries
import nltk
import re
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.util import ngrams

# Download necessary NLTK data (run only once)
nltk.download('punkt')
nltk.download('stopwords')

#################################
# Helper Functions for Preprocessing
#################################
def fetch_text(url):
    """
    Uses requests and BeautifulSoup to extract raw text from a webpage.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def clean_text(text):
    """
    Lowercase conversion, removal of extra spaces and non-alphabetic characters.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_words(text):
    """
    Tokenizes text into words and removes punctuation and stopwords.
    """
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return words

def tokenize_sentences(text):
    """
    Splits the text into sentences.
    """
    sentences = nltk.sent_tokenize(text)
    return sentences

#################################
# Task 1: Summarization using Word Frequencies
#################################

# 1) Web scraping from the Wikipedia page
url = "https://en.wikipedia.org/wiki/Natural_language_processing"
raw_text = fetch_text(url)
cleaned_text = clean_text(raw_text)

# 2) Preprocessing
words = tokenize_words(cleaned_text)
sentences = tokenize_sentences(raw_text)  # use raw_text to preserve punctuation for sentence splitting

# 3) Calculate word frequencies using NLTK's FreqDist
freq_dist = nltk.FreqDist(words)

# 4) Score sentences: sum the frequency of each word present in the sentence
def calculate_sentence_scores(sent_list, freq_dist):
    scores = {}
    for sent in sent_list:
        score = 0
        word_tokens = nltk.word_tokenize(sent.lower())
        for word in word_tokens:
            if word.isalpha() and word in freq_dist:
                score += freq_dist[word]
        scores[sent] = score
    return scores

sentence_scores = calculate_sentence_scores(sentences, freq_dist)
# Rank sentences by score (highest first)
ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

# 5) Build summaries using different criteria
def summarize_by_sentence_count(ranked_sents, count):
    return ' '.join(ranked_sents[:count])

def summarize_by_word_count(ranked_sents, word_limit):
    summary = []
    total_words = 0
    for sent in ranked_sents:
        sent_words = nltk.word_tokenize(sent)
        if total_words + len(sent_words) <= word_limit:
            summary.append(sent)
            total_words += len(sent_words)
        else:
            # If adding the whole sentence exceeds the limit, do not include it.
            break
    return ' '.join(summary)

def summarize_by_percentage(ranked_sents, full_text, percentage):
    total_words = len(nltk.word_tokenize(full_text))
    word_limit = int(total_words * (percentage / 100))
    return summarize_by_word_count(ranked_sents, word_limit)

# Generate three types of summaries from word frequency method
summary_sentence_count = summarize_by_sentence_count(ranked_sentences, 3)
summary_word_count = summarize_by_word_count(ranked_sentences, 50)
summary_percentage = summarize_by_percentage(ranked_sentences, raw_text, 20)

print("Summary (Sentence Count = 3):\n", summary_sentence_count, "\n")
print("Summary (Word Count = 50):\n", summary_word_count, "\n")
print("Summary (20% of Total Words):\n", summary_percentage, "\n")

#################################
# Task 2: Summarization using N-grams
#################################

# 1) Generate n-grams from text using a helper function.
def generate_ngrams(text, n):
    tokens = nltk.word_tokenize(text.lower())
    return [' '.join(gram) for gram in ngrams(tokens, n)]

# 2) Get n-gram frequency distribution (example using tri-grams)
def get_ngram_frequencies(text, n):
    ngram_list = generate_ngrams(text, n)
    return nltk.FreqDist(ngram_list)

# Using n=3 (trigrams) for demonstration. You can test with other n values (e.g., 2, 4)
ngram_freqs = get_ngram_frequencies(cleaned_text, 3)

# 3) Calculate sentence scores based on the sum of frequencies of contained n-grams.
def calculate_sentence_scores_ngram(sent_list, ngram_freqs, n):
    scores = {}
    for sent in sent_list:
        tokens = nltk.word_tokenize(sent.lower())
        sent_ngrams = [' '.join(gram) for gram in ngrams(tokens, n)]
        score = sum(ngram_freqs.get(gram, 0) for gram in sent_ngrams)
        scores[sent] = score
    return scores

sentence_scores_ngram = calculate_sentence_scores_ngram(sentences, ngram_freqs, 3)
ranked_sentences_ngram = sorted(sentence_scores_ngram, key=sentence_scores_ngram.get, reverse=True)

# 4) Generate summary using the same three criteria for n-gram based scores
ngram_summary_sentence_count = summarize_by_sentence_count(ranked_sentences_ngram, 3)
ngram_summary_word_count = summarize_by_word_count(ranked_sentences_ngram, 50)
ngram_summary_percentage = summarize_by_percentage(ranked_sentences_ngram, raw_text, 20)

print("N-gram Summary (Sentence Count = 3):\n", ngram_summary_sentence_count, "\n")
print("N-gram Summary (Word Count = 50):\n", ngram_summary_word_count, "\n")
print("N-gram Summary (20% of Total Words):\n", ngram_summary_percentage, "\n")

#################################
# Task 3: Comparison and References
#################################

comparison_text = (
    "The word-frequency summarization method calculates the importance of a sentence by "
    "adding the frequencies of individual words after tokenization and stopword removal. "
    "This approach is simple and computationally efficient but may overlook contextual nuances. "
    "Conversely, the n-gram based method analyzes contiguous word groupings (such as trigrams), "
    "capturing local language structures and phrase patterns that provide richer context. "
    "However, n-gram analysis can be more sensitive to phrase repetition and may require tuning of "
    "the n parameter to achieve coherence. Both methods demonstrate unique advantages depending on the application."
)
print("Method Comparison:\n", comparison_text, "\n")

references_text = (
    "References:\n"
    "- NLTK Documentation and examples.\n"
    "- BeautifulSoup Documentation and examples.\n"
    "- Code tutorials and algorithms discussed in lecture slides."
)
print("References:\n", references_text)