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
    Includes error handling.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        if not text:
            raise ValueError("No text content found")
        return text
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"Error processing content: {e}")
        return None

def clean_text(text):
    """
    Lowercase conversion, removal of extra spaces and non-alphabetic characters.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def tokenize_words(text):
    """
    Tokenizes text into words and removes punctuation and stopwords.
    """
    if not text:
        return []
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return words

def tokenize_sentences(text):
    """
    Splits the text into sentences.
    """
    if not text:
        return []
    sentences = nltk.sent_tokenize(text)
    return sentences

#################################
# Task 1: Summarization using Word Frequencies
#################################

# 1) Web scraping from the Wikipedia page
url = "https://en.wikipedia.org/wiki/Natural_language_processing"
raw_text = fetch_text(url)

if raw_text:
    # 2) Preprocessing - now using consistent text processing
    cleaned_text = clean_text(raw_text)
    words = tokenize_words(cleaned_text)
    sentences = tokenize_sentences(cleaned_text)  # Using cleaned_text for consistency

    # 3) Calculate word frequencies using NLTK's FreqDist
    freq_dist = nltk.FreqDist(words)

    # 4) Score sentences
    def calculate_sentence_scores(sent_list, freq_dist):
        scores = {}
        for sent in sent_list:
            score = 0
            word_tokens = nltk.word_tokenize(clean_text(sent))  # Clean each sentence
            for word in word_tokens:
                if word.isalpha() and word in freq_dist:
                    score += freq_dist[word]
            scores[sent] = score
        return scores

    sentence_scores = calculate_sentence_scores(sentences, freq_dist)
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    # 5) Build summaries using different criteria
    def summarize_by_sentence_count(ranked_sents, count):
        return ' '.join(ranked_sents[:min(count, len(ranked_sents))])

    def summarize_by_word_count(ranked_sents, word_limit):
        if not ranked_sents or word_limit <= 0:
            return ""
        summary = []
        total_words = 0
        for sent in ranked_sents:
            sent_words = nltk.word_tokenize(sent)
            if total_words + len(sent_words) <= word_limit:
                summary.append(sent)
                total_words += len(sent_words)
            else:
                break
        return ' '.join(summary)

    def summarize_by_percentage(ranked_sents, full_text, percentage):
        if not ranked_sents or not full_text or percentage <= 0:
            return ""
        total_words = len(nltk.word_tokenize(full_text))
        word_limit = int(total_words * (percentage / 100))
        return summarize_by_word_count(ranked_sents, word_limit)

    # Generate summaries
    summary_sentence_count = summarize_by_sentence_count(ranked_sentences, 3)
    summary_word_count = summarize_by_word_count(ranked_sentences, 50)
    summary_percentage = summarize_by_percentage(ranked_sentences, raw_text, 20)

    print("Summary (Sentence Count = 3):\n", summary_sentence_count, "\n")
    print("Summary (Word Count = 50):\n", summary_word_count, "\n")
    print("Summary (20% of Total Words):\n", summary_percentage, "\n")

    #################################
    # Task 2: Summarization using N-grams
    #################################

    def generate_ngrams(text, n):
        if not text or n < 1:
            return []
        tokens = nltk.word_tokenize(clean_text(text))
        return [' '.join(gram) for gram in ngrams(tokens, n)]

    def get_ngram_frequencies(text, n):
        ngram_list = generate_ngrams(text, n)
        return nltk.FreqDist(ngram_list)

    ngram_freqs = get_ngram_frequencies(cleaned_text, 3)

    def calculate_sentence_scores_ngram(sent_list, ngram_freqs, n):
        scores = {}
        for sent in sent_list:
            tokens = nltk.word_tokenize(clean_text(sent))
            sent_ngrams = [' '.join(gram) for gram in ngrams(tokens, n)]
            score = sum(ngram_freqs.get(gram, 0) for gram in sent_ngrams)
            scores[sent] = score
        return scores

    sentence_scores_ngram = calculate_sentence_scores_ngram(sentences, ngram_freqs, 3)
    ranked_sentences_ngram = sorted(sentence_scores_ngram, key=sentence_scores_ngram.get, reverse=True)

    ngram_summary_sentence_count = summarize_by_sentence_count(ranked_sentences_ngram, 3)
    ngram_summary_word_count = summarize_by_word_count(ranked_sentences_ngram, 50)
    ngram_summary_percentage = summarize_by_percentage(ranked_sentences_ngram, raw_text, 20)

    print("N-gram Summary (Sentence Count = 3):\n", ngram_summary_sentence_count, "\n")
    print("N-gram Summary (Word Count = 50):\n", ngram_summary_word_count, "\n")
    print("N-gram Summary (20% of Total Words):\n", ngram_summary_percentage, "\n")

    # Rest of the code remains the same...
else:
    print("Failed to fetch text from URL. Please check your internet connection and try again.")