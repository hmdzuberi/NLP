import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

raw_text = "Natural_language_processing is technology that allows @computer systems to #analyze and #understand normal human-language."

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(raw_text.lower())
tokens = [token for token in tokens if token not in stopwords.words('english')]