
# Download required resources

""" import nltk

nltk.download('punkt_tab')
nltk.download('stopwords') """

import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Sample Text
text = "The movie was AMAZING!! I really loved it. It was one of the best movies I have ever watched :)"

print("Original Text:")
print(text)


# Text Cleaning

clean_text = re.sub(r'[^a-zA-Z\s]', '', text)

clean_text = clean_text.lower()

print("\nCleaned Text:")
print(clean_text)


# Tokenization

tokens = word_tokenize(clean_text)

print("\nTokens:")
print(tokens)


# Stop-word Removal

stop_words = set(stopwords.words('english'))

filtered_words = []

for word in tokens:
    if word not in stop_words:
        filtered_words.append(word)

print("\nAfter Stop-word Removal:")
print(filtered_words)


""" 
#Workflow

Original Text
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Stop-word Removal
      ↓
Processed Text

 """